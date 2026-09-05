/*
 * This file follows intgemm and is a means of retrofitting ruy into the intgemm based wiring in
 * `intgemm_interface.h`. ruy is an inference backend used in tensorflow and android deployment and
 * has an optimized ARM backend for the multiply operations required. Optimized code for quantize,
 * unquantize, transpose are added separately to connect the multiply library to marian.
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "common/lifecycle.h"  // D0: prepacked-cache tracing
#include "integer_common.h"
#include "smmla_gemm.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#include "ruy/context.h"
#include "ruy/context_get_ctx.h"
#include "ruy/ctx.h"
#include "ruy/platform.h"
#include "ruy/prepacked_cache.h"
#include "ruy/system_aligned_alloc.h"
#pragma GCC diagnostic pop

#if RUY_PLATFORM_NEON
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace integer {

using Index = unsigned int;

/// PATCH A / D0: everything the int8 ruy GEMM keeps per thread. A Context owns
/// a thread pool, allocators and the prepacked-weight cache; rebuilding it per
/// matmul is pure overhead, and Context is not thread-safe, so thread_local is
/// the matching lifetime -- it maps 1:1 onto the AsyncService worker model.
/// Hoisted into a named accessor (D0) so an explicit release can clear the
/// cache from outside the GEMM. `inline` gives one instance per thread across
/// all translation units.
struct RuyThreadState {
  ruy::Context context;
  uint64_t seenGeneration = 0;
  /// This thread's share of lifecycle::ruyPrepackedBytes(). Subtracted here so
  /// the process-wide total drops when a worker thread exits.
  long long accountedPrepackedBytes = 0;
  ~RuyThreadState() {
    if (accountedPrepackedBytes != 0)
      lifecycle::ruyPrepackedBytes().fetch_sub(accountedPrepackedBytes, std::memory_order_relaxed);
  }
};

inline RuyThreadState &threadRuyState() {
  static thread_local RuyThreadState state;
  return state;
}

/// D0: drop THIS thread's ruy prepacked-weight cache. Only affects the calling
/// thread; the next GEMM re-packs, so this is a memory/latency trade only.
inline void releaseThreadRuyCache() {
  RuyThreadState &state = threadRuyState();
  state.context.ClearPrepackedCache();
  if (state.accountedPrepackedBytes != 0) {
    lifecycle::event("ruy_prepack_released bytes=%lld", state.accountedPrepackedBytes);
    lifecycle::ruyPrepackedBytes().fetch_sub(state.accountedPrepackedBytes, std::memory_order_relaxed);
    state.accountedPrepackedBytes = 0;
  }
  // We just cleared, so adopt the current generation: no point clearing an
  // empty cache again on the next GEMM.
  state.seenGeneration = prepackGeneration().load(std::memory_order_relaxed);
}

#if RUY_PLATFORM_NEON

/*
 * Optimized path using ARM NEON SIMD intrinsics. Currently only supports int8_t.
 */
inline void quantize(const float *input, int8_t *output, float scale, Index rows, Index width) {
  const float32x4_t *Input = reinterpret_cast<const float32x4_t *>(input);
  const float32x4_t *InputEnd = reinterpret_cast<const float32x4_t *>(input + rows * width);

  int8x8_t *Output = reinterpret_cast<int8x8_t *>(output);
  while(Input != InputEnd) {
    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_lo = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_lo = vcvtnq_s32_f32(scaledFloat_lo);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);   // VQMOVN.S32 d0,q0
    int16x4_t s16x4_lo = vqmovn_s32(scaledInt_lo);

    // Vector multiply by scalar
    // float32x4_t vmulq_n_f32(float32x4_t a, float32_t b);
    // VMUL.F32 q0,q0,d0[0]
    float32x4_t scaledFloat_hi = vmulq_n_f32(*Input++, scale);

    // Convert from float
    // int32x4_t  vcvtnq_s32_f32(float32x4_t a);
    // VCVT.S32.F32 q0, q0
    int32x4_t scaledInt_hi = vcvtnq_s32_f32(scaledFloat_hi);

    // Vector saturating narrow integer
    // int16x4_t  vqmovn_s32(int32x4_t a);
    // VQMOVN.S32 d0,q0
    int16x4_t s16x4_hi = vqmovn_s32(scaledInt_hi);

    // Combine two ints.
    // int16x8_t   vcombine_s16(int16x4_t low, int16x4_t high);
    int16x8_t s16x8 = vcombine_s16(s16x4_lo, s16x4_hi);

    // Vector saturating narrow integer
    int8x8_t s8x8 = vqmovn_s16(s16x8);

    *Output = s8x8;
    ++Output;
  };
}

inline void _transpose_16x16(const int8_t *src,
                             Index i,
                             Index j,
                             Index rows,
                             Index cols,
                             int8_t *dst) {
  // Implemented following the algorithm described in
  // https://stackoverflow.com/a/29587984/4565794
  //
  // permute n 32-bit rows
  // permute n 64-bit rows
  // ...
  // permute n simd_width/2-bit rows

  // clang-format off
    
    // Permute 8 8-bit rows.
    // Load int8x16x2 from memory into SIMD registers, transpose as 2x2 matrices.

    Index srcRowBegin = i*cols + j;
    int8x16x2_t r0 = vtrnq_s8(vld1q_s8(&src[ 0*cols + srcRowBegin]), vld1q_s8(&src[ 1*cols + srcRowBegin]));
    int8x16x2_t r1 = vtrnq_s8(vld1q_s8(&src[ 2*cols + srcRowBegin]), vld1q_s8(&src[ 3*cols + srcRowBegin]));
    int8x16x2_t r2 = vtrnq_s8(vld1q_s8(&src[ 4*cols + srcRowBegin]), vld1q_s8(&src[ 5*cols + srcRowBegin]));
    int8x16x2_t r3 = vtrnq_s8(vld1q_s8(&src[ 6*cols + srcRowBegin]), vld1q_s8(&src[ 7*cols + srcRowBegin]));
    int8x16x2_t r4 = vtrnq_s8(vld1q_s8(&src[ 8*cols + srcRowBegin]), vld1q_s8(&src[ 9*cols + srcRowBegin]));
    int8x16x2_t r5 = vtrnq_s8(vld1q_s8(&src[10*cols + srcRowBegin]), vld1q_s8(&src[11*cols + srcRowBegin]));
    int8x16x2_t r6 = vtrnq_s8(vld1q_s8(&src[12*cols + srcRowBegin]), vld1q_s8(&src[13*cols + srcRowBegin]));
    int8x16x2_t r7 = vtrnq_s8(vld1q_s8(&src[14*cols + srcRowBegin]), vld1q_s8(&src[15*cols + srcRowBegin]));


    // Permute 8 16-bit rows.
    // Next step is to treat the entries as int16x8x2 (via cast) and do
    // transpose for int16, which will now leave intra-2 pairs intact while
    // transposing inter 2-pairs into the right places.
    int16x8x2_t t0 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[0]), vreinterpretq_s16_s8(r1.val[0]));
    int16x8x2_t t1 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[0]), vreinterpretq_s16_s8(r3.val[0]));
    int16x8x2_t t2 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[0]), vreinterpretq_s16_s8(r5.val[0]));
    int16x8x2_t t3 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[0]), vreinterpretq_s16_s8(r7.val[0]));
    int16x8x2_t t4 = vtrnq_s16(vreinterpretq_s16_s8(r0.val[1]), vreinterpretq_s16_s8(r1.val[1]));
    int16x8x2_t t5 = vtrnq_s16(vreinterpretq_s16_s8(r2.val[1]), vreinterpretq_s16_s8(r3.val[1]));
    int16x8x2_t t6 = vtrnq_s16(vreinterpretq_s16_s8(r4.val[1]), vreinterpretq_s16_s8(r5.val[1]));
    int16x8x2_t t7 = vtrnq_s16(vreinterpretq_s16_s8(r6.val[1]), vreinterpretq_s16_s8(r7.val[1]));

    // Permute 8 32-bit rows.
    int32x4x2_t x0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]), vreinterpretq_s32_s16(t1.val[0]));
    int32x4x2_t x1 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[0]), vreinterpretq_s32_s16(t5.val[0]));
    int32x4x2_t x2 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]), vreinterpretq_s32_s16(t1.val[1]));
    int32x4x2_t x3 = vtrnq_s32(vreinterpretq_s32_s16(t4.val[1]), vreinterpretq_s32_s16(t5.val[1]));

    int32x4x2_t x4 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[0]), vreinterpretq_s32_s16(t3.val[0]));
    int32x4x2_t x5 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[0]), vreinterpretq_s32_s16(t7.val[0]));
    int32x4x2_t x6 = vtrnq_s32(vreinterpretq_s32_s16(t2.val[1]), vreinterpretq_s32_s16(t3.val[1]));
    int32x4x2_t x7 = vtrnq_s32(vreinterpretq_s32_s16(t6.val[1]), vreinterpretq_s32_s16(t7.val[1]));

    // There is no permute 8 64-bit rows available. 
    // Instead we follow extracting low and high and placing them into the right places.
    Index dstRowBegin = j*rows + i;
    vst1q_s8(&dst[ 0*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[0]),  vget_low_s32(x4.val[0])))); 
    vst1q_s8(&dst[ 1*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[0]),  vget_low_s32(x5.val[0]))));
    vst1q_s8(&dst[ 2*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[0]),  vget_low_s32(x6.val[0]))));
    vst1q_s8(&dst[ 3*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[0]),  vget_low_s32(x7.val[0]))));
    vst1q_s8(&dst[ 4*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x0.val[1]),  vget_low_s32(x4.val[1]))));
    vst1q_s8(&dst[ 5*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x1.val[1]),  vget_low_s32(x5.val[1]))));
    vst1q_s8(&dst[ 6*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x2.val[1]),  vget_low_s32(x6.val[1]))));
    vst1q_s8(&dst[ 7*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32( vget_low_s32(x3.val[1]),  vget_low_s32(x7.val[1]))));

    vst1q_s8(&dst[ 8*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[0]), vget_high_s32(x4.val[0]))));
    vst1q_s8(&dst[ 9*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[0]), vget_high_s32(x5.val[0]))));
    vst1q_s8(&dst[10*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[0]), vget_high_s32(x6.val[0]))));
    vst1q_s8(&dst[11*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[0]), vget_high_s32(x7.val[0]))));
    vst1q_s8(&dst[12*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x0.val[1]), vget_high_s32(x4.val[1]))));
    vst1q_s8(&dst[13*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x1.val[1]), vget_high_s32(x5.val[1]))));
    vst1q_s8(&dst[14*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x2.val[1]), vget_high_s32(x6.val[1]))));
    vst1q_s8(&dst[15*rows + dstRowBegin], vreinterpretq_s8_s32(vcombine_s32(vget_high_s32(x3.val[1]), vget_high_s32(x7.val[1]))));

  // clang-format on
}

// Specialization for int8_t
inline void transpose(const int8_t *input, Index rows, Index cols, int8_t *output) {
  constexpr Index tile_size = 16;
  // assert(rows % tile_size == 0 && cols & tile_size == 0);
  for(Index i = 0; i < rows; i += tile_size) {
    for(Index j = 0; j < cols; j += tile_size) {
      _transpose_16x16(input, i, j, rows, cols, output);
    }
  }
}

#endif

/*
 * The following nomenclature comes from intgemm. The current state of code is to keep the
 * intgemm_interface.h diff minimal. There are possibly better abstractions.
 */

    static void SelectColumnsB(const int8_t *input,
                               int8_t *output,
                               Index width,
                               const Index *cols,
                               const Index *cols_end) {
      // B_prepared is expected to be col-major, for our implementation via ruy. If
      // col-major we can memcpy the respective column entries as they're
      // sequential. There are width=rows entries.
      Index num_cols = static_cast<Index>(std::distance(cols, cols_end));
      for(Index c = 0; c < num_cols; ++c) {
        std::memcpy(&(output[c * width]), &(input[cols[c] * width]), width);
      }
    }


    static void Multiply8Rui(const int8_t *input_A_prepared,
                         const int8_t *input_B_prepared,
                         float *output,
                         Index rows_A,
                         Index width,
                         Index cols_B,
                         float unquant_multiplier,
                         float * bias=nullptr,
                         bool cacheableB=false) {
      // It is expected that somehow we have managed to call all prepare by the time
      // we are here, with inputs (prepared) in int8_t. All that's left to do is use
      // ruy for multiply and then start with the reverse ops to get to fp32.

      // Use ruy to multiply.
      // The following is adapted from
      // https://github.com/google/ruy/blob/878283640de7946a43053e8ebf4f15114fbc9156/example/example.cc#L129-L152

      // PATCH A / D0: the one ruy::Context per thread (see threadRuyState()).
      RuyThreadState &state = threadRuyState();
      ruy::Context &context = state.context;
      long long &accountedPrepackedBytes = state.accountedPrepackedBytes;

      // PATCH B: drop this thread's prepacked weights when any model has been
      // torn down since we last packed. Cheap: one relaxed atomic load per
      // GEMM, and the clear only ever runs on the generation change.
      const uint64_t generation =
          prepackGeneration().load(std::memory_order_relaxed);
      if (generation != state.seenGeneration) {
        context.ClearPrepackedCache();
        if (lifecycle::enabled()) {
          lifecycle::event("ruy_prepack_cleared bytes=%lld generation=%llu->%llu",
                           accountedPrepackedBytes,
                           static_cast<unsigned long long>(state.seenGeneration),
                           static_cast<unsigned long long>(generation));
          lifecycle::ruyPrepackedBytes().fetch_sub(accountedPrepackedBytes,
                                                   std::memory_order_relaxed);
          accountedPrepackedBytes = 0;
        }
        state.seenGeneration = generation;
      }

      // PATCH C: on CPUs with i8mm (Armv9 cores) the int8 GEMM runs the
      // SMMLA kernel in smmla_gemm.cpp -- same integer math, bit-identical
      // accumulators, measured 1.15-1.4x end to end on 8 Gen 1 / 8 Gen 3.
      // Weights are prepacked per thread under the same generation guard as
      // the ruy cache above. Other CPUs take the ruy path below unchanged;
      // BERGAMOT_NO_I8MM=1 forces it everywhere (same-binary A/B, diagnostics).
      if (smmla::available()) {
        smmla::gemm8(input_A_prepared, input_B_prepared,
                     reinterpret_cast<std::int32_t *>(output),
                     static_cast<int>(rows_A), static_cast<int>(width),
                     static_cast<int>(cols_B), cacheableB, generation);
      } else {

      ruy::Matrix<std::int8_t> lhs;
      ruy::MakeSimpleLayout(rows_A, width, ruy::Order::kRowMajor, lhs.mutable_layout());
      lhs.set_data(input_A_prepared);

      ruy::Matrix<std::int8_t> rhs;
      ruy::MakeSimpleLayout(width, cols_B, ruy::Order::kColMajor, rhs.mutable_layout());
      rhs.set_data(input_B_prepared);
      // PATCH B: B holds constant model weights whenever the caller says so, so
      // ruy may keep the packed form across calls instead of re-packing every
      // GEMM. Because dst is row-major, ruy transposes internally and this
      // operand lands on Side::kLhs -- the policy travels with the matrix, so
      // setting it here is correct. A shortlist-selected B (RuySelectColumnsB)
      // reuses one buffer with changing contents and must stay uncached; the
      // cache key is only {data pointer, packed layout, zero point} and would
      // not notice the change.
      if (cacheableB) {
        rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
      }

      ruy::Matrix<std::int32_t> dst;
      ruy::MakeSimpleLayout(rows_A, cols_B, ruy::Order::kRowMajor, dst.mutable_layout());

      std::int32_t *dest_ptr = reinterpret_cast<std::int32_t *>(output);
      dst.set_data(dest_ptr);

      // When Dst is int32, mul_params is unused.
      ruy::MulParams<std::int32_t, std::int32_t> mul_params;
      ruy::Mul(lhs, rhs, mul_params, &context, &dst);

      // D0: refresh this thread's share of the global prepacked-byte total.
      // Only under the lifecycle gate -- it costs a cache read per GEMM.
      if (lifecycle::enabled()) {
        const long long bytes = static_cast<long long>(
            ruy::get_ctx(&context)->GetPrepackedCache()->BuffersBytes());
        if (bytes != accountedPrepackedBytes) {
          lifecycle::ruyPrepackedBytes().fetch_add(bytes - accountedPrepackedBytes,
                                                   std::memory_order_relaxed);
          accountedPrepackedBytes = bytes;
        }
      }

      }  // PATCH C: end of the ruy path

      // Unquantise:
      float32x4_t multiplier = vdupq_n_f32(unquant_multiplier);

      if (!bias) {
        for (Index i = 0; i < rows_A*cols_B; i+=4) {
          const int32x4_t *in = reinterpret_cast<const int32x4_t *>(&output[i]);
          float32x4_t *out = reinterpret_cast<float32x4_t *>(&output[i]);
          *out = vmulq_f32(vcvtq_f32_s32(*in), multiplier);
        }
      } else {
        for (Index i = 0; i < rows_A; i++) {
          for (Index j = 0; j < cols_B; j+=4) {
            const float32x4_t *mybias = reinterpret_cast<const float32x4_t *>(&bias[j]);
            const int32x4_t *in = reinterpret_cast<const int32x4_t *>(&output[i*cols_B + j]);
            float32x4_t *out = reinterpret_cast<float32x4_t *>(&output[i*cols_B + j]);
            *out = vaddq_f32(vmulq_f32(vcvtq_f32_s32(*in), multiplier), *mybias);
          }
        }
      }
    }

  static float MaxAbsolute(const float *begin, const float *end) {
    float result = 0;
    for(auto p = begin; p < end; ++p) {
      result = std::max(result, std::abs(*p));
    }
    return result;
  }


struct PrepareNode : public NaryNodeOp {
float quantMult_;
bool isB_;
bool transpose_;
  PrepareNode(Expr input, Expr quant_mult, bool transposeme=false, bool isB=false)
      : NaryNodeOp({input, quant_mult}, newShape(input, transposeme), Type::intgemm8), isB_(isB),
      // Since we are doing RowM x ColM, we should ALWAYS transpose B when prepare
      // UNLESS it's already transposed (eg the output layer).
      // On the other hand transposing A should work as normal
      transpose_(isB_ ? !transposeme : transposeme) {

    setMemoize(isB_); // Only Memoise if this is a BNodeOp
    set_name(input->name());

    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "A cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of A cannot be null");
  }

  NodeOps forwardOps() override {
    return { [=]() {
      quantMult_ = *child(1)->val()->data();
      countWembOp(isB_ ? "prepare_b_quantize" : "prepare_a_quantize",
                  (size_t)child(0)->val()->shape().elements());
      quantize(child(0)->val()->data(), /*input*/
                val_->data<int8_t>(), /*output*/
                *child(1)->val()->data(), /*Quant Mult*/
                rows(child(0)->val()),
                cols(child(0)->val()));

      if (transpose_) {
        int myrows = rows(child(0)->val());
        int mycols = cols(child(0)->val());
        auto allocator = New<TensorAllocator>(graph()->getBackend());

        Tensor temp;
        allocator->allocate(temp, shape(), Type::int8);

        transpose(val_->data<int8_t>(), myrows, mycols, temp->data<int8_t>());
        std::memcpy(val_->data<int8_t>(), temp->data<int8_t>(), sizeof(int8_t)*myrows*mycols);
        allocator->free(temp);
      }
    }};
  }

  static Shape newShape(Expr input, bool transposed) {
    Shape ret = input->shape();
    if (transposed) {
      ret.set(0, input->shape()[-1]);
      ret.set(1, input->shape()[0]);
    } else {
      ret = input->shape();
    }
    return ret;
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isB_){
      return "RuyPrepareB";
    } else {
      return "RuyPrepareA";
    }
  }
};

struct SelectColumnsBRuyNodeOp : public NaryNodeOp {
public:
  float quantMult_;
  SelectColumnsBRuyNodeOp(Expr input, const std::vector<uint_least32_t>  &indices)
      : NaryNodeOp({input}, newShape(input, indices), input->value_type()),  indices_(indices) {

    set_name(input->name());
    setMemoize(false); // Enabling memoization leads to a massive memory leak.
                       // Still, I don't understand why setMemoize(true) still leaks.
    // Check if arguments are not null
    ABORT_IF(child(0) == nullptr, "B cannot be null");
  }

  // PATCH D1: variant that selects straight out of an already-quantised model
  // parameter (an int8 Wemb, reshaped to the [dim x vocab] column-major view
  // ruy wants). There is no PrepareB node to take the multiplier from, so it
  // arrives as an explicit second child.
  SelectColumnsBRuyNodeOp(Expr input, Expr quantMult, const std::vector<uint_least32_t>  &indices)
      : NaryNodeOp({input, quantMult}, newShape(input, indices), input->value_type()), indices_(indices) {

    set_name(input->name());
    setMemoize(false);
    ABORT_IF(child(0) == nullptr, "B cannot be null");
    ABORT_IF(child(1) == nullptr, "Quant mult of B cannot be null");
    ABORT_IF(!isIntgemm(input->value_type()),
             "SelectColumnsB with an explicit quant mult expects an already quantised B");
  }

  NodeOps forwardOps() override {
    return { [=]() {
      //We get the quantization multiplier from a PrepareB, an explicit child, or directly from the input
      if (children().size() == 2) {
        quantMult_ = *child(1)->val()->data();
      } else if (child(0)->type() == "RuyPrepareB") {
        auto bPreppedNode = std::static_pointer_cast<PrepareNode>(child(0));
        quantMult_ = bPreppedNode->quantMult_;
      } else {
        ABORT("We should not be here");
      }
      auto input = child(0)->val();
      SelectColumnsB(reinterpret_cast<int8_t *>(input->data()),
                    val_->data<int8_t>(),
                    rows(input),
                    indices_.data(),
                    indices_.data()+indices_.size());
    }};
  }

  const std::string type() override { return "RuySelectColumnsB"; }

  size_t hash() override {
    if (!hash_) {
      hash_ = NaryNodeOp::hash();
      for(auto i : indices_)
        util::hash_combine(hash_, i);
    }
    return hash_;
  }

  bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node)) return false;
    auto cnode = std::dynamic_pointer_cast<SelectColumnsBRuyNodeOp>(node);
    if (!cnode) return false;
    return indices_ == cnode->indices_;
  }

private:
  static Shape newShape(Expr a, const std::vector<uint_least32_t>& indices) {
    Shape ret = a->shape();
    ret.set(1, indices.size());
    return ret;
  }

  std::vector<uint_least32_t> indices_;
};

struct QuantMultRuyNodeOp : public UnaryNodeOp {
  bool isB_;
  QuantMultRuyNodeOp(Expr input, bool isB, std::string bname) : UnaryNodeOp(input, Shape({1}), Type::float32), isB_(isB) {
    if (isB_) {
      set_name(input->name() + "_QuantMultB");
    } else {
      setMemoize(false);
      set_name(bname + "_QuantMultA");
    }
  }
#pragma warning(push)
#pragma warning(disable: 4127) //VSCODE thinks line 222 is constant conditional expression, which it is only after the template resolution, not before.
  NodeOps forwardOps() override {
    return  {[=](){
      if (child(0)->type() == "RuySelectColumnsB") { // @TODO
        *val_->data() = std::static_pointer_cast<SelectColumnsBRuyNodeOp>(child(0))->quantMult_;
      } else if (isIntgemm(child(0)->value_type())) {                    /* So we can template*/
        *val_->data() = *(reinterpret_cast<float *>(reinterpret_cast<int8_t *>(child(0)->val()->data()) + child(0)->val()->shape().elements()));
      } else {
        auto input = child(0)->val();
        countWembOp(isB_ ? "max_absolute_b" : "max_absolute_a", (size_t)input->size());
        *val_->data() = 127.0f / MaxAbsolute(input->data(), input->data() + input->size());
      }
    }};
  }
#pragma warning(pop)
  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override {
    if (isB_)
      return "RuyQuantMultB";
    else
      return "RuyQuantMultA";
  }

  /* @TODO This is not correct in the case of none-static alphas but we are leaving it for now to battle memory leaks. */
  bool equal(Expr node) override {
    if (!isB_) {
      return UnaryNodeOp::equal(node);
    }
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

};

/*
 * PATCH D1: B multiplier for an embedding table that stayed int8.
 *
 * The FP32 path derived it as 127 / MaxAbsolute(table) over the *dequantised*
 * table, i.e. over int8[i] * (1/q). Because 1/q > 0, that maximum is exactly
 * max|int8| * (1/q) in float32, so the value can be reproduced bit for bit from
 * the quantised bytes alone -- no 4x-sized FP32 copy needed. Reproducing it
 * (rather than using the stored q) matters: on jaen the two differ by 2 ulp,
 * which would perturb every logit.
 *
 * Memoised like QuantMultRuyNodeOp's B case: one pass over the table per graph.
 */
struct WembQuantMultNodeOp : public UnaryNodeOp {
  WembQuantMultNodeOp(Expr input) : UnaryNodeOp(input, Shape({1}), Type::float32) {
    set_name(input->name() + "_QuantMultBWemb");
    ABORT_IF(!isIntgemm(input->value_type()), "WembQuantMult expects a quantised table");
  }

  NodeOps forwardOps() override {
    return {[=]() {
      auto input = child(0)->val();
      const int8_t *data = reinterpret_cast<const int8_t *>(input->data());
      const size_t n = (size_t)input->shape().elements();
      const float storedQuantMult = *(reinterpret_cast<const float *>(data + n));

      int maxAbs = 0;
      for(size_t i = 0; i < n; ++i) {
        int v = data[i] < 0 ? -(int)data[i] : (int)data[i];
        if(v > maxAbs)
          maxAbs = v;
      }
      countWembOp("wemb_max_absolute_int8", n);
      const float maxAbsolute = (float)maxAbs * (1 / storedQuantMult);
      *val_->data() = 127.0f / maxAbsolute;

      if(wembCheckEnabled())
        selfCheck(name(), data, n, storedQuantMult, *val_->data());
    }};
  }

  // Runs the pipeline this node replaces -- dequantise the whole table, take
  // MaxAbsolute, requantise with the production NEON kernel -- and reports
  // whether the bytes and the multiplier come out identical.
  static void selfCheck(const std::string &name,
                        const int8_t *data,
                        size_t n,
                        float storedQuantMult,
                        float derivedQuantMult) {
    std::vector<float> dequantized(n);
    const float unquant = 1 / storedQuantMult;
    int maxAbs = 0;
    for(size_t i = 0; i < n; ++i) {
      dequantized[i] = data[i] * unquant;
      int v = data[i] < 0 ? -(int)data[i] : (int)data[i];
      if(v > maxAbs)
        maxAbs = v;
    }
    float maxAbsolute = MaxAbsolute(dequantized.data(), dequantized.data() + n);
    float requantMult = 127.0f / maxAbsolute;
    std::vector<int8_t> requantized(n);
    quantize(dequantized.data(), requantized.data(), requantMult, 1, (Index)n);
    size_t mismatches = 0;
    for(size_t i = 0; i < n; ++i)
      if(requantized[i] != data[i])
        ++mismatches;
    uint32_t storedBits, requantBits, derivedBits;
    std::memcpy(&storedBits, &storedQuantMult, sizeof(storedBits));
    std::memcpy(&requantBits, &requantMult, sizeof(requantBits));
    std::memcpy(&derivedBits, &derivedQuantMult, sizeof(derivedBits));
    // stderr, not LOG(): the bergamot configs run with quiet: true.
    std::fprintf(stderr,
                 "[wemb-check] %s elements=%zu max_abs_int8=%d "
                 "stored_q=%.17g (0x%08x) fp32_path_q=%.17g (0x%08x) derived_q=%.17g (0x%08x) "
                 "q_bits_equal=%d requant_mismatches=%zu\n",
                 name.c_str(), n, maxAbs, (double)storedQuantMult, storedBits, (double)requantMult,
                 requantBits, (double)derivedQuantMult, derivedBits,
                 requantBits == derivedBits ? 1 : 0, mismatches);
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "RuyQuantMultBWemb"; }

  bool equal(Expr node) override {
    if(hash() == node->hash())
      return true;
    return false;
  }

  size_t hash() override { return std::hash<std::string>{}(name()); }
};

/*
 * PATCH D2: gather-and-dequantise for an int8 embedding table.
 *
 * Replaces rows(E, idx) when E stayed quantised. Dequantises only the rows the
 * batch actually looks up, using the exact expression (and expression order)
 * of cpu::integer::unquantizeWemb, so the result is bit-identical to gathering
 * from the fully dequantised table.
 */
struct RowsDequantNodeOp : public NaryNodeOp {
  RowsDequantNodeOp(Expr table, Expr indices)
      : NaryNodeOp({table, indices}, newShape(table, indices), Type::float32) {
    matchOrAbort<IndexType>(indices->value_type());
    ABORT_IF(!isIntgemm(table->value_type()),
             "RowsDequantNodeOp expects a quantised embedding table, got {}", table->value_type());
    ABORT_IF(table->shape().size() != 2, "RowsDequantNodeOp expects a 2-dimensional table");
    set_name(table->name());
    setMemoize(false);
  }

  NodeOps forwardOps() override {
    return {[=]() {
      auto table = child(0)->val();
      auto indices = child(1)->val();
      const int8_t *src = reinterpret_cast<const int8_t *>(table->data());
      const size_t width = (size_t)table->shape()[-1];
      const size_t vocab = (size_t)table->shape()[0];
      const float quantMult
          = *(reinterpret_cast<const float *>(src + table->shape().elements()));
      // Same expression, same order, as unquantizeWemb: one reciprocal, then
      // one float32 multiply per element.
      const float unquant = 1 / quantMult;
      const IndexType *idx = indices->data<IndexType>();
      const size_t rows = (size_t)indices->shape().elements();
      float *dst = val_->data<float>();
      for(size_t r = 0; r < rows; ++r) {
        const size_t row = (size_t)idx[r];
        ABORT_IF(row >= vocab, "Embedding index {} out of range (vocab {})", row, vocab);
        const int8_t *in = src + row * width;
        float *out = dst + r * width;
        for(size_t c = 0; c < width; ++c)
          out[c] = in[c] * unquant;
      }
      if(wembCheckEnabled())
        check(name(), idx, rows, width, dst);
    }};
  }

  // Compares against the FP32 table the reference unquantizeWemb() produced at
  // load time (BERGAMOT_WEMB_CHECK=1 keeps one around).
  static void check(const std::string &name,
                    const IndexType *idx,
                    size_t rows,
                    size_t width,
                    const float *out) {
    const std::vector<float> *shadow = findWembShadow(stripParamNamespace(name));
    if(!shadow)
      return;
    size_t mismatches = 0;
    for(size_t r = 0; r < rows; ++r) {
      const float *ref = shadow->data() + (size_t)idx[r] * width;
      for(size_t c = 0; c < width; ++c) {
        uint32_t a, b;
        std::memcpy(&a, &out[r * width + c], sizeof(a));
        std::memcpy(&b, &ref[c], sizeof(b));
        if(a != b)
          ++mismatches;
      }
    }
    reportWembRowCheck(name, rows, rows * width, mismatches);
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "RuyRowsDequant"; }

private:
  static Shape newShape(Expr table, Expr indices) {
    Shape shape = table->shape();
    shape.set(0, (int)indices->shape().elements());
    return shape;
  }
};

class AffineOrDotNodeOp : public NaryNodeOp {

public:
  AffineOrDotNodeOp(Expr a, Expr b, Expr bias)
      : NaryNodeOp({a, b, bias}, newShape(a, b), Type::float32) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }

  AffineOrDotNodeOp(Expr a, Expr b)
      : NaryNodeOp({a, b}, newShape(a, b), Type::float32) {
        setMemoize(false); // AFAIK affine is never called with the same matrices
      }


  Shape newShape(Expr a, Expr b) {
    Shape result = a->shape();
    result.set(-1, b->shape()[-1]);
    return result;
  }

  NodeOps forwardOps() override {
    return { [=]() {
          float aQuantMult = std::static_pointer_cast<PrepareNode >(child(0))->quantMult_;
          float bQuantMult;
          // PATCH B: a RuySelectColumnsB child rewrites one reused buffer every
          // step (shortlist column selection), so its packed form must never be
          // cached. The other two cases -- a memoized RuyPrepareB and an already
          // prepared parameter tensor -- are constant for the model's lifetime.
          bool cacheableB = child(1)->type() != "RuySelectColumnsB";
          if (child(1)->type() == "RuySelectColumnsB") {
            bQuantMult = std::static_pointer_cast<SelectColumnsBRuyNodeOp>(child(1))->quantMult_;
          } else if (child(1)->type() == "RuyPrepareB") {
            bQuantMult = std::static_pointer_cast<PrepareNode>(child(1))->quantMult_;
          } else {
            // Fetch the quant mult directly in case of already prepared B
            bQuantMult = *(reinterpret_cast<float *>(reinterpret_cast<int8_t *>(child(1)->val()->data()) + child(1)->val()->shape().elements()));
          }
          float unquant_mult = 1.0f/(aQuantMult*bQuantMult);
          // The bias is either nullptr or the third child
          float * bias = nullptr;
          if (children().size() == 3) {
            bias = child(2)->val()->data();
          }
          Multiply8Rui(reinterpret_cast<const int8_t *>(child(0)->val()->data()), /*A*/
                                      reinterpret_cast<const int8_t *>(child(1)->val()->data()), /*B*/
                                      val_->data(), /*output*/
                                      rows(child(0)->val()),
                                      cols(child(0)->val()),
                                      cols(child(1)->val()),
                                      unquant_mult,
                                      bias,
                                      cacheableB);
    }};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "intgemmAffine"; }
};

static inline Expr affineOrDotRUI(Expr a, Expr b, Expr bias, bool transA, bool transB, float /*clipValue*/) {
    Type bElementType = b->value_type();
    Expr aQuantMult = nullptr;
    bool precomputedAlphas = b->graph()->getBackend()->isPrecomputedAlpha();
    if (precomputedAlphas) { //Shifting here maybe should check?
      aQuantMult = Expression<fetchAlphaFromModelNodeOp>(b);
    } else {
      aQuantMult = Expression<QuantMultRuyNodeOp>(a, false, b->name());
    }
    auto aQuant = Expression<PrepareNode>(a, aQuantMult, transA, false);
    Expr bQuantMult = Expression<QuantMultRuyNodeOp>(b, true, b->name());
    Expr bQuant = nullptr;
    if (isIntgemm(bElementType)) {
      //This is the case where we already run SelectColumnB or we loaded a prepacked model.
      bQuant = b;
    } else {
      bQuant = Expression<PrepareNode>(b, bQuantMult, transB, true);
    }
    if (bias) {
      return Expression<AffineOrDotNodeOp>(aQuant, bQuant, bias);
    } else {
      return Expression<AffineOrDotNodeOp>(aQuant, bQuant);
    }
}

}  // namespace integer
}  // namespace cpu
}  // namespace marian
