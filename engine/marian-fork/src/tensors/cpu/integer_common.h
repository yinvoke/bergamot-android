#pragma once

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "tensors/cpu/aligned.h"
#include "graph/expression_graph.h"
#include "graph/node.h"
#include "graph/node_operators_unary.h"
#include "common/io_item.h"
#ifdef USE_INTGEMM
#include "3rd_party/intgemm/intgemm/intgemm.h"
#else // USE_INTGEMM
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"
#include <ruy/ruy.h>
#pragma GCC diagnostic pop

#endif // USE_INTGEMM
#if defined(WASM)
#include "wasm_intgemm_interface.h"
#endif

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace marian {
namespace cpu {
namespace integer {

// PATCH B: staleness guard for ruy's per-thread prepacked-weight cache.
//
// The cache is keyed on {src data pointer, packed layout, zero point} and does
// NOT hash the buffer contents. A freed model whose weight allocation is later
// handed back for a different model would therefore silently hit a stale
// packed buffer. Every TranslationModel destruction bumps this counter; each
// worker thread compares it against the generation it last packed under and
// drops its whole prepacked cache on a mismatch.
//
// Relaxed ordering is sufficient: the value is only ever compared for
// inequality against a thread-local snapshot, and model teardown is already
// ordered against translation by the service's own synchronisation.
std::atomic<uint64_t> &prepackGeneration();
void bumpPrepackGeneration();

// D0: drop THIS thread's GEMM weight-packing caches -- ruy's prepacked cache
// and, on i8mm CPUs, the SMMLA packed-B cache and scratch buffers. They are
// thread_local, so a model teardown on another thread cannot free them; the
// generation guard above only reclaims them lazily, at the next GEMM this
// thread happens to run, which never comes if translation has stopped. The
// service's release path calls this on every worker.
//
// Correctness-neutral: the next GEMM on this thread re-packs from the model's
// own weights. Only affects the calling thread.
void releaseThreadPackingCaches();

// PATCH D1/D2: the embedding tables (Wemb) stay int8 in memory.
//
// Upstream dequantises every Wemb item at load time (binary.cpp ->
// unquantizeWemb), which costs 4x the bytes of the stored table and then makes
// the output layer quantise the very same numbers back to int8 on first use.
// Both are avoidable: the stored bytes are already exactly what both consumers
// want. These helpers gate the new path and carry the equivalence harness.
//
// BERGAMOT_FP32_WEMB=1  -> keep the old FP32-resident behaviour (kill switch,
//                          same binary, for A/B and rollback).
// BERGAMOT_WEMB_CHECK=1 -> additionally keep an FP32 shadow produced by the
//                          reference unquantizeWemb() and compare every
//                          dequantised embedding row against it, and run the
//                          old requantisation pipeline over the whole table to
//                          prove the output layer sees identical bytes.
bool wembKeepFp32();
bool wembCheckEnabled();

// Only the real [vocab x dim] tables qualify. Note that the "*.alphas" models
// also carry <name>_QuantMultA as a 1x1 *intgemm8* item whose dequantised value
// IS the alpha; those must keep going through unquantizeWemb.
bool isWembTableName(const std::string &name, size_t elements);

// Equivalence harness (only populated when BERGAMOT_WEMB_CHECK=1).
void registerWembShadow(const std::string &name, std::vector<float> &&table);
const std::vector<float> *findWembShadow(const std::string &name);
// Strips a graph namespace prefix ("F0::decoder_Wemb" -> "decoder_Wemb").
std::string stripParamNamespace(const std::string &name);
void reportWembRowCheck(const std::string &name, size_t rows, size_t elements, size_t mismatches);
// Counts the work the two paths do differently (load-time dequantisation,
// first-batch MaxAbsolute + requantisation). Reported at exit under the same
// env flag; a no-op otherwise.
void countWembOp(const char *what, size_t elements);

// Making sure we have access to common functions for RUY and INTGEMM
class fetchAlphaFromModelNodeOp : public UnaryNodeOp {
public:
  fetchAlphaFromModelNodeOp(Expr b)
      : UnaryNodeOp(b, Shape({1}), Type::float32) {

    std::string bname = b->name();
    std::string aQuantKey = b->name() + "_QuantMultA";
    //Very Hacky Bit. Unnamed matrix is notpart of the F0 parameter namespace
    if (aQuantKey.at(0) != 'F') {
      aQuantKey = "F0::" + aQuantKey;
    }
    set_name(aQuantKey);
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      auto map = child(0)->graph()->params()->getMap();
      const auto mapiter = map.find(name());
      if (mapiter != map.end()) {
        val_ = mapiter->second->val();
      } else {
        ABORT("We did not find an alpha in the model named: {}.", name());
      }
    )};
  }

  bool equal(Expr node) override {
    if(hash() == node->hash()) return true;
    return false;
  }

  size_t hash() override {
    return std::hash<std::string>{}(name());
  }

  const std::string type() override { return "alphaNodeOp"; }
};

//Convenient function to get rows and columns of a tensor, shadowed by namespace.
inline int cols(Tensor& tensor) { return tensor->shape()[-1]; }
inline int rows(Tensor& tensor) { return tensor->shape().elements() / cols(tensor); }

inline int cols(Shape& shape) { return shape[-1]; }
inline int rows(Shape& shape) { return shape.elements() / cols(shape); }

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias);

#ifdef USE_INTGEMM

template<Type type> struct intgemm_;
template <> struct intgemm_<Type::int8> {using width = intgemm::Int8;
                                         using type = int8_t;
                                         constexpr static const Type intgemmType = Type::intgemm8;};
template <> struct intgemm_<Type::int16> {using width = intgemm::Int16;
                                          using type = int16_t;
                                          constexpr static const Type intgemmType = Type::intgemm16;};



#else // USE_INTGEMM

struct fakeGemm {
  struct Int8 {};
  struct Int16 {};
};

template<Type type> struct intgemm_;
template <> struct intgemm_<Type::int8> {using width = fakeGemm::Int8;
                                         using type = int8_t;
                                         constexpr static const Type intgemmType = Type::intgemm8;};
template <> struct intgemm_<Type::int16> {using width = fakeGemm::Int16;
                                          using type = int16_t;
                                          constexpr static const Type intgemmType = Type::intgemm16;};

#endif // USE_INTGEMM

// For loading architecture agnostic models. We do PrepareAndTranpose, because we already transposed
// in our binary format. Then we copy the quantizationMultiplier information at the end
template<Type vtype>
void prepareAndTransposeB(io::Item& item, const char * input) {
#ifdef COMPILE_CPU
    typedef typename intgemm_<vtype>::type Integer;
    Integer * output_tensor = reinterpret_cast<Integer *>(&(*item.bytes->begin()));
#ifdef ARM
    // On arm we do RowM * ColM. Our input already comes transposed due to the way we prepare matrices in the binary
    // So on arm ALL we need to do is just copy. No need for pre
    std::memcpy(output_tensor, reinterpret_cast<const Integer *>(input), sizeof(Integer) * item.shape.elements());
#else
    // Sometimes we will end up with misaligned intput (and output) so we can't use them directly.
    // If this is the case, we will need to temporary allocate aligned memory, copy the results, and then free it
    if (reinterpret_cast<uintptr_t>(input) % 64 == 0 && reinterpret_cast<uintptr_t>(output_tensor) % 64 == 0) {
    #if defined(WASM)
        ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
                "Int16::PrepareBQuantizedTransposed is not implemented for wasm.");
        int8PrepareBFromQuantizedTransposed(reinterpret_cast<const int8_t *>(input),
                                        (Index)rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary 
                                        (Index)cols(item.shape), //rows here returns the columns of the transposed input matrix, and cols -> the rows
                                        (int8_t *)output_tensor);
    #elif defined(USE_INTGEMM)
        intgemm_<vtype>::width::PrepareBQuantizedTransposed(reinterpret_cast<const Integer *>(input),
                                                   output_tensor,
                                                   rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                                   cols(item.shape)); //rows here returns the columns of the transposed input matrix, and cols -> the rows
    #endif
    } else {
        Integer * aligned_input = reinterpret_cast<Integer *>(genericMalloc(512, rows(item.shape)*cols(item.shape)*sizeof(Integer)));
        std::copy(input, input + rows(item.shape)*cols(item.shape), aligned_input);
        Integer * aligned_output = reinterpret_cast<Integer *>(genericMalloc(512, rows(item.shape)*cols(item.shape)*sizeof(Integer)));
    #if defined(WASM)
        ABORT_IF(intgemm_<vtype>::intgemmType == Type::intgemm16,
                "Int16::PrepareBQuantizedTransposed is not implemented for wasm.");
        int8PrepareBFromQuantizedTransposed(reinterpret_cast<const int8_t *>(aligned_input),
                                        (Index)rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                        (Index)cols(item.shape), //rows here returns the columns of the transposed input matrix, and cols -> the rows
                                        reinterpret_cast<int8_t *>(aligned_output));
    #elif defined(USE_INTGEMM)
        intgemm_<vtype>::width::PrepareBQuantizedTransposed(reinterpret_cast<const Integer *>(aligned_input),
                                                   reinterpret_cast<Integer *>(aligned_output),
                                                   rows(item.shape),  //Since we only transposed, but didn't update the shape when constructing the binary, 
                                                   cols(item.shape)); //rows here returns the columns of the transposed input matrix, and cols -> the rows
    #endif
        // Copy to output tensor
        std::copy(aligned_output, aligned_output + rows(item.shape)*cols(item.shape), output_tensor);
        genericFree(aligned_input);
        genericFree(aligned_output);
    }
#endif
    //Copy the quantMult
    float quantMult = *(reinterpret_cast<const float *>(reinterpret_cast<const Integer *>(input) + item.shape.elements()));
    *(reinterpret_cast<float *>(&(*(output_tensor + item.shape.elements())))) = quantMult;
#else // COMPILE_CPU
    ABORT("Using intgemm models is supported only with -DCOMPILE_CPU=on");
#endif
}

template<Type vtype>
void unquantizeWemb(io::Item& item, const char * input) {
    typedef typename intgemm_<vtype>::type Integer;
    float quantMult = *(reinterpret_cast<const float *>(reinterpret_cast<const Integer *>(input) + item.shape.elements()));
    float * output_tensor = reinterpret_cast<float *>(&(*item.bytes->begin()));
    // Explicitly calculate n once beforehand because the compiler does not pick up on its
    // static nature, and will end up calling marian::Shape::dim() a lot.
    const size_t n = rows(item.shape) * cols(item.shape);
    for (size_t i = 0; i < n; i++) {
        output_tensor[i] = reinterpret_cast<const Integer *>(input)[i]*(1/quantMult);
    }
}

} //integer
} //cpu
} //marian
