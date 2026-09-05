// See small_sgemm_neon.h for the contract and the bit-exactness argument.
#include "tensors/cpu/small_sgemm_neon.h"

#include <cmath>
#include <cstdlib>

#if defined(__aarch64__) || defined(_M_ARM64)
#define BG_SMALLGEMM_NEON 1
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace smallgemm {

#ifdef BG_SMALLGEMM_NEON

bool available() {
  static const bool ok = [] {
    const char* kill = std::getenv("BERGAMOT_NO_SMALLGEMM");
    return !(kill && kill[0] == '1');
  }();
  return ok;
}

// Where tools/float-bench says these kernels actually win:
//   M == 1 (the whole decode side)                       3.4x - 7.3x
//   transB == 0, N <= 64 (encoder A*V, N is head dim 48) ~1x
// The encoder's Q*K^T (transB == 1, M == N == srcLen) is NOT here: ruy's
// packed 8x8 kernel beats the on-the-fly 4x4 transpose by 1.4x - 1.7x there,
// so it stays on ruy.
bool shapeIsSmall(bool transB, int M, int N, int K) {
  (void)K;
  return M == 1 || (!transB && N <= 64);
}

namespace {

// --- reproducing ruy's accumulation order -----------------------------------
//
// KernelFloatNeon (kernel_arm64.cc:7703) holds an 8x8 tile of accumulators and
// is software-pipelined: the rows-0..3 / cols-0..3 quadrant (v16, v18, v20,
// v22) runs one depth level ahead of the other three quadrants. That quadrant
// therefore sees k in plain ascending order. The lagging quadrants normally do
// too -- their last depth level is simply applied later in program order, at
// the "79:" drain -- but once RUY_OPT(MAX_STREAMING) engages (depth >= 8) the
// streaming loop leaves them one level behind at the hand-over, and that level
// is the one the drain applies. With W = K & ~3 and D = W - 4, their order is
//
//     0 .. D-1,  D+1 .. W-1,  D,  W .. K-1
//
// i.e. exactly one interior level, D, deferred past the last full group of
// four. Below depth 8 there is no streaming and every quadrant is ascending.
// Verified against ruy on both Apple M3 and Cortex-A710 by tools/float-bench.
//
// Reproducing this is what makes these kernels bit-identical to the path they
// replace, which is what keeps tools/regress-hash.sh green without a COMET
// run. It is also the one place here that is coupled to ruy's kernel source:
// if that kernel or RUY_OPT_MAX_STREAMING ever changes, float_bench's
// "differs" column goes non-zero and says so.
inline int deferredDepth(int K) {
  return K >= 8 ? (K & ~3) - 4 : -1;  // -1: plain ascending
}

// True when an output element in row block mb (rows 4*mb..4*mb+3) and column
// block nb (cols 4*nb..4*nb+3) lands in ruy's leading quadrant. Tiles are 8x8
// and start at multiples of 8, so the quadrant is just the block parity.
inline bool leading(int mb, int nb) {
  return (mb & 1) == 0 && (nb & 1) == 0;
}

// One output element, ruy's order, scalar. bstep is the stride between
// successive k of B (1 when B is k-contiguous, ldb when it is n-contiguous).
inline float dotOrdered(const float* a, const float* b, int bstep, int K, int D) {
  float s = 0.f;
  if(D < 0) {
    for(int k = 0; k < K; k++) s = std::fma(a[k], b[(size_t)k * bstep], s);
    return s;
  }
  for(int k = 0; k < D; k++) s = std::fma(a[k], b[(size_t)k * bstep], s);
  for(int k = D + 1; k < D + 4; k++) s = std::fma(a[k], b[(size_t)k * bstep], s);
  s = std::fma(a[D], b[(size_t)D * bstep], s);
  for(int k = D + 4; k < K; k++) s = std::fma(a[k], b[(size_t)k * bstep], s);
  return s;
}

// ---------------------------------------------------------------------------
// transB == false: B is row-major K x N, so the N axis is contiguous and can
// be vectorized directly. acc[r][j] holds four outputs of row r; the k loop
// walks in ruy's order and every lane sees exactly one fused multiply-add per
// k. The quadrant, and with it the k order, changes every four columns, but
// the two orders differ ONLY in where step D sits relative to D+1..D+3 -- so
// the bulk of the k range still runs as one wide uniform loop and only that
// four-step window is split per column group.
// ---------------------------------------------------------------------------
template <int MR, int NV>
void blockNv(int nbase, int nb0, bool leadRows, int K, int D, int W, const float* A, int lda,
             const float* B, int ldb, float* C, int ldc) {
  float32x4_t acc[MR][NV];
  for(int r = 0; r < MR; r++)
    for(int j = 0; j < NV; j++) acc[r][j] = vdupq_n_f32(0.f);
  bool ascv[NV];
  for(int j = 0; j < NV; j++) ascv[j] = D < 0 || (leadRows && (((nb0 + j) & 1) == 0));

  auto step = [&](int k, int j) {
    const float32x4_t b = vld1q_f32(B + (size_t)k * ldb + nbase + 4 * j);
    for(int r = 0; r < MR; r++) acc[r][j] = vfmaq_n_f32(acc[r][j], b, A[(size_t)r * lda + k]);
  };

  const int kmain = D < 0 ? K : D;
  for(int k = 0; k < kmain; k++)
    for(int j = 0; j < NV; j++) step(k, j);
  if(D >= 0) {
    for(int j = 0; j < NV; j++) {
      if(ascv[j]) {
        step(D, j); step(D + 1, j); step(D + 2, j); step(D + 3, j);
      } else {
        step(D + 1, j); step(D + 2, j); step(D + 3, j); step(D, j);
      }
    }
    for(int k = W; k < K; k++)
      for(int j = 0; j < NV; j++) step(k, j);
  }
  for(int r = 0; r < MR; r++)
    for(int j = 0; j < NV; j++) vst1q_f32(C + (size_t)r * ldc + nbase + 4 * j, acc[r][j]);
}

template <int MR>
void blockN(bool leadRows, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
  const int D = deferredDepth(K);
  const int W = K & ~3;
  int n = 0;
  for(; n + 16 <= N; n += 16)
    blockNv<MR, 4>(n, n / 4, leadRows, K, D, W, A, lda, B, ldb, C, ldc);
  for(; n + 4 <= N; n += 4)
    blockNv<MR, 1>(n, n / 4, leadRows, K, D, W, A, lda, B, ldb, C, ldc);
  for(; n < N; n++) {
    const int d = (D < 0 || leading(leadRows ? 0 : 1, n / 4)) ? -1 : D;
    for(int r = 0; r < MR; r++)
      C[(size_t)r * ldc + n] = dotOrdered(A + (size_t)r * lda, B + n, ldb, K, d);
  }
}

// ---------------------------------------------------------------------------
// transB == true: both operands are K-contiguous, so four rows of B are
// transposed on the fly in blocks of four k. acc[r] holds four outputs of row
// r; the four vfmaq_laneq per block feed lanes k, k+1, k+2, k+3 in order.
// ruy's deferred level D is always the first of the last full group of four
// (D == W - 4), so reproducing its order costs exactly one thing here: that
// group runs its lanes 1, 2, 3, 0 instead of 0, 1, 2, 3.
// ---------------------------------------------------------------------------
template <int MR>
void blockT(bool leadRows, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
  const int D = deferredDepth(K);
  const int W = K & ~3;
  int n = 0;
  for(int nb = 0; n + 4 <= N; n += 4, nb++) {
    const bool asc = D < 0 || leading(leadRows ? 0 : 1, nb);
    float32x4_t acc[MR];
    for(int r = 0; r < MR; r++) acc[r] = vdupq_n_f32(0.f);
    const float* b0p = B + (size_t)(n + 0) * ldb;
    const float* b1p = B + (size_t)(n + 1) * ldb;
    const float* b2p = B + (size_t)(n + 2) * ldb;
    const float* b3p = B + (size_t)(n + 3) * ldb;
    for(int k = 0; k < W; k += 4) {
      const float32x4_t b0 = vld1q_f32(b0p + k);
      const float32x4_t b1 = vld1q_f32(b1p + k);
      const float32x4_t b2 = vld1q_f32(b2p + k);
      const float32x4_t b3 = vld1q_f32(b3p + k);
      const float32x4x2_t t0 = vtrnq_f32(b0, b1);
      const float32x4x2_t t1 = vtrnq_f32(b2, b3);
      const float32x4_t bt0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
      const float32x4_t bt1 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
      const float32x4_t bt2 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
      const float32x4_t bt3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
      if(asc || k != D) {
        for(int r = 0; r < MR; r++) {
          const float32x4_t a = vld1q_f32(A + (size_t)r * lda + k);
          acc[r] = vfmaq_laneq_f32(acc[r], bt0, a, 0);
          acc[r] = vfmaq_laneq_f32(acc[r], bt1, a, 1);
          acc[r] = vfmaq_laneq_f32(acc[r], bt2, a, 2);
          acc[r] = vfmaq_laneq_f32(acc[r], bt3, a, 3);
        }
      } else {
        for(int r = 0; r < MR; r++) {
          const float32x4_t a = vld1q_f32(A + (size_t)r * lda + k);
          acc[r] = vfmaq_laneq_f32(acc[r], bt1, a, 1);
          acc[r] = vfmaq_laneq_f32(acc[r], bt2, a, 2);
          acc[r] = vfmaq_laneq_f32(acc[r], bt3, a, 3);
          acc[r] = vfmaq_laneq_f32(acc[r], bt0, a, 0);
        }
      }
    }
    if(W != K) {
      // k >= W comes last in ruy's order too, and is short enough to finish
      // scalar. std::fma is the fused op, so these elements match as well.
      float tmp[MR][4];
      for(int r = 0; r < MR; r++) vst1q_f32(tmp[r], acc[r]);
      for(int k = W; k < K; k++)
        for(int r = 0; r < MR; r++)
          for(int c = 0; c < 4; c++)
            tmp[r][c] = std::fma(A[(size_t)r * lda + k], B[(size_t)(n + c) * ldb + k], tmp[r][c]);
      for(int r = 0; r < MR; r++) acc[r] = vld1q_f32(tmp[r]);
    }
    for(int r = 0; r < MR; r++) vst1q_f32(C + (size_t)r * ldc + n, acc[r]);
  }
  for(; n < N; n++) {
    const int d = (D < 0 || leading(leadRows ? 0 : 1, n / 4)) ? -1 : D;
    const float* b = B + (size_t)n * ldb;
    for(int r = 0; r < MR; r++)
      C[(size_t)r * ldc + n] = dotOrdered(A + (size_t)r * lda, b, 1, K, d);
  }
}

// One branch per four output rows, i.e. outside every hot loop.
template <int MR>
inline void block(bool transB, bool leadRows, int N, int K, const float* A, int lda,
                  const float* B, int ldb, float* C, int ldc) {
  if(transB)
    blockT<MR>(leadRows, N, K, A, lda, B, ldb, C, ldc);
  else
    blockN<MR>(leadRows, N, K, A, lda, B, ldb, C, ldc);
}

void runRows(bool transB, int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
  int m = 0, mb = 0;
  for(; m + 4 <= M; m += 4, mb++)
    block<4>(transB, (mb & 1) == 0, N, K, A + (size_t)m * lda, lda, B, ldb, C + (size_t)m * ldc, ldc);
  const bool lead = (mb & 1) == 0;
  switch(M - m) {
    case 3: block<3>(transB, lead, N, K, A + (size_t)m * lda, lda, B, ldb, C + (size_t)m * ldc, ldc); break;
    case 2: block<2>(transB, lead, N, K, A + (size_t)m * lda, lda, B, ldb, C + (size_t)m * ldc, ldc); break;
    case 1: block<1>(transB, lead, N, K, A + (size_t)m * lda, lda, B, ldb, C + (size_t)m * ldc, ldc); break;
    default: break;
  }
}

}  // namespace

bool sgemmSmall(bool transB,
                int M,
                int N,
                int K,
                float alpha,
                const float* A,
                int lda,
                const float* B,
                int ldb,
                float* C,
                int ldc) {
  if(!available()) return false;
  if(M <= 0 || N <= 0 || K <= 0) return false;
  // GemmRuy builds its ruy::Matrix with MakeSimpleLayout(), which derives the
  // strides from the dimensions and ignores lda/ldb/ldc entirely. Refuse
  // anything where that assumption would not hold rather than silently
  // computing something different from the path we are replacing.
  if(lda != K) return false;
  if(ldb != (transB ? K : N)) return false;
  if(ldc != N) return false;
  if(!shapeIsSmall(transB, M, N, K)) return false;

  runRows(transB, M, N, K, A, lda, B, ldb, C, ldc);

  if(alpha != 1.0f) {
    // Element for element what GemmRuy's post-pass does.
    const size_t size = (size_t)M * N;
    for(size_t i = 0; i < size; i++) C[i] = alpha * C[i];
  }
  return true;
}

#else  // no NEON: the engine keeps using sgemm() everywhere.

bool available() {
  return false;
}
bool shapeIsSmall(bool, int, int, int) {
  return false;
}
bool sgemmSmall(bool, int, int, int, float, const float*, int, const float*, int, float*, int) {
  return false;
}

#endif

}  // namespace smallgemm
}  // namespace cpu
}  // namespace marian
