#pragma once
// Small float GEMM fast path for the attention batched matmuls.
//
// ProdBatchedOld() on ARM turns one bdot() into batchC separate ruy::Mul
// calls (prod.cpp, non-MKL branch -> prod_blas.h GemmRuy). Every one of those
// pays ruy's per-call framework cost (MakeTrMulParams, thread-count probe,
// FPCR denormal suppression, allocator reset, LHS+RHS pack) and then runs an
// 8x8 float kernel. The attention shapes are tiny -- head dim 48, and during
// decoding M == 1 -- so the framework cost and the 8x row padding dominate.
//
// This TU replaces those calls with straight-line NEON kernels. Contract:
//
//   C = alpha * op(A) * op(B), row-major, beta == 0 (the only case the
//   forward pass uses), transA == false, and the leading dimensions equal
//   the natural ones (lda == K, ldb == (transB ? K : N), ldc == N) -- which
//   is exactly what GemmRuy assumes too, since MakeSimpleLayout() ignores
//   the lda/ldb/ldc it is handed.
//
// Numerics are bit-identical to ruy, and getting there took reproducing one
// quirk of ruy's own kernel. KernelFloatNeon (kernel_arm64.cc:7703) is
// software-pipelined: the rows-0..3/cols-0..3 quadrant of each 8x8 tile runs
// one depth level ahead of the other three, and once RUY_OPT(MAX_STREAMING)
// engages (depth >= 8) the lagging quadrants defer one interior level past
// the last full group of four. So ruy is NOT a plain ascending fma chain, and
// the kernels here follow its order per quadrant instead (see the comment on
// deferredDepth() in the .cpp). Every output element still has a single
// accumulator and every step is a fused multiply-add; nothing splits the K
// reduction. The alpha rescale afterwards mirrors GemmRuy's post-pass element
// for element. tools/float-bench compares exactly and prints how many
// elements differ -- that count must stay zero; tools/regress-hash.sh is the
// engine-level gate.
//
// BERGAMOT_NO_SMALLGEMM=1 forces the ruy path (same-binary A/B), like
// BERGAMOT_NO_I8MM=1 does for the int8 kernels.
#include <cstddef>

namespace marian {
namespace cpu {
namespace smallgemm {

// True when this build has the NEON kernels and the kill switch is unset.
bool available();

// Returns false when the shape/layout is outside the contract above; the
// caller must then fall back to sgemm(). Never partially writes C in that
// case.
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
                int ldc);

// Shape gate, exposed so the microbench can report the same decision the
// engine makes. Assumes the layout contract already holds.
bool shapeIsSmall(bool transB, int M, int N, int K);

}  // namespace smallgemm
}  // namespace cpu
}  // namespace marian
