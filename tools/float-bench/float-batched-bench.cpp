// Shape-level A/B for the attention float matmuls: what ProdBatchedOld does
// today (one ruy::Mul per batch item, via GemmRuy) against the small-GEMM
// NEON kernels, on the exact shapes transformer.h's two bdot() calls produce.
//
// Three arms per shape:
//   ruy    -- the current path, batchC calls to GemmRuy
//   fixed  -- the same batchC calls at K=1, i.e. ruy's per-call framework
//             cost with the arithmetic removed. This is the number the whole
//             experiment turns on.
//   small  -- the new kernels (tensors/cpu/small_sgemm_neon.*)
//
// Results are compared exactly, not with a tolerance: the kernels reproduce
// ruy's accumulation order (including the pipelining quirk FLOAT_BENCH_ORDER
// documents), so the "differs" column must read 0. A non-zero count means the
// kernels and ruy have drifted apart and the regression hash will move.
//
// Built as `float_bench` under -DBUILD_FLOAT_BENCH=ON (host and Android).
// On a phone pin it to one core class (taskset) and read scaling_max_freq and
// the board temperature around the run.
//   float_bench [reps-scale]      reps-scale > 1 lengthens every measurement
//   FLOAT_BENCH_ORDER=1           skip the timings, probe ruy's accumulation
//                                 order instead
//   FLOAT_BENCH_SWEEP=1           skip the timings, check bit-identity over
//                                 4000 random shapes; exit 1 on any mismatch
// Every buffer stays alive for the whole run so that no ruy cache keyed by
// pointer can be fooled by a recycled address (the float path sets no cache
// policy, but the bench should not depend on that).
#include "tensors/cpu/prod_blas.h"
#include "tensors/cpu/small_sgemm_neon.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <random>
#include <vector>

namespace small = marian::cpu::smallgemm;

namespace {

struct Case {
  const char* tag;
  int batch;  // B, sentences in the mini-batch
  int M, N, K;
  bool transB;
};

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Is ruy's float result reproducible as a single accumulator fed in ascending
// k with fused multiply-add? That was the premise this experiment started
// from. It holds below depth 8 and fails above it, because KernelFloatNeon is
// software-pipelined: the rows-0..3/cols-0..3 quadrant of each 8x8 tile runs
// one depth ahead, and once RUY_OPT(MAX_STREAMING) engages the other three
// quadrants apply one interior depth level after the last one.
void orderProbe() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  std::uniform_int_distribution<int> di(-8, 8);
  std::printf("--- ruy accumulation order probe (transB=1, N=24)\n");
  std::printf("%3s %3s  %-22s %s\n", "K", "M", "exact-arith wrong", "vs ascending-fma");
  for(int K : {4, 7, 8, 9, 16, 48}) {
    for(int M : {1, 8}) {
      const int N = 24;
      std::vector<float> A((size_t)M * K), B((size_t)N * K), C((size_t)M * N, 0.f);
      // (a) integer operands: every partial sum is exact, so any order agrees.
      for(auto& x : A) x = (float)di(rng);
      for(auto& x : B) x = (float)di(rng);
      GemmRuy(false, true, M, N, K, 1.f, A.data(), K, B.data(), K, 0.f, C.data(), N);
      int exactBad = 0;
      for(int m = 0; m < M; m++)
        for(int n = 0; n < N; n++) {
          double ref = 0;
          for(int k = 0; k < K; k++) ref += (double)A[(size_t)m * K + k] * B[(size_t)n * K + k];
          if((double)C[(size_t)m * N + n] != ref) exactBad++;
        }
      // (b) random operands: does ruy match an ascending fma chain?
      for(auto& x : A) x = d(rng);
      for(auto& x : B) x = d(rng);
      GemmRuy(false, true, M, N, K, 1.f, A.data(), K, B.data(), K, 0.f, C.data(), N);
      int ascBad = 0, ascBadLead = 0;
      for(int m = 0; m < M; m++)
        for(int n = 0; n < N; n++) {
          float s = 0.f;
          for(int k = 0; k < K; k++) s = std::fma(A[(size_t)m * K + k], B[(size_t)n * K + k], s);
          if(C[(size_t)m * N + n] != s) {
            ascBad++;
            if(m % 8 < 4 && n % 8 < 4) ascBadLead++;
          }
        }
      std::printf("%3d %3d  %4d/%-17d %4d/%-4d (%d in the leading quadrant)\n", K, M, exactBad,
                  M * N, ascBad, M * N, ascBadLead);
    }
  }
  std::printf("exact-arith wrong == 0 everywhere: ruy sums the right products,\n"
              "so every difference above is purely accumulation order.\n");
}

// Randomised shape sweep: every shape the engine's gate would actually hand
// to the small-GEMM kernels must come back bit-identical to ruy. Covers the
// awkward cases the fixed case table does not -- K and N not multiples of 4,
// M straddling ruy's 8-row tile boundary, K below the streaming threshold.
int bitSweep() {
  std::mt19937 rng(20260904);
  std::uniform_real_distribution<float> d(-1.f, 1.f);
  // M/N/K reach past max-length-break (128): encoder A*V has M = K = srcLen.
  std::uniform_int_distribution<int> dm(1, 136), dn(1, 160), dk(1, 160), db(0, 1);
  int taken = 0, bad = 0;
  for(int t = 0; t < 4000; t++) {
    const int M = (db(rng) ? 1 : dm(rng)), N = dn(rng), K = dk(rng);
    const bool transB = db(rng) != 0;
    const int lda = K, ldb = transB ? K : N, ldc = N;
    std::vector<float> A((size_t)M * K), B((size_t)N * K), C1((size_t)M * N, 1.f),
        C2((size_t)M * N, 2.f);
    for(auto& x : A) x = d(rng);
    for(auto& x : B) x = d(rng);
    const float alpha = (t % 3 == 0) ? 1.f : 1.0f / std::sqrt((float)48);
    if(!small::sgemmSmall(transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, C2.data(), ldc))
      continue;  // shape gate keeps it on ruy; nothing to compare
    taken++;
    GemmRuy(false, transB, M, N, K, alpha, A.data(), lda, B.data(), ldb, 0.f, C1.data(), ldc);
    if(std::memcmp(C1.data(), C2.data(), C1.size() * sizeof(float)) != 0) {
      if(++bad <= 5)
        std::printf("  MISMATCH M=%d N=%d K=%d transB=%d alpha=%g\n", M, N, K, (int)transB,
                    (double)alpha);
    }
  }
  std::printf("bit sweep: %d shapes taken by the fast path, %d not bit-identical to ruy\n", taken,
              bad);
  return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if(std::getenv("FLOAT_BENCH_ORDER")) {
    orderProbe();
    return 0;
  }
  if(std::getenv("FLOAT_BENCH_SWEEP")) return bitSweep();
  const double scale = argc > 1 ? std::atof(argv[1]) : 1.0;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> d(-1.f, 1.f);

  // Spin so a cold process gets migrated off the efficiency cores first.
  { volatile long x = 0; for(long i = 0; i < 800000000L; i++) x += i; }

  // heads is folded into batchC: one bdot is batchC = B * 8 small matmuls.
  const int heads = 8;
  const std::vector<Case> cases = {
      // encoder self-attention, srcLen 24 / 64 / 128
      {"enc QK^T  B=1  L=24", 1, 24, 24, 48, true},
      {"enc QK^T  B=21 L=24", 21, 24, 24, 48, true},
      {"enc AV    B=21 L=24", 21, 24, 48, 24, false},
      {"enc QK^T  B=8  L=64", 8, 64, 64, 48, true},
      {"enc AV    B=8  L=64", 8, 64, 48, 64, false},
      {"enc QK^T  B=4  L=128", 4, 128, 128, 48, true},
      {"enc AV    B=4  L=128", 4, 128, 48, 128, false},
      // decoder cross-attention, one step
      {"dec QK^T  B=1  L=24", 1, 1, 24, 48, true},
      {"dec AV    B=1  L=24", 1, 1, 48, 24, false},
      {"dec QK^T  B=21 L=24", 21, 1, 24, 48, true},
      {"dec AV    B=21 L=24", 21, 1, 48, 24, false},
      {"dec QK^T  B=8  L=64", 8, 1, 64, 48, true},
      {"dec AV    B=8  L=64", 8, 1, 48, 64, false},
      {"dec QK^T  B=4  L=128", 4, 1, 128, 48, true},
      {"dec AV    B=4  L=128", 4, 1, 48, 128, false},
  };

  std::deque<std::vector<float>> keepAlive;
  std::printf("%-22s %6s %10s %10s %10s  %7s %9s  %s\n", "shape", "batchC", "ruy", "fixed", "small",
              "fixed%", "speed-up", "differs");
  double geo = 0.0;
  int geoN = 0;

  for(const auto& c : cases) {
    const int batchC = c.batch * heads;
    const int lda = c.K;
    const int ldb = c.transB ? c.K : c.N;
    const int ldc = c.N;
    const float alpha = 1.0f / std::sqrt((float)48);

    keepAlive.emplace_back((size_t)batchC * c.M * c.K);
    auto& A = keepAlive.back();
    keepAlive.emplace_back((size_t)batchC * c.N * c.K);
    auto& B = keepAlive.back();
    for(auto& x : A) x = d(rng);
    for(auto& x : B) x = d(rng);
    keepAlive.emplace_back((size_t)batchC * c.M * c.N);
    auto& C1 = keepAlive.back();
    keepAlive.emplace_back((size_t)batchC * c.M * c.N);
    auto& C2 = keepAlive.back();

    const long long macs = (long long)batchC * c.M * c.N * c.K;
    const int reps = (int)((200000000.0 / (double)macs) * scale) + 3;

    const size_t strideA = (size_t)c.M * c.K;
    const size_t strideB = (size_t)c.N * c.K;
    const size_t strideC = (size_t)c.M * c.N;

    auto runRuy = [&](float* C) {
      for(int i = 0; i < batchC; i++)
        GemmRuy(false, c.transB, c.M, c.N, c.K, alpha, A.data() + i * strideA, lda,
                B.data() + i * strideB, ldb, 0.f, C + i * strideC, ldc);
    };
    // Same call count, same batch walk, K = 1: everything ruy does per call
    // except the arithmetic worth measuring.
    auto runFixed = [&](float* C) {
      for(int i = 0; i < batchC; i++)
        GemmRuy(false, c.transB, c.M, c.N, 1, alpha, A.data() + i * strideA, 1,
                B.data() + i * strideB, c.transB ? 1 : c.N, 0.f, C + i * strideC, ldc);
    };
    auto runSmall = [&](float* C) {
      for(int i = 0; i < batchC; i++)
        if(!small::sgemmSmall(c.transB, c.M, c.N, c.K, alpha, A.data() + i * strideA, lda,
                              B.data() + i * strideB, ldb, C + i * strideC, ldc))
          return false;
      return true;
    };

    runRuy(C1.data());
    double t0 = now();
    for(int r = 0; r < reps; r++) runRuy(C1.data());
    const double rt = (now() - t0) / reps;

    runFixed(C2.data());
    t0 = now();
    for(int r = 0; r < reps; r++) runFixed(C2.data());
    const double ft = (now() - t0) / reps;

    std::memset(C2.data(), 0, C2.size() * sizeof(float));
    if(!runSmall(C2.data())) {
      std::printf("%-22s %6d %9.1fus %9.1fus %10s  %6.1f%% %9s  %s\n", c.tag, batchC, rt * 1e6,
                  ft * 1e6, "-", 100.0 * ft / rt, "-", "(off: shape gate keeps this on ruy)");
      continue;
    }
    t0 = now();
    for(int r = 0; r < reps; r++) runSmall(C2.data());
    const double st = (now() - t0) / reps;

    size_t diff = 0;
    double maxrel = 0;
    for(size_t i = 0; i < C1.size(); i++)
      if(C1[i] != C2[i]) {
        diff++;
        maxrel = std::max(
            maxrel, std::fabs((double)C1[i] - (double)C2[i]) / (std::fabs((double)C1[i]) + 1e-30));
      }
    std::printf("%-22s %6d %9.1fus %9.1fus %9.1fus  %6.1f%% %8.2fx  %zu/%zu maxrel=%.1e\n", c.tag,
                batchC, rt * 1e6, ft * 1e6, st * 1e6, 100.0 * ft / rt, rt / st, diff, C1.size(),
                maxrel);
    geo += std::log(rt / st);
    geoN++;
  }

  if(geoN)
    std::printf("%-22s %6s %10s %10s %10s  %7s %8.2fx  (over %d shapes actually taken)\n", "geomean",
                "", "", "", "", "", std::exp(geo / geoN), geoN);
  if(!small::available()) std::printf("NOTE: small-GEMM path unavailable (no NEON, or kill switch)\n");
  return 0;
}
