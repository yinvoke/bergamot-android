// SDOT vs SMMLA micro-benchmark — the i8mm pilot for ruy-backend go/no-go.
//
// ruy's int8 kernels are hand-encoded SDOT (16 MACs/instruction). Armv8.6
// i8mm adds SMMLA (32 MACs/instruction): a potential 2x ALU ceiling on cores
// that issue both at the same rate. This tool measures, per core:
//   *_alu:    register-only throughput (no loads) — the ceiling.
//   *_stream: 2 loads per MAC instruction over L1-resident buffers — a floor
//             for what a packed GEMM inner loop would sustain.
// Reported as GMAC/s plus the smmla/sdot ratio. It decides whether writing
// ruy i8mm kernels (they do not exist upstream) is worth the effort; it is
// not itself a GEMM.
//
// Usage: i8mm-bench <core-id> — pins itself, runs both variants.
#include <arm_neon.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(__linux__)
#include <sched.h>
#include <sys/auxv.h>
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#endif

namespace {

double nowSeconds() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

bool pinTo(int core) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
  (void)core;
  return false;
#endif
}

// Keep the accumulators live without spilling: an empty asm that claims to
// read+write them defeats both dead-code elimination and loop-invariant
// hoisting of the MAC instructions.
#define KEEP8(a0, a1, a2, a3, a4, a5, a6, a7) \
  asm volatile("" : "+w"(a0), "+w"(a1), "+w"(a2), "+w"(a3), "+w"(a4), "+w"(a5), "+w"(a6), "+w"(a7))

constexpr long kAluIters = 64L * 1000 * 1000;
constexpr int kAluUnroll = 8;

// 8 independent accumulator chains, register operands only.
double sdotAluGmacs() {
  int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(-5);
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  double t0 = nowSeconds();
  for (long i = 0; i < kAluIters; ++i) {
    c0 = vdotq_s32(c0, a, b);
    c1 = vdotq_s32(c1, a, b);
    c2 = vdotq_s32(c2, a, b);
    c3 = vdotq_s32(c3, a, b);
    c4 = vdotq_s32(c4, a, b);
    c5 = vdotq_s32(c5, a, b);
    c6 = vdotq_s32(c6, a, b);
    c7 = vdotq_s32(c7, a, b);
    KEEP8(c0, c1, c2, c3, c4, c5, c6, c7);
  }
  double dt = nowSeconds() - t0;
  return double(kAluIters) * kAluUnroll * 16 / dt * 1e-9;  // 16 MACs per sdot
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
double smmlaAluGmacs() {
  int8x16_t a = vdupq_n_s8(3), b = vdupq_n_s8(-5);
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  double t0 = nowSeconds();
  for (long i = 0; i < kAluIters; ++i) {
    c0 = vmmlaq_s32(c0, a, b);
    c1 = vmmlaq_s32(c1, a, b);
    c2 = vmmlaq_s32(c2, a, b);
    c3 = vmmlaq_s32(c3, a, b);
    c4 = vmmlaq_s32(c4, a, b);
    c5 = vmmlaq_s32(c5, a, b);
    c6 = vmmlaq_s32(c6, a, b);
    c7 = vmmlaq_s32(c7, a, b);
    KEEP8(c0, c1, c2, c3, c4, c5, c6, c7);
  }
  double dt = nowSeconds() - t0;
  return double(kAluIters) * kAluUnroll * 32 / dt * 1e-9;  // 32 MACs per smmla
}
#endif

// Streaming variant: every MAC instruction is fed by two fresh L1 loads, the
// worst-case load:MAC ratio of a packed GEMM inner loop (real kernels reuse
// registers and do better; the two variants bracket a kernel's range).
constexpr int kStreamBytes = 16 * 1024;  // per operand buffer: fits 8 Gen 3 L1D (64K+)
constexpr long kStreamReps = 40L * 1000;

double sdotStreamGmacs(const int8_t *bufA, const int8_t *bufB) {
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  double t0 = nowSeconds();
  for (long r = 0; r < kStreamReps; ++r) {
    for (int off = 0; off < kStreamBytes; off += 8 * 16) {
      const int8_t *pa = bufA + off, *pb = bufB + off;
      c0 = vdotq_s32(c0, vld1q_s8(pa), vld1q_s8(pb));
      c1 = vdotq_s32(c1, vld1q_s8(pa + 16), vld1q_s8(pb + 16));
      c2 = vdotq_s32(c2, vld1q_s8(pa + 32), vld1q_s8(pb + 32));
      c3 = vdotq_s32(c3, vld1q_s8(pa + 48), vld1q_s8(pb + 48));
      c4 = vdotq_s32(c4, vld1q_s8(pa + 64), vld1q_s8(pb + 64));
      c5 = vdotq_s32(c5, vld1q_s8(pa + 80), vld1q_s8(pb + 80));
      c6 = vdotq_s32(c6, vld1q_s8(pa + 96), vld1q_s8(pb + 96));
      c7 = vdotq_s32(c7, vld1q_s8(pa + 112), vld1q_s8(pb + 112));
      KEEP8(c0, c1, c2, c3, c4, c5, c6, c7);
    }
  }
  double dt = nowSeconds() - t0;
  double macs = double(kStreamReps) * (kStreamBytes / 16) * 16;
  return macs / dt * 1e-9;
}

#if defined(__ARM_FEATURE_MATMUL_INT8)
double smmlaStreamGmacs(const int8_t *bufA, const int8_t *bufB) {
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
  double t0 = nowSeconds();
  for (long r = 0; r < kStreamReps; ++r) {
    for (int off = 0; off < kStreamBytes; off += 8 * 16) {
      const int8_t *pa = bufA + off, *pb = bufB + off;
      c0 = vmmlaq_s32(c0, vld1q_s8(pa), vld1q_s8(pb));
      c1 = vmmlaq_s32(c1, vld1q_s8(pa + 16), vld1q_s8(pb + 16));
      c2 = vmmlaq_s32(c2, vld1q_s8(pa + 32), vld1q_s8(pb + 32));
      c3 = vmmlaq_s32(c3, vld1q_s8(pa + 48), vld1q_s8(pb + 48));
      c4 = vmmlaq_s32(c4, vld1q_s8(pa + 64), vld1q_s8(pb + 64));
      c5 = vmmlaq_s32(c5, vld1q_s8(pa + 80), vld1q_s8(pb + 80));
      c6 = vmmlaq_s32(c6, vld1q_s8(pa + 96), vld1q_s8(pb + 96));
      c7 = vmmlaq_s32(c7, vld1q_s8(pa + 112), vld1q_s8(pb + 112));
      KEEP8(c0, c1, c2, c3, c4, c5, c6, c7);
    }
  }
  double dt = nowSeconds() - t0;
  double macs = double(kStreamReps) * (kStreamBytes / 16) * 32;
  return macs / dt * 1e-9;
}
#endif

}  // namespace

int main(int argc, char **argv) {
  int core = argc > 1 ? atoi(argv[1]) : -1;
  bool pinned = core >= 0 && pinTo(core);
  printf("core=%d pinned=%d\n", core, pinned ? 1 : 0);

#if defined(__linux__)
  bool hasI8mm = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
  bool hasI8mm = false;
#endif
  printf("hwcap2_i8mm=%d\n", hasI8mm ? 1 : 0);

  alignas(64) static int8_t bufA[kStreamBytes], bufB[kStreamBytes];
  for (int i = 0; i < kStreamBytes; ++i) {
    bufA[i] = int8_t(i * 7);
    bufB[i] = int8_t(i * 13 + 1);
  }

  // Warm the core up (frequency ramp) before the timed sections.
  sdotStreamGmacs(bufA, bufB);

  double sdotAlu = sdotAluGmacs();
  double sdotStream = sdotStreamGmacs(bufA, bufB);
  printf("sdot_alu_gmacs=%.1f\nsdot_stream_gmacs=%.1f\n", sdotAlu, sdotStream);

#if defined(__ARM_FEATURE_MATMUL_INT8)
  if (hasI8mm) {
    double smmlaAlu = smmlaAluGmacs();
    double smmlaStream = smmlaStreamGmacs(bufA, bufB);
    printf("smmla_alu_gmacs=%.1f\nsmmla_stream_gmacs=%.1f\n", smmlaAlu, smmlaStream);
    printf("ratio_alu=%.2f\nratio_stream=%.2f\n", smmlaAlu / sdotAlu, smmlaStream / sdotStream);
  } else {
    printf("smmla=skipped (no hwcap)\n");
  }
#else
  printf("smmla=skipped (built without +i8mm)\n");
#endif
  return 0;
}
