// SMMLA int8 GEMM. This TU is compiled with -march=...+i8mm (see
// src/CMakeLists.txt); everything here runs only after smmla::available()
// said yes, so no other TU may call the kernels directly. The i8mm opcodes
// must stay confined to this TU's symbols -- tools/smmla-test/check-opcodes.sh
// enforces that on the linked binary.
#include "smmla_gemm.h"

#if defined(__aarch64__)

#include <arm_neon.h>

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "common/lifecycle.h"  // D0: packing-cache tracing

#if defined(__linux__)
#include <sys/auxv.h>
#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace marian {
namespace cpu {
namespace integer {
namespace smmla {

bool available() {
  static const bool ok = [] {
    const char* kill = std::getenv("BERGAMOT_NO_I8MM");
    if (kill && kill[0] == '1') return false;
#if defined(__linux__)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#elif defined(__APPLE__)
    int v = 0;
    size_t sz = sizeof(v);
    return sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &sz, nullptr, 0) == 0 && v == 1;
#else
    return false;
#endif
  }();
  return ok;
}

// Packed operand layout, shared by A and B: pairs of K-strips interleaved in
// 8-byte octets, zero-padded to pair count and to K8 = ceil(K/8) octets:
//   packed[(pair * K8 + kt) * 16 + 0..7]  = strip(2*pair)  bytes [kt*8, kt*8+8)
//   packed[(pair * K8 + kt) * 16 + 8..15] = strip(2*pair+1) bytes [kt*8, kt*8+8)
// A strip = one row of A (row-major => K contiguous); B strip = one column of
// B (col-major => K contiguous). Zero pad contributes zero to the dot. Every
// byte of the output is written, so callers need not clear it.
void packStrips(const int8_t* base,  // first strip
                int strips,          // M (rows of A) or N (cols of B)
                int K,
                int8_t* out) {       // sized pairs*K8*16
  const int K8 = (K + 7) / 8;
  const int pairs = (strips + 1) / 2;
  const int kTail = K - (K8 - 1) * 8;  // bytes of the last octet that exist
  for (int p = 0; p < pairs; ++p) {
    const int s0 = 2 * p, s1 = 2 * p + 1;
    const int8_t* src0 = base + static_cast<size_t>(s0) * K;
    const int8_t* src1 = s1 < strips ? base + static_cast<size_t>(s1) * K : nullptr;
    int8_t* dst = out + static_cast<size_t>(p) * K8 * 16;
    int kt = 0;
    for (; kt < K8 - 1; ++kt, dst += 16) {
      std::memcpy(dst, src0 + kt * 8, 8);
      if (src1) std::memcpy(dst + 8, src1 + kt * 8, 8);
      else std::memset(dst + 8, 0, 8);
    }
    // last octet, possibly partial
    std::memset(dst, 0, 16);
    std::memcpy(dst, src0 + kt * 8, kTail);
    if (src1) std::memcpy(dst + 8, src1 + kt * 8, kTail);
  }
}

namespace {

// Full 8x8 tile: 16 named accumulators so they live in registers; the
// generic variant below (spilled accumulator array) only runs edge tiles.
inline void tile8x8Full(const int8_t* aPack,
                        const int8_t* bPack,
                        int K8,
                        int32_t* C,
                        int ldc) {
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
  int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  int32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;
  int32x4_t c30 = c00, c31 = c00, c32 = c00, c33 = c00;
  const size_t stride = static_cast<size_t>(K8) * 16;
  const int8_t* a0 = aPack;
  const int8_t* a1 = aPack + stride;
  const int8_t* a2 = aPack + 2 * stride;
  const int8_t* a3 = aPack + 3 * stride;
  const int8_t* b0 = bPack;
  const int8_t* b1 = bPack + stride;
  const int8_t* b2 = bPack + 2 * stride;
  const int8_t* b3 = bPack + 3 * stride;
  for (int kt = 0; kt < K8; ++kt) {
    const int8x16_t va0 = vld1q_s8(a0 + kt * 16);
    const int8x16_t va1 = vld1q_s8(a1 + kt * 16);
    const int8x16_t va2 = vld1q_s8(a2 + kt * 16);
    const int8x16_t va3 = vld1q_s8(a3 + kt * 16);
    const int8x16_t vb0 = vld1q_s8(b0 + kt * 16);
    const int8x16_t vb1 = vld1q_s8(b1 + kt * 16);
    const int8x16_t vb2 = vld1q_s8(b2 + kt * 16);
    const int8x16_t vb3 = vld1q_s8(b3 + kt * 16);
    c00 = vmmlaq_s32(c00, va0, vb0);
    c01 = vmmlaq_s32(c01, va0, vb1);
    c02 = vmmlaq_s32(c02, va0, vb2);
    c03 = vmmlaq_s32(c03, va0, vb3);
    c10 = vmmlaq_s32(c10, va1, vb0);
    c11 = vmmlaq_s32(c11, va1, vb1);
    c12 = vmmlaq_s32(c12, va1, vb2);
    c13 = vmmlaq_s32(c13, va1, vb3);
    c20 = vmmlaq_s32(c20, va2, vb0);
    c21 = vmmlaq_s32(c21, va2, vb1);
    c22 = vmmlaq_s32(c22, va2, vb2);
    c23 = vmmlaq_s32(c23, va2, vb3);
    c30 = vmmlaq_s32(c30, va3, vb0);
    c31 = vmmlaq_s32(c31, va3, vb1);
    c32 = vmmlaq_s32(c32, va3, vb2);
    c33 = vmmlaq_s32(c33, va3, vb3);
  }
  auto store2x2 = [&](int32x4_t v, int r, int c) {
    vst1_s32(C + static_cast<size_t>(r) * ldc + c, vget_low_s32(v));
    vst1_s32(C + static_cast<size_t>(r + 1) * ldc + c, vget_high_s32(v));
  };
  store2x2(c00, 0, 0); store2x2(c01, 0, 2); store2x2(c02, 0, 4); store2x2(c03, 0, 6);
  store2x2(c10, 2, 0); store2x2(c11, 2, 2); store2x2(c12, 2, 4); store2x2(c13, 2, 6);
  store2x2(c20, 4, 0); store2x2(c21, 4, 2); store2x2(c22, 4, 4); store2x2(c23, 4, 6);
  store2x2(c30, 6, 0); store2x2(c31, 6, 2); store2x2(c32, 6, 4); store2x2(c33, 6, 6);
}

// Row-pair tile variants for small M: 1 row-pair (M<=2) and 2 row-pairs
// (M<=4) across 4 col-pairs, so a decode-tail GEMM does not burn 8-row
// padding. Full-width (cpCount==4) only; edges fall through to the generic
// tile below.
inline void tile2x8Full(const int8_t* aPack, const int8_t* bPack, int K8,
                        int32_t* C, int ldc, int mLive) {
  int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;
  const size_t stride = static_cast<size_t>(K8) * 16;
  for (int kt = 0; kt < K8; ++kt) {
    const int8x16_t a = vld1q_s8(aPack + kt * 16);
    c0 = vmmlaq_s32(c0, a, vld1q_s8(bPack + kt * 16));
    c1 = vmmlaq_s32(c1, a, vld1q_s8(bPack + stride + kt * 16));
    c2 = vmmlaq_s32(c2, a, vld1q_s8(bPack + 2 * stride + kt * 16));
    c3 = vmmlaq_s32(c3, a, vld1q_s8(bPack + 3 * stride + kt * 16));
  }
  const int32x4_t acc[4] = {c0, c1, c2, c3};
  for (int j = 0; j < 4; ++j) {
    vst1_s32(C + 2 * j, vget_low_s32(acc[j]));
    if (mLive > 1) vst1_s32(C + ldc + 2 * j, vget_high_s32(acc[j]));
  }
}

inline void tile4x8Full(const int8_t* aPack, const int8_t* bPack, int K8,
                        int32_t* C, int ldc, int mLive) {
  int32x4_t c00 = vdupq_n_s32(0), c01 = c00, c02 = c00, c03 = c00;
  int32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
  const size_t stride = static_cast<size_t>(K8) * 16;
  const int8_t* a1 = aPack + stride;
  for (int kt = 0; kt < K8; ++kt) {
    const int8x16_t va0 = vld1q_s8(aPack + kt * 16);
    const int8x16_t va1 = vld1q_s8(a1 + kt * 16);
    const int8x16_t vb0 = vld1q_s8(bPack + kt * 16);
    const int8x16_t vb1 = vld1q_s8(bPack + stride + kt * 16);
    const int8x16_t vb2 = vld1q_s8(bPack + 2 * stride + kt * 16);
    const int8x16_t vb3 = vld1q_s8(bPack + 3 * stride + kt * 16);
    c00 = vmmlaq_s32(c00, va0, vb0);
    c01 = vmmlaq_s32(c01, va0, vb1);
    c02 = vmmlaq_s32(c02, va0, vb2);
    c03 = vmmlaq_s32(c03, va0, vb3);
    c10 = vmmlaq_s32(c10, va1, vb0);
    c11 = vmmlaq_s32(c11, va1, vb1);
    c12 = vmmlaq_s32(c12, va1, vb2);
    c13 = vmmlaq_s32(c13, va1, vb3);
  }
  const int32x4_t r0[4] = {c00, c01, c02, c03};
  const int32x4_t r1[4] = {c10, c11, c12, c13};
  for (int j = 0; j < 4; ++j) {
    vst1_s32(C + 2 * j, vget_low_s32(r0[j]));
    vst1_s32(C + ldc + 2 * j, vget_high_s32(r0[j]));
  }
  if (mLive > 2) {
    for (int j = 0; j < 4; ++j) {
      vst1_s32(C + 2 * ldc + 2 * j, vget_low_s32(r1[j]));
      if (mLive > 3) vst1_s32(C + 3 * ldc + 2 * j, vget_high_s32(r1[j]));
    }
  }
}

// Edge-tile variant: any rpCount/cpCount in [1,4], stores guarded by
// mLive/nLive.
inline void tile8x8(const int8_t* aPack,  // 4 row-pair strips base (rp stride K8*16)
                    const int8_t* bPack,  // 4 col-pair strips base
                    int rpCount,
                    int cpCount,
                    int K8,
                    int32_t* C,
                    int ldc,   // = N
                    int mLive,  // rows to store (1..8)
                    int nLive)  // cols to store (1..8)
{
  int32x4_t acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = vdupq_n_s32(0);

  const size_t stride = static_cast<size_t>(K8) * 16;
  for (int kt = 0; kt < K8; ++kt) {
    int8x16_t a[4], b[4];
    for (int i = 0; i < rpCount; ++i) a[i] = vld1q_s8(aPack + i * stride + kt * 16);
    for (int j = 0; j < cpCount; ++j) b[j] = vld1q_s8(bPack + j * stride + kt * 16);
    for (int i = 0; i < rpCount; ++i)
      for (int j = 0; j < cpCount; ++j) acc[i][j] = vmmlaq_s32(acc[i][j], a[i], b[j]);
  }

  // acc[i][j] lanes = [C(2i,2j), C(2i,2j+1), C(2i+1,2j), C(2i+1,2j+1)]
  for (int i = 0; i < rpCount; ++i) {
    for (int j = 0; j < cpCount; ++j) {
      const int r0 = 2 * i, c0 = 2 * j;
      const int32x4_t v = acc[i][j];
      if (r0 + 1 < mLive && c0 + 1 < nLive) {
        vst1_s32(C + static_cast<size_t>(r0) * ldc + c0, vget_low_s32(v));
        vst1_s32(C + static_cast<size_t>(r0 + 1) * ldc + c0, vget_high_s32(v));
      } else {
        int32_t lanes[4];
        vst1q_s32(lanes, v);
        if (r0 < mLive && c0 < nLive) C[static_cast<size_t>(r0) * ldc + c0] = lanes[0];
        if (r0 < mLive && c0 + 1 < nLive) C[static_cast<size_t>(r0) * ldc + c0 + 1] = lanes[1];
        if (r0 + 1 < mLive && c0 < nLive) C[static_cast<size_t>(r0 + 1) * ldc + c0] = lanes[2];
        if (r0 + 1 < mLive && c0 + 1 < nLive) C[static_cast<size_t>(r0 + 1) * ldc + c0 + 1] = lanes[3];
      }
    }
  }
}

struct PackedB {
  std::vector<int8_t> data;
  int K = 0;
  int N = 0;
  uint64_t generation = 0;
};

// Per-thread caches, mirroring the ruy prepacked-cache lifetime rules from
// patch 0011: keyed by the weight data pointer, dropped whenever the model
// generation counter moves (model unload can hand a new model the same
// addresses).
thread_local std::unordered_map<const void*, PackedB> bCache;
thread_local std::vector<int8_t> aScratch;
thread_local std::vector<int8_t> bScratch;
thread_local uint64_t seenGeneration = 0;
thread_local testing::TileCounters tileCounts;
// D0: this thread's contribution to the process-wide cache/scratch totals,
// maintained only while lifecycle tracing is on. Declared after the caches
// above so it is destroyed before them at thread exit: the global totals then
// drop when a worker thread goes away.
struct CacheAccount {
  long long packed = 0;
  long long scratch = 0;
  ~CacheAccount() {
    if (packed != 0) lifecycle::smmlaPackedBytes().fetch_sub(packed, std::memory_order_relaxed);
    if (scratch != 0) lifecycle::smmlaScratchBytes().fetch_sub(scratch, std::memory_order_relaxed);
  }
};
thread_local CacheAccount cacheAccount;

// D0: publish this thread's current bCache / scratch footprint. Called only
// where a size can actually have changed.
void publishCacheBytes() {
  if (!lifecycle::enabled()) return;
  long long packed = 0;
  for (const auto& entry : bCache) packed += static_cast<long long>(entry.second.data.capacity());
  const long long scratch =
      static_cast<long long>(aScratch.capacity()) + static_cast<long long>(bScratch.capacity());
  if (packed != cacheAccount.packed) {
    lifecycle::smmlaPackedBytes().fetch_add(packed - cacheAccount.packed, std::memory_order_relaxed);
    cacheAccount.packed = packed;
  }
  if (scratch != cacheAccount.scratch) {
    lifecycle::smmlaScratchBytes().fetch_add(scratch - cacheAccount.scratch, std::memory_order_relaxed);
    cacheAccount.scratch = scratch;
  }
}

}  // namespace

namespace testing {
size_t packedBytes(int strips, int K) {
  return static_cast<size_t>((strips + 1) / 2) * static_cast<size_t>((K + 7) / 8) * 16;
}
void packStrips(const int8_t* base, int strips, int K, int8_t* out) {
  ::marian::cpu::integer::smmla::packStrips(base, strips, K, out);
}
TileCounters tileCounters() { return tileCounts; }
void resetTileCounters() { tileCounts = TileCounters{}; }
}  // namespace testing

void gemm8(const int8_t* A,
           const int8_t* B,
           int32_t* C,
           int M,
           int K,
           int N,
           bool cacheableB,
           uint64_t generation) {
  const int K8 = (K + 7) / 8;
  const int mPairs = (M + 1) / 2;
  const int nPairs = (N + 1) / 2;
  const size_t stripBytes = static_cast<size_t>(K8) * 16;

  if (generation != seenGeneration) {
    if (lifecycle::enabled()) {
      lifecycle::event("smmla_bcache_cleared entries=%zu bytes=%lld scratch_bytes=%lld generation=%llu->%llu",
                       bCache.size(), cacheAccount.packed,
                       static_cast<long long>(aScratch.capacity()) + static_cast<long long>(bScratch.capacity()),
                       static_cast<unsigned long long>(seenGeneration),
                       static_cast<unsigned long long>(generation));
    }
    bCache.clear();
    seenGeneration = generation;
    publishCacheBytes();
  }

  // packStrips writes every byte of its output (zero pads included), so the
  // scratch buffers only ever grow; no per-call clearing.
  auto ensure = [](std::vector<int8_t>& v, size_t n) { if (v.size() < n) v.resize(n); };

  const int8_t* bPacked;
  if (cacheableB) {
    PackedB& e = bCache[B];
    if (e.K != K || e.N != N || e.generation != generation) {
      e.data.resize(static_cast<size_t>(nPairs) * stripBytes);
      packStrips(B, N, K, e.data.data());
      e.K = K;
      e.N = N;
      e.generation = generation;
      publishCacheBytes();
    }
    bPacked = e.data.data();
  } else {
    const size_t before = bScratch.capacity();
    ensure(bScratch, static_cast<size_t>(nPairs) * stripBytes);
    if (bScratch.capacity() != before) publishCacheBytes();
    packStrips(B, N, K, bScratch.data());
    bPacked = bScratch.data();
  }

  const size_t aBefore = aScratch.capacity();
  ensure(aScratch, static_cast<size_t>(mPairs) * stripBytes);
  if (aScratch.capacity() != aBefore) publishCacheBytes();
  packStrips(A, M, K, aScratch.data());

  for (int rp = 0; rp < mPairs; rp += 4) {
    const int rpCount = mPairs - rp < 4 ? mPairs - rp : 4;
    const int r0 = 2 * rp;
    const int mLive = M - r0 < 8 ? M - r0 : 8;
    for (int cp = 0; cp < nPairs; cp += 4) {
      const int cpCount = nPairs - cp < 4 ? nPairs - cp : 4;
      const int c0 = 2 * cp;
      const int nLive = N - c0 < 8 ? N - c0 : 8;
      if (rpCount == 4 && cpCount == 4 && mLive == 8 && nLive == 8) {
        ++tileCounts.full8x8;
        tile8x8Full(aScratch.data() + static_cast<size_t>(rp) * stripBytes,
                    bPacked + static_cast<size_t>(cp) * stripBytes,
                    K8, C + static_cast<size_t>(r0) * N + c0, N);
      } else if (rpCount == 1 && cpCount == 4 && nLive == 8) {
        ++tileCounts.tile2x8;
        tile2x8Full(aScratch.data() + static_cast<size_t>(rp) * stripBytes,
                    bPacked + static_cast<size_t>(cp) * stripBytes,
                    K8, C + static_cast<size_t>(r0) * N + c0, N, mLive);
      } else if (rpCount == 2 && cpCount == 4 && nLive == 8) {
        ++tileCounts.tile4x8;
        tile4x8Full(aScratch.data() + static_cast<size_t>(rp) * stripBytes,
                    bPacked + static_cast<size_t>(cp) * stripBytes,
                    K8, C + static_cast<size_t>(r0) * N + c0, N, mLive);
      } else {
        ++tileCounts.edge;
        tile8x8(aScratch.data() + static_cast<size_t>(rp) * stripBytes,
                bPacked + static_cast<size_t>(cp) * stripBytes,
                rpCount, cpCount, K8,
                C + static_cast<size_t>(r0) * N + c0, N, mLive, nLive);
      }
    }
  }
}

void releaseThreadCaches() {
  const size_t entries = bCache.size();
  const long long packed = cacheAccount.packed;
  const long long scratch = cacheAccount.scratch;
  // swap-with-empty rather than clear(): clear() keeps the hash buckets and
  // vector capacity, which is exactly the memory we are here to give back.
  std::unordered_map<const void*, PackedB>().swap(bCache);
  std::vector<int8_t>().swap(aScratch);
  std::vector<int8_t>().swap(bScratch);
  publishCacheBytes();
  if (lifecycle::enabled() && (entries != 0 || scratch != 0)) {
    lifecycle::event("smmla_released entries=%zu bytes=%lld scratch_bytes=%lld", entries, packed, scratch);
  }
}

}  // namespace smmla
}  // namespace integer
}  // namespace cpu
}  // namespace marian

#else  // !__aarch64__

namespace marian {
namespace cpu {
namespace integer {
namespace smmla {
bool available() { return false; }
void gemm8(const int8_t*, const int8_t*, int32_t*, int, int, int, bool, uint64_t) {}
void releaseThreadCaches() {}
namespace testing {
size_t packedBytes(int, int) { return 0; }
void packStrips(const int8_t*, int, int, int8_t*) {}
TileCounters tileCounters() { return TileCounters{}; }
void resetTileCounters() {}
}  // namespace testing
}  // namespace smmla
}  // namespace integer
}  // namespace cpu
}  // namespace marian

#endif
