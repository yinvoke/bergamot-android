#pragma once
// SMMLA (Armv8.6 i8mm) int8 GEMM: the int8 matmul path Multiply8Rui takes
// on CPUs that have i8mm (Armv9 cores such as Cortex-X2/X4, A710/A720);
// everything else keeps the ruy SDOT path. Layout contract mirrors that call
// site exactly:
//   A: row-major M x K int8 (quantized activations)
//   B: col-major K x N int8 (N strips of K contiguous bytes; model weights or
//      shortlist-selected columns)
//   C: row-major M x N int32 (raw accumulators; caller unquantizes in place)
// Same integer math as ruy => results are bit-identical (verified on 8 Gen 3,
// 8 Gen 1 and 865 against the same canonical corpus hashes); the engine hash
// regression (tools/regress-hash.sh) and tools/smmla-test are the gates.
// Runtime-gated on HWCAP2_I8MM (macOS: the FEAT_I8MM sysctl);
// BERGAMOT_NO_I8MM=1 forces the ruy path (same-binary A/B, diagnostics).
// Tested SoCs/devices: docs/smmla-compatibility.md.
#include <cstddef>
#include <cstdint>

namespace marian {
namespace cpu {
namespace integer {
namespace smmla {

// True when the CPU has i8mm and the kill switch is not set. Cheap after the
// first call.
bool available();

// C = A * B. cacheableB marks constant model weights: their SMMLA-layout pack
// is cached per thread and invalidated when `generation` changes (same
// counter the ruy prepacked cache uses). Non-cacheable B is packed per call.
void gemm8(const int8_t* A,
           const int8_t* B,
           int32_t* C,
           int M,
           int K,
           int N,
           bool cacheableB,
           uint64_t generation);

// D0: drop THIS thread's packed-B cache and release its scratch buffers.
// Affects only the calling thread; safe on a thread that never ran a GEMM.
// The next gemm8 on this thread simply re-packs, so this is a memory/latency
// trade, never a correctness one.
void releaseThreadCaches();

// Test hooks -- not part of the engine contract. They let the test set check
// the packer on its own (guard-paged output, byte compare against an
// independent reference) and assert that every tile variant actually ran.
namespace testing {
// Bytes packStrips() writes for `strips` K-strips: pairs * ceil(K/8) * 16.
size_t packedBytes(int strips, int K);
// Pair-interleaved packing of A rows or B columns (both are K-contiguous).
void packStrips(const int8_t* base, int strips, int K, int8_t* out);
struct TileCounters {
  uint64_t full8x8 = 0, tile4x8 = 0, tile2x8 = 0, edge = 0;
};
TileCounters tileCounters();  // this thread's counts since resetTileCounters()
void resetTileCounters();
}  // namespace testing

}  // namespace smmla
}  // namespace integer
}  // namespace cpu
}  // namespace marian
