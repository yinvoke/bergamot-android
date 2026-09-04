// Correctness test set for the SMMLA int8 GEMM (smmla_gemm.cpp), modelled on
// ruy/test.h: exact-match against a naive reference over shape families,
// guard-page buffers that turn any pack/store overrun into a SIGSEGV, value
// ranges including the int8 extremes, cache/generation semantics, thread
// isolation, and an explicit ISA-coverage assertion.
//
// Exit 0 = all sections passed (or kernel sections SKIPPED on a CPU without
// i8mm); nonzero = failure. Modes:
//   (none)              run everything
//   --killswitch-probe  print available() as 0/1 and exit with that value
//                       (run with BERGAMOT_NO_I8MM=1 to test the switch)
//   --force-sigill      call the kernel even when available() is false; on a
//                       CPU without i8mm this MUST die with SIGILL (exit 132)
//                       -- the negative test that the runtime gate matters
// Env: SMMLA_TEST_REQUIRE_I8MM=1 fails instead of skipping when the CPU has
// no i8mm (CI on Apple Silicon / 8 Gen 3 uses this so coverage can't silently
// regress, same idea as ruy's "we haven't tested the kNeonDotprod path").
#include "tensors/cpu/smmla_gemm.h"

#include "ruy/ruy.h"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using marian::cpu::integer::smmla::available;
using marian::cpu::integer::smmla::gemm8;

namespace {

int failures = 0;
int checks = 0;

void fail(const std::string& what) {
  ++failures;
  if (failures <= 20) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

// Buffer whose last byte is the last mapped byte: one page past the end is
// unmapped, so a read or write overrun faults instead of silently passing.
// (ruy's SeparateMappingAllocator.)
template <typename T>
class GuardBuf {
 public:
  explicit GuardBuf(size_t n) : n_(n) {
    const size_t page = static_cast<size_t>(getpagesize());
    const size_t bytes = n * sizeof(T);
    rounded_ = (bytes + page - 1) / page * page;
    if (rounded_ == 0) rounded_ = page;
    void* m = mmap(nullptr, rounded_ + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { std::perror("mmap"); std::abort(); }
    if (munmap(static_cast<char*>(m) + rounded_, page) != 0) { std::perror("munmap"); std::abort(); }
    base_ = m;
    ptr_ = reinterpret_cast<T*>(static_cast<char*>(m) + (rounded_ - bytes));
  }
  ~GuardBuf() { munmap(base_, rounded_); }
  GuardBuf(const GuardBuf&) = delete;
  GuardBuf& operator=(const GuardBuf&) = delete;
  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return n_; }
  T& operator[](size_t i) { return ptr_[i]; }

 private:
  void* base_ = nullptr;
  T* ptr_ = nullptr;
  size_t n_ = 0;
  size_t rounded_ = 0;
};

enum class Fill { kGeneral, kAvoidMin, kAllMin, kAllMax, kAlternate, kZeros, kOneHot };
const char* fillName(Fill f) {
  switch (f) {
    case Fill::kGeneral: return "general";
    case Fill::kAvoidMin: return "avoid-min";
    case Fill::kAllMin: return "all-min";
    case Fill::kAllMax: return "all-max";
    case Fill::kAlternate: return "alternate";
    case Fill::kZeros: return "zeros";
    case Fill::kOneHot: return "one-hot";
  }
  return "?";
}

void fillValues(int8_t* p, size_t n, Fill f, std::mt19937& rng) {
  std::uniform_int_distribution<int> general(-128, 127), avoidMin(-127, 127);
  for (size_t i = 0; i < n; ++i) {
    switch (f) {
      case Fill::kGeneral: p[i] = static_cast<int8_t>(general(rng)); break;
      case Fill::kAvoidMin: p[i] = static_cast<int8_t>(avoidMin(rng)); break;
      case Fill::kAllMin: p[i] = -128; break;
      case Fill::kAllMax: p[i] = 127; break;
      case Fill::kAlternate: p[i] = (i & 1) ? 127 : -128; break;
      case Fill::kZeros: p[i] = 0; break;
      case Fill::kOneHot: p[i] = (i % 97 == 0) ? static_cast<int8_t>(general(rng)) : 0; break;
    }
  }
}

void reference(const int8_t* A, const int8_t* B, int32_t* C, int M, int K, int N) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int64_t s = 0;
      for (int k = 0; k < K; ++k)
        s += static_cast<int64_t>(A[static_cast<size_t>(m) * K + k]) *
             static_cast<int64_t>(B[static_cast<size_t>(n) * K + k]);
      if (s != static_cast<int32_t>(s)) { fail("reference int32 overflow"); }
      C[static_cast<size_t>(m) * N + n] = static_cast<int32_t>(s);
    }
}

// One shape, one fill: A/B/C all guard-paged; C pre-poisoned so an unwritten
// element is caught too.
void checkShape(int M, int K, int N, Fill fill, std::mt19937& rng, bool cacheable, uint64_t generation) {
  GuardBuf<int8_t> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(N) * K);
  GuardBuf<int32_t> C(static_cast<size_t>(M) * N), R(static_cast<size_t>(M) * N);
  fillValues(A.data(), A.size(), fill, rng);
  fillValues(B.data(), B.size(), fill == Fill::kOneHot ? Fill::kGeneral : fill, rng);
  for (size_t i = 0; i < C.size(); ++i) C[i] = 0x7EADBEEF;
  reference(A.data(), B.data(), R.data(), M, K, N);
  gemm8(A.data(), B.data(), C.data(), M, K, N, cacheable, generation);
  ++checks;
  for (size_t i = 0; i < C.size(); ++i) {
    if (C[i] != R[i]) {
      fail("M=" + std::to_string(M) + " K=" + std::to_string(K) + " N=" + std::to_string(N) +
           " fill=" + fillName(fill) + " idx=" + std::to_string(i) + " got " + std::to_string(C[i]) +
           " want " + std::to_string(R[i]));
      return;
    }
  }
}

void runShapes(const std::vector<std::tuple<int, int, int>>& shapes, const char* family, int seeds) {
  std::printf("[%s] %zu shapes x %d seeds\n", family, shapes.size(), seeds);
  uint64_t gen = 100;
  for (auto [M, K, N] : shapes) {
    for (int s = 0; s < seeds; ++s) {
      std::mt19937 rng(1234u + static_cast<unsigned>(s) * 7919u + static_cast<unsigned>(M * 31 + K * 7 + N));
      checkShape(M, K, N, Fill::kGeneral, rng, /*cacheable=*/false, gen);
      checkShape(M, K, N, Fill::kAvoidMin, rng, /*cacheable=*/true, ++gen);
    }
  }
}

void runExtremes() {
  std::printf("[extremes] int8 boundary fills on engine-like shapes\n");
  const std::vector<std::tuple<int, int, int>> shapes = {
      {1, 384, 384}, {2, 384, 8}, {7, 1536, 384}, {48, 384, 1536}, {9, 392, 17}, {16, 8, 16}, {1, 32767, 1}};
  uint64_t gen = 500;
  for (Fill f : {Fill::kAllMin, Fill::kAllMax, Fill::kAlternate, Fill::kZeros, Fill::kOneHot}) {
    for (auto [M, K, N] : shapes) {
      std::mt19937 rng(99u);
      checkShape(M, K, N, f, rng, /*cacheable=*/false, gen);
      checkShape(M, K, N, f, rng, /*cacheable=*/true, ++gen);
    }
  }
}

void runCacheSemantics() {
  std::printf("[cache] generation / same-pointer / interleaved / non-cacheable-mutation\n");
  const int M = 16, K = 384, N = 64;
  std::mt19937 rng(7u);
  std::vector<int8_t> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(N) * K), B2(static_cast<size_t>(N) * K);
  std::vector<int32_t> C(static_cast<size_t>(M) * N), R(static_cast<size_t>(M) * N), R2(static_cast<size_t>(M) * N);
  fillValues(A.data(), A.size(), Fill::kGeneral, rng);
  fillValues(B.data(), B.size(), Fill::kGeneral, rng);
  fillValues(B2.data(), B2.size(), Fill::kGeneral, rng);
  reference(A.data(), B.data(), R.data(), M, K, N);
  reference(A.data(), B2.data(), R2.data(), M, K, N);

  gemm8(A.data(), B.data(), C.data(), M, K, N, true, 1); ++checks;
  if (C != R) fail("cache: first cacheable call");
  gemm8(A.data(), B.data(), C.data(), M, K, N, true, 1); ++checks;
  if (C != R) fail("cache: cached hit");
  // Two different cacheable weights alive at once (encoder + decoder layers).
  gemm8(A.data(), B2.data(), C.data(), M, K, N, true, 1); ++checks;
  if (C != R2) fail("cache: second weight interleaved");
  gemm8(A.data(), B.data(), C.data(), M, K, N, true, 1); ++checks;
  if (C != R) fail("cache: first weight still correct after interleave");
  // Same address, new contents, new generation: must repack (model reload).
  std::vector<int8_t> saved = B;
  B = B2;
  gemm8(A.data(), B.data(), C.data(), M, K, N, true, 2); ++checks;
  if (C != R2) fail("cache: generation bump must invalidate");
  B = saved;
  // Same address and generation but a different N: key must include shape.
  gemm8(A.data(), B.data(), C.data(), M, K, N / 2, true, 2); ++checks;
  std::vector<int32_t> Rhalf(static_cast<size_t>(M) * (N / 2));
  reference(A.data(), B.data(), Rhalf.data(), M, K, N / 2);
  if (!std::equal(Rhalf.begin(), Rhalf.end(), C.begin())) fail("cache: shape change at same pointer");
  // Non-cacheable B (shortlist pattern): buffer reused with new contents
  // every call and must always reflect the current contents.
  gemm8(A.data(), B.data(), C.data(), M, K, N, false, 2); ++checks;
  if (C != R) fail("non-cacheable: contents 1");
  B = B2;
  gemm8(A.data(), B.data(), C.data(), M, K, N, false, 2); ++checks;
  if (C != R2) fail("non-cacheable: contents 2 at same address");
}

void runThreads() {
  std::printf("[threads] 4 workers, shared cacheable B, private A, 40 iterations each\n");
  const int M = 24, K = 384, N = 96;
  std::mt19937 rng(3u);
  std::vector<int8_t> B(static_cast<size_t>(N) * K);
  fillValues(B.data(), B.size(), Fill::kGeneral, rng);
  std::atomic<int> bad{0};
  std::vector<std::thread> workers;
  for (int t = 0; t < 4; ++t) {
    workers.emplace_back([&, t] {
      std::mt19937 r(100u + static_cast<unsigned>(t));
      std::vector<int8_t> A(static_cast<size_t>(M) * K);
      std::vector<int32_t> C(static_cast<size_t>(M) * N), R(static_cast<size_t>(M) * N);
      for (int it = 0; it < 40; ++it) {
        fillValues(A.data(), A.size(), Fill::kGeneral, r);
        reference(A.data(), B.data(), R.data(), M, K, N);
        gemm8(A.data(), B.data(), C.data(), M, K, N, /*cacheable=*/true, 7);
        if (C != R) { ++bad; return; }
        // and the per-call packed path in the same thread
        gemm8(A.data(), B.data(), C.data(), M, K, N, /*cacheable=*/false, 7);
        if (C != R) { ++bad; return; }
      }
    });
  }
  for (auto& w : workers) w.join();
  checks += 4;
  if (bad) fail("threads: " + std::to_string(bad.load()) + " worker(s) produced wrong results");
}

// Packer on its own (KleidiAI pattern): guard-paged output sized exactly to
// packedBytes(), byte-compared against an independent reference packer, so a
// pack overrun faults and a wrong interleave is caught even when the GEMM
// result would happen to be right.
void referencePack(const int8_t* base, int strips, int K, std::vector<int8_t>& out) {
  const int K8 = (K + 7) / 8, pairs = (strips + 1) / 2;
  out.assign(static_cast<size_t>(pairs) * K8 * 16, 0);
  for (int p = 0; p < pairs; ++p)
    for (int kt = 0; kt < K8; ++kt)
      for (int half = 0; half < 2; ++half) {
        const int s = 2 * p + half;
        if (s >= strips) continue;
        for (int j = 0; j < 8; ++j) {
          const int k = kt * 8 + j;
          if (k < K) out[(static_cast<size_t>(p) * K8 + kt) * 16 + half * 8 + j] = base[static_cast<size_t>(s) * K + k];
        }
      }
}

void runPacker() {
  using namespace marian::cpu::integer::smmla::testing;
  std::printf("[pack] guard-paged packer vs reference, strips x K grid\n");
  std::mt19937 rng(5u);
  for (int strips : {1, 2, 3, 7, 8, 9, 16, 17, 33, 64, 127})
    for (int K : {1, 7, 8, 9, 15, 16, 17, 24, 383, 384, 385, 392, 1536}) {
      GuardBuf<int8_t> src(static_cast<size_t>(strips) * K);
      fillValues(src.data(), src.size(), Fill::kGeneral, rng);
      const size_t bytes = packedBytes(strips, K);
      GuardBuf<int8_t> packed(bytes);
      for (size_t i = 0; i < bytes; ++i) packed[i] = 0x5A;  // poison: zero pad must be written
      packStrips(src.data(), strips, K, packed.data());
      std::vector<int8_t> ref;
      referencePack(src.data(), strips, K, ref);
      ++checks;
      if (ref.size() != bytes) { fail("pack: size mismatch"); continue; }
      for (size_t i = 0; i < bytes; ++i)
        if (packed[i] != ref[i]) {
          fail("pack: strips=" + std::to_string(strips) + " K=" + std::to_string(K) + " byte " +
               std::to_string(i) + " got " + std::to_string(packed[i]) + " want " + std::to_string(ref[i]));
          break;
        }
    }
}

// Structured one-hot (research recommendation): A = e(i,k) makes C row i equal
// B column k and every other row zero -- the fastest way to localise a pack
// index bug, swept across tile and octet boundaries.
void runOneHot() {
  std::printf("[one-hot] A = e(i,k) => C[i][:] == B[:][k], other rows zero\n");
  std::mt19937 rng(8u);
  for (auto [M, K, N] : std::vector<std::tuple<int, int, int>>{{9, 384, 17}, {8, 16, 8}, {17, 24, 33}, {3, 9, 5}, {16, 1536, 384}}) {
    std::vector<int8_t> B(static_cast<size_t>(N) * K);
    fillValues(B.data(), B.size(), Fill::kGeneral, rng);
    std::vector<int> is = {0, 1, 7, M - 1};
    if (M > 8) is.push_back(8);
    std::vector<int> ks = {0, 7, K - 1};
    if (K > 8) ks.push_back(8);
    for (int i : is)
      for (int k : ks) {
        if (i >= M || k >= K) continue;
        std::vector<int8_t> A(static_cast<size_t>(M) * K, 0);
        A[static_cast<size_t>(i) * K + k] = 1;
        std::vector<int32_t> C(static_cast<size_t>(M) * N, -1);
        gemm8(A.data(), B.data(), C.data(), M, K, N, false, 3);
        ++checks;
        bool ok = true;
        for (int m = 0; m < M && ok; ++m)
          for (int n = 0; n < N; ++n) {
            const int32_t want = (m == i) ? B[static_cast<size_t>(n) * K + k] : 0;
            if (C[static_cast<size_t>(m) * N + n] != want) { ok = false; break; }
          }
        if (!ok) fail("one-hot M=" + std::to_string(M) + " K=" + std::to_string(K) + " N=" + std::to_string(N) +
                      " i=" + std::to_string(i) + " k=" + std::to_string(k));
      }
  }
}

// Every tile variant must have run at least once on an i8mm CPU (ruy's
// "we haven't tested the kNeonDotprod path" check): a gate written the wrong
// way round, or a variant that is never dispatched, fails loudly here.
void assertTileCoverage() {
  const auto c = marian::cpu::integer::smmla::testing::tileCounters();
  std::printf("[tiles] full8x8=%llu tile4x8=%llu tile2x8=%llu edge=%llu\n",
              static_cast<unsigned long long>(c.full8x8), static_cast<unsigned long long>(c.tile4x8),
              static_cast<unsigned long long>(c.tile2x8), static_cast<unsigned long long>(c.edge));
  ++checks;
  if (!c.full8x8 || !c.tile4x8 || !c.tile2x8 || !c.edge)
    fail("tile coverage: a tile variant never ran -- dispatch or gate is wrong");
}

// The engine's own contract: same bytes as ruy on realistic shapes. B buffers
// are kept alive for the whole section because ruy's prepacked cache keys on
// the data pointer -- a freed-and-reused address would hand ruy a stale pack
// and fake a mismatch (a test artifact, not an engine bug: the engine only
// caches true constants).
void runRuyEquality() {
  std::printf("[ruy] bytewise equality on decode/encoder/shortlist shapes\n");
  ruy::Context ctx;
  std::mt19937 rng(11u);
  const std::vector<std::tuple<int, int, int, bool>> cases = {
      {1, 384, 384, true},  {2, 384, 384, true},   {3, 384, 1536, true},  {8, 1536, 384, true},
      {48, 384, 384, true}, {48, 384, 1536, true}, {320, 384, 384, true}, {1, 384, 5000, false},
      {8, 384, 5000, false}, {48, 384, 8000, false}, {17, 1536, 1536, true}, {512, 384, 384, true}};
  std::vector<std::vector<int8_t>> keep;
  keep.reserve(cases.size() * 2);
  uint64_t gen = 900;
  for (auto [M, K, N, cacheable] : cases) {
    keep.emplace_back(static_cast<size_t>(M) * K);
    auto& A = keep.back();
    keep.emplace_back(static_cast<size_t>(N) * K);
    auto& B = keep.back();
    fillValues(A.data(), A.size(), Fill::kGeneral, rng);
    fillValues(B.data(), B.size(), Fill::kGeneral, rng);
    std::vector<int32_t> C1(static_cast<size_t>(M) * N), C2(static_cast<size_t>(M) * N);
    ruy::Matrix<std::int8_t> lhs;
    ruy::MakeSimpleLayout(M, K, ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(A.data());
    ruy::Matrix<std::int8_t> rhs;
    ruy::MakeSimpleLayout(K, N, ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(B.data());
    if (cacheable) rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    ruy::Matrix<std::int32_t> dst;
    ruy::MakeSimpleLayout(M, N, ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(C1.data());
    ruy::MulParams<std::int32_t, std::int32_t> mp;
    ruy::Mul(lhs, rhs, mp, &ctx, &dst);
    ruy::Mul(lhs, rhs, mp, &ctx, &dst);  // second call exercises ruy's cached path
    gemm8(A.data(), B.data(), C2.data(), M, K, N, cacheable, ++gen);
    gemm8(A.data(), B.data(), C2.data(), M, K, N, cacheable, gen);
    ++checks;
    if (C1 != C2) fail("ruy mismatch M=" + std::to_string(M) + " K=" + std::to_string(K) + " N=" + std::to_string(N));
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  const bool have = available();
  if (mode == "--killswitch-probe") {
    std::printf("%d\n", have ? 1 : 0);
    return have ? 1 : 0;
  }
  std::printf("smmla available=%d\n", have ? 1 : 0);
  if (!have && mode != "--force-sigill") {
    if (const char* req = std::getenv("SMMLA_TEST_REQUIRE_I8MM"); req && req[0] == '1') {
      std::fprintf(stderr, "FAIL: i8mm required by SMMLA_TEST_REQUIRE_I8MM but not available on this CPU\n");
      return 3;
    }
    std::printf("SKIPPED: no i8mm on this CPU (fallback path is covered by the engine hash regression)\n");
    return 0;
  }
  if (mode == "--force-sigill") {
    std::printf("forcing the kernel without the gate; on a CPU without i8mm this must SIGILL now\n");
    std::fflush(stdout);
    std::vector<int8_t> A(384), B(384 * 8);
    std::vector<int32_t> C(8);
    gemm8(A.data(), B.data(), C.data(), 1, 384, 8, false, 1);
    std::printf("kernel ran: this CPU has i8mm (or the gate is not the only guard)\n");
    return 0;
  }

  // ruy-style shape families (ruy/test_fast.cc) plus the engine's own shapes.
  std::vector<std::tuple<int, int, int>> square;
  for (int s : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 48, 64, 15, 31, 63, 17, 33, 65}) square.emplace_back(s, s, s);
  runShapes(square, "square", 2);
  runShapes({{2, 3, 4}, {7, 6, 5}, {12, 23, 6}, {19, 3, 11}, {3, 10, 17}, {30, 21, 43}, {7, 57, 9},
             {49, 69, 71}, {38, 111, 29}, {87, 98, 76}, {16, 96, 16}, {16, 88, 16}, {16, 84, 16},
             {16, 92, 16}, {16, 82, 16}, {16, 81, 16}, {16, 95, 16}, {3, 128, 5}},
            "misc", 2);
  runShapes({{1, 32767, 1}, {5, 5001, 4}, {9, 1025, 10}}, "deep", 1);
  runShapes({{101, 1, 103}, {71, 2, 53}, {51, 3, 73}, {51, 4, 43}, {50, 5, 40}, {40, 6, 50}, {33, 7, 31}}, "shallow", 2);
  {
    std::vector<std::tuple<int, int, int>> narrow;
    for (int w : {1, 2, 3, 4, 5, 8}) {
      narrow.emplace_back(w, 12, 13); narrow.emplace_back(15, 19, w);
      narrow.emplace_back(w, 123, 137); narrow.emplace_back(158, 119, w);
    }
    runShapes(narrow, "narrow", 2);
  }
  {
    std::vector<std::tuple<int, int, int>> gemv;
    for (int size = 1; size < 1024; size *= 2)
      for (int depth = 1; depth < 500; depth += 47) { gemv.emplace_back(size, depth, 1); gemv.emplace_back(1, depth, size); }
    gemv.emplace_back(5, 5001, 1); gemv.emplace_back(1, 17, 8193);
    runShapes(gemv, "gemv", 1);
  }
  {
    std::vector<std::tuple<int, int, int>> tails;
    for (int K : {8, 9, 15, 16, 17, 23, 24, 25, 383, 384, 385, 391, 392, 393})
      for (int M : {1, 2, 3, 7, 8, 9}) for (int N : {1, 2, 7, 8, 9, 15, 16, 17}) tails.emplace_back(M, K, N);
    runShapes(tails, "k-tails", 1);
  }
  {
    std::vector<std::tuple<int, int, int>> engine;
    for (int K : {384, 1536})
      for (int M : {1, 2, 3, 4, 7, 8, 9, 16, 48, 320, 512})
        for (int N : {384, 1536, 8000}) engine.emplace_back(M, K, N);
    runShapes(engine, "engine-shapes", 1);
  }
  runExtremes();
  runPacker();
  runOneHot();
  runCacheSemantics();
  runThreads();
  runRuyEquality();
  assertTileCoverage();

  std::printf("checks=%d failures=%d\n", checks, failures);
  return failures ? 1 : 0;
}
