// Shape-level GEMM A/B: ruy (SDOT path) against the SMMLA kernel on the same
// shapes the engine's decode / encode / shortlist calls use. Prints per-shape
// time and speed-up, flags any result that differs from ruy's (must never
// happen: exact integer math) and ends with the geomean. Built as `ab_bench`
// under -DBUILD_SMMLA_TEST=ON for host and Android alike; on a phone pin it
// to one core class (taskset) and read scaling_max_freq around the run.
//   ab_bench [reps-scale]   reps-scale > 1 lengthens every measurement
// Every B buffer stays alive for the whole run: ruy's prepacked cache is
// keyed by pointer + layout, so a freed-and-reused address would serve a
// stale pack (bench artefact, not an engine bug -- the engine only caches
// true constant weights).
#include "tensors/cpu/smmla_gemm.h"

#include "ruy/ruy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace smmla = marian::cpu::integer::smmla;

int main(int argc, char** argv) {
  const double scale = argc > 1 ? std::atof(argv[1]) : 1.0;
  if (!smmla::available()) {
    std::printf("no i8mm on this CPU; nothing to compare\n");
    return 3;
  }
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> d(-128, 127);
  ruy::Context ctx;
  // spin so a cold process gets migrated off the efficiency cores first
  { volatile long x = 0; for (long i = 0; i < 800000000L; i++) x += i; }
  std::vector<std::vector<int8_t>> keepAlive;
  keepAlive.reserve(64);
  struct Case { int M, K, N; bool cacheB; };
  const std::vector<Case> cases = {
    {1, 384, 384, true},  {2, 384, 384, true},  {4, 384, 384, true},  {8, 384, 384, true},
    {16, 384, 384, true}, {48, 384, 384, true},
    {8, 384, 1536, true}, {8, 1536, 384, true}, {48, 384, 1536, true}, {48, 1536, 384, true},
    {1, 384, 5000, false}, {8, 384, 5000, false}, {48, 384, 5000, false},  // shortlist output layer, packs B every call
    {320, 384, 384, true}, {320, 384, 1536, true},                          // encoder
  };
  std::printf("%-30s %10s %10s   speed-up\n", "shape", "ruy", "smmla");
  double geo = 0.0;
  bool allOk = true;
  for (const auto& c : cases) {
    keepAlive.emplace_back(static_cast<size_t>(c.M) * c.K);
    auto& A = keepAlive.back();
    keepAlive.emplace_back(static_cast<size_t>(c.N) * c.K);
    auto& B = keepAlive.back();
    for (auto& x : A) x = static_cast<int8_t>(d(rng));
    for (auto& x : B) x = static_cast<int8_t>(d(rng));
    std::vector<int32_t> C1(static_cast<size_t>(c.M) * c.N), C2(C1.size());
    const long long macs = static_cast<long long>(c.M) * c.K * c.N;
    const int reps = static_cast<int>((600000000 / macs) * scale) + 3;

    ruy::Matrix<std::int8_t> lhs;
    ruy::MakeSimpleLayout(c.M, c.K, ruy::Order::kRowMajor, lhs.mutable_layout());
    lhs.set_data(A.data());
    ruy::Matrix<std::int8_t> rhs;
    ruy::MakeSimpleLayout(c.K, c.N, ruy::Order::kColMajor, rhs.mutable_layout());
    rhs.set_data(B.data());
    if (c.cacheB) rhs.set_cache_policy(ruy::CachePolicy::kAlwaysCache);
    ruy::Matrix<std::int32_t> dst;
    ruy::MakeSimpleLayout(c.M, c.N, ruy::Order::kRowMajor, dst.mutable_layout());
    dst.set_data(C1.data());
    ruy::MulParams<std::int32_t, std::int32_t> mp;
    ruy::Mul(lhs, rhs, mp, &ctx, &dst);  // first call packs B (cacheable) outside the timed region
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) ruy::Mul(lhs, rhs, mp, &ctx, &dst);
    const double rt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;

    smmla::gemm8(A.data(), B.data(), C2.data(), c.M, c.K, c.N, c.cacheB, 1);  // same: pack once, then time
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) smmla::gemm8(A.data(), B.data(), C2.data(), c.M, c.K, c.N, c.cacheB, 1);
    const double st = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    const bool ok = (C1 == C2);
    allOk = allOk && ok;
    std::printf("M=%3d K=%4d N=%4d cacheB=%d %8.1fus %8.1fus   %.2f%s\n", c.M, c.K, c.N,
                static_cast<int>(c.cacheB), rt * 1e6, st * 1e6, rt / st, ok ? "" : "  !MISMATCH");
    geo += std::log(rt / st);
  }
  std::printf("%-30s %10s %10s   %.2f%s\n", "geomean over shapes", "", "",
              std::exp(geo / cases.size()), allOk ? "" : "  (MISMATCHES ABOVE)");
  return allOk ? 0 : 1;
}
