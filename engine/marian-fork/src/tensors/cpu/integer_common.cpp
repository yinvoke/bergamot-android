#include "integer_common.h"

// D0: releaseThreadPackingCaches() needs the per-thread cache owners. Both are
// architecture-gated exactly as their GEMM paths are.
#if defined(ARM)
#include "ruy_interface.h"
#endif
#if defined(__aarch64__)
#include "smmla_gemm.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

#ifdef __SSE__
#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace marian {
namespace cpu {
namespace integer {

// PATCH B: see integer_common.h. Function-local static so the counter is
// initialised on first use regardless of translation-unit ordering.
std::atomic<uint64_t> &prepackGeneration() {
  static std::atomic<uint64_t> generation{0};
  return generation;
}

void bumpPrepackGeneration() {
  prepackGeneration().fetch_add(1, std::memory_order_relaxed);
}

// D0: see integer_common.h. Both caches are per-thread, so this only ever
// touches the calling thread's state -- no locking needed, and no ordering
// against other threads' GEMMs.
void releaseThreadPackingCaches() {
#if defined(ARM)
  releaseThreadRuyCache();
#endif
#if defined(__aarch64__)
  smmla::releaseThreadCaches();
#endif
}

// PATCH D1/D2: see integer_common.h.
namespace {
bool envFlag(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::mutex &wembShadowMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, std::vector<float>> &wembShadows() {
  static std::map<std::string, std::vector<float>> shadows;
  return shadows;
}
}  // namespace

bool wembKeepFp32() {
  static const bool keep = envFlag("BERGAMOT_FP32_WEMB");
  return keep;
}

bool wembCheckEnabled() {
  static const bool check = envFlag("BERGAMOT_WEMB_CHECK");
  return check;
}

bool isWembTableName(const std::string &name, size_t elements) {
  if(name.find("Wemb") == std::string::npos)
    return false;
  // "<name>_QuantMultA" is a 1x1 intgemm8 item that only *looks* like a Wemb.
  if(elements <= 1)
    return false;
  static const std::string suffix = "_QuantMultA";
  if(name.size() >= suffix.size()
     && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
    return false;
  return true;
}

void registerWembShadow(const std::string &name, std::vector<float> &&table) {
  std::lock_guard<std::mutex> lock(wembShadowMutex());
  wembShadows()[name] = std::move(table);
}

const std::vector<float> *findWembShadow(const std::string &name) {
  std::lock_guard<std::mutex> lock(wembShadowMutex());
  auto it = wembShadows().find(name);
  return it == wembShadows().end() ? nullptr : &it->second;
}

std::string stripParamNamespace(const std::string &name) {
  auto pos = name.rfind("::");
  return pos == std::string::npos ? name : name.substr(pos + 2);
}

// Aggregated per table so a 150-sentence run does not emit one line per batch.
// Printed straight to stderr: the bergamot configs run with quiet: true, which
// silences the marian logger.
namespace {
struct RowCheckTally {
  size_t rows = 0;
  size_t elements = 0;
  size_t mismatches = 0;
};

std::map<std::string, RowCheckTally> &rowCheckTallies() {
  static std::map<std::string, RowCheckTally> tallies;
  return tallies;
}

struct OpCounter {
  size_t calls = 0;
  size_t elements = 0;
};

std::map<std::string, OpCounter> &opCounters() {
  static std::map<std::string, OpCounter> counters;
  return counters;
}

struct RowCheckReporter {
  ~RowCheckReporter() {
    // No lock: this runs during static destruction, after translation stopped.
    for(const auto &entry : rowCheckTallies())
      std::fprintf(stderr,
                   "[wemb-check] %s rows-dequant total: %zu rows / %zu elements, %zu mismatches\n",
                   entry.first.c_str(), entry.second.rows, entry.second.elements,
                   entry.second.mismatches);
    for(const auto &entry : opCounters())
      std::fprintf(stderr, "[wemb-count] %s calls=%zu elements=%zu\n", entry.first.c_str(),
                   entry.second.calls, entry.second.elements);
  }
};

// Constructed on first use by either reporter path; both make sure the mutex
// and the maps above already exist, so this is destroyed before them.
RowCheckReporter &reporter() {
  static RowCheckReporter instance;
  return instance;
}
}  // namespace

void countWembOp(const char *what, size_t elements) {
  if(!wembCheckEnabled())
    return;
  std::lock_guard<std::mutex> lock(wembShadowMutex());
  auto &counters = opCounters();
  rowCheckTallies();
  reporter();
  auto &counter = counters[what];
  ++counter.calls;
  counter.elements += elements;
}

void reportWembRowCheck(const std::string &name, size_t rows, size_t elements, size_t mismatches) {
  std::lock_guard<std::mutex> lock(wembShadowMutex());
  auto &tallies = rowCheckTallies();
  opCounters();
  reporter();  // flushes the totals at exit
  auto &tally = tallies[name];
  tally.rows += rows;
  tally.elements += elements;
  tally.mismatches += mismatches;
  if(mismatches != 0)
    std::fprintf(stderr, "[wemb-check] %s rows-dequant: %zu rows / %zu elements, %zu MISMATCHES\n",
                 name.c_str(), rows, elements, mismatches);
}

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;

#ifdef __AVX512F__
    // Multiples of 16 add together.
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#elif __SSE__
    // Multiples of 4 add together.
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    int n4 = (n / 4) * 4;
    using __m128 = float32x4_t;
    for(; i < n4; i += 4) {
      __m128 ai = vld1q_f32(x + j * n + i);
      __m128 bi = vld1q_f32(bias + i);
      __m128 yi = vaddq_f32(ai, bi);
      vst1q_f32(y + j * n + i, yi);
    }

#else
    // StandardCPP No SIMD case.
    for(i = 0;  i < n; i++) {
        y[j * n + i] = x[j * n + i] + bias[i];
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }
}

} //integer
} //cpu
} //marian
