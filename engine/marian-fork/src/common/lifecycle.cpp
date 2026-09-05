#include "common/lifecycle.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace marian {
namespace lifecycle {

namespace {

bool readEnv() {
  const char *value = std::getenv("BERGAMOT_LIFECYCLE");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

std::chrono::steady_clock::time_point &epoch() {
  static std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  return start;
}

}  // namespace

std::atomic<bool> gEnabled{readEnv()};

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - epoch()).count();
}

void event(const char *fmt, ...) {
  if (!enabled()) return;
  char body[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(body, sizeof(body), fmt, args);
  va_end(args);
  // One fprintf so concurrent workers cannot interleave inside a line.
  std::fprintf(stderr, "[lifecycle t=%.3f] %s\n", nowMs(), body);
#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_INFO, "bergamot-lifecycle", "t=%.3f %s", nowMs(), body);
#endif
}

std::atomic<long> &liveModels() {
  static std::atomic<long> counter{0};
  return counter;
}

std::atomic<long> &liveGraphs() {
  static std::atomic<long> counter{0};
  return counter;
}

std::atomic<long long> &ruyPrepackedBytes() {
  static std::atomic<long long> counter{0};
  return counter;
}

std::atomic<long long> &smmlaPackedBytes() {
  static std::atomic<long long> counter{0};
  return counter;
}

std::atomic<long long> &smmlaScratchBytes() {
  static std::atomic<long long> counter{0};
  return counter;
}

}  // namespace lifecycle
}  // namespace marian
