#pragma once
// D0: opt-in lifecycle tracing for model / graph / GEMM-packing-cache teardown.
//
// Everything here is inert unless BERGAMOT_LIFECYCLE=1 is set in the
// environment before the process starts. The cost on a hot path is then
// exactly one relaxed atomic load (`lifecycle::enabled()`), which is why the
// flag is a plain extern atomic read inline here rather than a function-local
// static behind a guard variable.
//
// The live-object counters are the exception: they are maintained
// unconditionally (one relaxed fetch_add per TranslationModel /
// ExpressionGraph construction and destruction, i.e. a handful of times per
// process) so a harness or a test can assert "the destructor actually ran"
// without turning tracing on.
#include <atomic>
#include <cstdint>

namespace marian {
namespace lifecycle {

/// Set once during static initialisation from BERGAMOT_LIFECYCLE. Read it
/// through enabled(); it is public only so that read can stay inline.
extern std::atomic<bool> gEnabled;

/// One relaxed load. False until this translation unit's dynamic initialiser
/// has run, so events raised from other translation units' static
/// initialisers are silently dropped -- there are none today.
inline bool enabled() { return gEnabled.load(std::memory_order_relaxed); }

/// Milliseconds on a steady clock, zeroed on the first call in the process.
double nowMs();

/// printf-style event line, prefixed with "[lifecycle t=<ms>] ", to stderr and
/// (on Android) to logcat. No-op when disabled; callers may still guard with
/// enabled() to skip argument evaluation.
void event(const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

/// Live TranslationModel / ExpressionGraph objects. Always maintained.
std::atomic<long> &liveModels();
std::atomic<long> &liveGraphs();

/// Bytes currently retained by the per-thread GEMM packing caches, summed over
/// every thread that has one. Only maintained while enabled() -- the ruy
/// figure needs a cache-size read per GEMM, which we do not want in a
/// production build. Reading these from another thread is a benign race by
/// construction: each is a single atomic.
std::atomic<long long> &ruyPrepackedBytes();   ///< ruy::Context PrepackedCache
std::atomic<long long> &smmlaPackedBytes();    ///< smmla bCache PackedB payloads
std::atomic<long long> &smmlaScratchBytes();   ///< smmla aScratch + bScratch

}  // namespace lifecycle
}  // namespace marian
