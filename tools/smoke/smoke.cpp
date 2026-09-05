// Host smoke tool. Not a deliverable: exists so CI can catch a broken engine
// patch in seconds, on the same ruy/NEON code path devices use — and so perf
// experiments can run controlled A/B matrices on an ARM host.
//
// Usage: smoke [flags] <model-config.yml> [<second-config.yml>] < text-lines
//   One input line = one source text, translated as a batch (mirrors the JNI
//   surface). With a second config, translates via pivot (first -> second).
//
// Flags (all optional; defaults reproduce the original CI behaviour):
//   --workers N     use AsyncService with N worker threads (default 0 = BlockingService)
//   --cache-size N  translation cache entries, 0 = off (default 0)
//   --repeat R      run the corpus R times (default 1); stdout carries the last pass
//   --bench         print "[bench] key=value" stats to stderr
//   --cache-stats   query hit/miss counters; needs an ENABLE_CACHE_STATS build,
//                   otherwise the engine ABORTs when a cache is enabled
//   --mem           print "[mem] <stage> footprint_mb=... peak_rss_mb=..." for
//                   baseline / after_load / steady / after_unload
//   --dump-prefix P write each pass's output to P.passN.txt (determinism diffs)
//   --lifecycle S   run release-lifecycle scenario S instead of a plain pass;
//                   see kScenarioHelp below. Pair it with BERGAMOT_LIFECYCLE=1
//                   to get the engine's own event trace interleaved.
//   --release-mode M  how the scenarios hand a model back: "release" (default,
//                   the service's release() closure) or "reset" (drop the
//                   shared_ptr only -- the pre-D0 behaviour, for A/B).
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

#include "common/lifecycle.h"
#include "ruy/context.h"
#include "ruy/cpuinfo.h"
#ifdef __ANDROID__
#endif
#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

namespace {

using namespace marian::bergamot;
namespace lifecycle = marian::lifecycle;

const char *kScenarioHelp =
    "  blocking   BlockingService, translate, drop the handle\n"
    "  w1|w2|w4   AsyncService with 1/2/4 workers, translate, drop the handle\n"
    "  w4-1sent   AsyncService with 4 workers translating ONE line (lazy replicas)\n"
    "  pivot      two models, pivot-translate, drop only the first handle\n"
    "  reload     translate, drop, load the same config again, translate again\n"
    "  inflight   drop the handle after submitting but before completion\n"
    "  switch     translate with A, drop A, then load B and translate with it\n";

double peakRssMb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#ifdef __APPLE__
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
#endif
}

/// Live process memory, for the load / steady-state / after-unload ledger.
/// macOS reports phys_footprint (what the kernel bills the process, the number
/// Instruments and jetsam use); Linux reports VmRSS. Both also carry the
/// high-water mark so a load-time spike is visible after the fact.
struct FootprintSample {
  double footprintMb = 0.0;  // phys_footprint (mac) / VmRSS (linux)
  double peakMb = 0.0;       // peak resident, monotone over the process lifetime
};

/// Hand freed-but-still-dirty pages back to the OS before sampling. Without
/// this, "steady" and "after unload" both read as the load-time high-water mark
/// (the allocator keeps the pages) and the ledger says nothing.
void releaseFreedPages() {
#ifdef __APPLE__
  malloc_zone_pressure_relief(nullptr, 0);
#elif defined(__ANDROID__)
  mallopt(M_PURGE, 0);
#elif defined(__GLIBC__)
  malloc_trim(0);
#endif
}

FootprintSample footprintSample() {
  releaseFreedPages();
  FootprintSample sample;
#ifdef __APPLE__
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
      KERN_SUCCESS) {
    sample.footprintMb = static_cast<double>(info.phys_footprint) / 1e6;
  }
#else
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    if (line.rfind("VmRSS:", 0) == 0) {
      sample.footprintMb = std::stod(line.substr(6)) / 1000.0;  // kB -> MB(1e6)
    }
  }
#endif
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
    sample.peakMb = static_cast<double>(usage.ru_maxrss) / 1e6;  // bytes
#else
    sample.peakMb = static_cast<double>(usage.ru_maxrss) / 1000.0;  // kB
#endif
  }
  return sample;
}

double msSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

/// FNV-1a over the concatenated target texts. The determinism probe shared by
/// the plain pass and the lifecycle scenarios, so both compare against the
/// same canonical hashes.
uint64_t responseHash(const std::vector<Response> &responses) {
  uint64_t h = 1469598103934665603ull;
  for (const auto &r : responses) {
    for (unsigned char c : r.target.text) h = (h ^ c) * 1099511628211ull;
    h = (h ^ '\n') * 1099511628211ull;
  }
  return h;
}

std::string hex16(uint64_t value) {
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return buf;
}

/// Process memory. Two implementations on purpose: the same binary is meant to
/// be cross-compiled to Android and produce comparable columns there.
///   Linux/Android: /proc/self/status VmRSS / RssAnon / VmHWM.
///   macOS: TASK_VM_INFO resident_size, and phys_footprint as the "anon"
///          column -- it is the dirty+compressed charge, the closest analogue
///          of RssAnon; peak comes from getrusage since Mach has no VmHWM.
struct MemSample {
  double rssMb = 0.0;
  double anonMb = 0.0;
  double hwmMb = 0.0;
};

MemSample readMemory() {
  MemSample sample;
#if defined(__linux__) || defined(__ANDROID__)
  std::ifstream status("/proc/self/status");
  for (std::string line; std::getline(status, line);) {
    auto valueKb = [&line](const char *key) -> double {
      size_t n = std::strlen(key);
      if (line.compare(0, n, key) != 0) return -1.0;
      return std::strtod(line.c_str() + n, nullptr);
    };
    double v;
    if ((v = valueKb("VmRSS:")) >= 0.0) {
      sample.rssMb = v / 1024.0;
    } else if ((v = valueKb("RssAnon:")) >= 0.0) {
      sample.anonMb = v / 1024.0;
    } else if ((v = valueKb("VmHWM:")) >= 0.0) {
      sample.hwmMb = v / 1024.0;
    }
  }
#elif defined(__APPLE__)
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    sample.rssMb = static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    sample.anonMb = static_cast<double>(info.phys_footprint) / (1024.0 * 1024.0);
  }
  sample.hwmMb = peakRssMb();
#endif
  return sample;
}

/// Prints one checkpoint row and remembers the first checkpoint at which the
/// model count reached zero -- that is the "release request -> destruction
/// actually happened" answer in checkpoint resolution.
struct Checkpointer {
  std::string dtorSeenAt = "never";
  bool sawLiveModel = false;

  void operator()(const char *name) {
    MemSample m = readMemory();
    const long models = lifecycle::liveModels().load(std::memory_order_relaxed);
    const long graphs = lifecycle::liveGraphs().load(std::memory_order_relaxed);
    if (models > 0) sawLiveModel = true;
    if (sawLiveModel && models == 0 && dtorSeenAt == "never") dtorSeenAt = name;
    std::fprintf(stderr,
                 "[lifecycle] %-22s rss_mb=%.1f anon_mb=%.1f hwm_mb=%.1f live_models=%ld live_graphs=%ld "
                 "ruy_prepack_kb=%lld smmla_pack_kb=%lld smmla_scratch_kb=%lld\n",
                 name, m.rssMb, m.anonMb, m.hwmMb, models, graphs,
                 lifecycle::ruyPrepackedBytes().load(std::memory_order_relaxed) / 1024,
                 lifecycle::smmlaPackedBytes().load(std::memory_order_relaxed) / 1024,
                 lifecycle::smmlaScratchBytes().load(std::memory_order_relaxed) / 1024);
    std::fflush(stderr);
  }
};

void sleepSeconds(double seconds) {
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

size_t countWords(const std::vector<std::string> &lines) {
  size_t words = 0;
  for (const auto &line : lines) {
    bool inWord = false;
    for (char c : line) {
      bool space = (c == ' ' || c == '\t');
      if (!space && !inWord) ++words;
      inWord = !space;
    }
  }
  return words;
}

/// Fans texts out to the AsyncService and blocks until every callback fired
/// (mirror of the JNI layer's collectAll).
template <typename Submit>
std::vector<Response> collectAll(std::vector<std::string> &&sources, Submit submit) {
  const size_t n = sources.size();
  std::vector<Response> responses(n);
  std::mutex mutex;
  std::condition_variable done;
  size_t pending = n;
  for (size_t i = 0; i < n; ++i) {
    submit(i, std::move(sources[i]), [&, i](Response &&response) {
      std::lock_guard<std::mutex> lock(mutex);
      responses[i] = std::move(response);
      if (--pending == 0) done.notify_all();
    });
  }
  std::unique_lock<std::mutex> lock(mutex);
  done.wait(lock, [&] { return pending == 0; });
  return responses;
}

// ---------------------------------------------------------------------------
// D0: release-lifecycle scenarios.
// ---------------------------------------------------------------------------

std::vector<Response> asyncTranslateAll(AsyncService &service, const std::shared_ptr<TranslationModel> &model,
                                        const std::vector<std::string> &corpus) {
  std::vector<std::string> sources = corpus;
  std::vector<ResponseOptions> options(sources.size());
  return collectAll(std::move(sources), [&](size_t i, std::string &&text, auto callback) {
    service.translate(model, std::move(text), std::move(callback), options[i]);
  });
}

/// Submit everything, then hand the caller a chance to drop its handle while
/// translation is still running, then wait. The `inflight` scenario.
template <typename BetweenSubmitAndWait>
std::vector<Response> asyncTranslateAllReleasingMidflight(AsyncService &service, std::shared_ptr<TranslationModel> &model,
                                                          const std::vector<std::string> &corpus,
                                                          BetweenSubmitAndWait between) {
  const size_t n = corpus.size();
  std::vector<Response> responses(n);
  std::mutex mutex;
  std::condition_variable done;
  size_t pending = n;
  ResponseOptions options;
  for (size_t i = 0; i < n; ++i) {
    std::string text = corpus[i];
    service.translate(model, std::move(text),
                      [&, i](Response &&response) {
                        std::lock_guard<std::mutex> lock(mutex);
                        responses[i] = std::move(response);
                        if (--pending == 0) done.notify_all();
                      },
                      options);
  }
  // Every request now owns its own shared_ptr copy, so dropping ours here is
  // race-free with respect to the workers.
  between();
  std::unique_lock<std::mutex> lock(mutex);
  done.wait(lock, [&] { return pending == 0; });
  return responses;
}

/// How the scenarios give a model handle back. "reset" is the pre-D0
/// behaviour (drop the shared_ptr and hope); "release" goes through the
/// service's release() closure. Kept switchable so a before/after table can be
/// produced from one binary.
enum class ReleaseMode { Reset, Release };

int runLifecycleScenario(const std::string &scenario, const std::vector<std::string> &corpus,
                         const std::vector<const char *> &configs, size_t workersFlag, ReleaseMode releaseMode) {
  Checkpointer cp;
  std::string summaryHash = "-";
  std::string extra;

  const bool blocking = (scenario == "blocking");
  size_t workers = workersFlag;
  if (scenario == "w1") workers = 1;
  if (scenario == "w2") workers = 2;
  if (scenario == "w4" || scenario == "w4-1sent") workers = 4;
  if (scenario == "inflight" && workers == 0) workers = 2;
  if (!blocking && workers == 0) workers = 1;

  std::vector<std::string> input = corpus;
  if (scenario == "w4-1sent") input.resize(std::min<size_t>(1, input.size()));

  const bool needsSecond = (scenario == "pivot" || scenario == "switch");
  if (needsSecond && configs.size() < 2) {
    std::cerr << "scenario " << scenario << " needs two model configs\n";
    return 2;
  }

  auto parse = [](const char *path) { return parseOptionsFromFilePath(path); };

  cp("00_process_start");

  if (blocking) {
    BlockingService::Config config;
    BlockingService service{config};
    // give(handle) is "hand the model back" in whichever mode is selected.
    auto give = [&](std::shared_ptr<TranslationModel> &handle) {
      lifecycle::event("harness_release_handle scenario=%s mode=%s", scenario.c_str(),
                       releaseMode == ReleaseMode::Release ? "release" : "reset");
      if (releaseMode == ReleaseMode::Release) {
        service.release(std::move(handle));
      } else {
        handle.reset();
      }
    };
    cp("01_service_created");
    auto model = std::make_shared<TranslationModel>(parse(configs[0]));
    cp("02_model_loaded");
    {
      std::vector<std::string> sources = input;
      std::vector<ResponseOptions> options(sources.size());
      summaryHash = hex16(responseHash(service.translateMultiple(model, std::move(sources), options)));
    }
    cp("03_translated");
    give(model);
    cp("04_released");
    sleepSeconds(1.0);
    cp("05_released_1s");
    sleepSeconds(3.0);
    cp("06_released_3s");
    if (configs.size() == 2) {
      // BlockingService has no workers: the packing caches live on THIS thread,
      // so one GEMM here is what triggers the generation-guard clear.
      auto other = std::make_shared<TranslationModel>(parse(configs[1]));
      std::vector<std::string> one{input.empty() ? std::string("hello") : input[0]};
      std::vector<ResponseOptions> options(1);
      service.translateMultiple(other, std::move(one), options);
      cp("07_other_model_gemm");
      give(other);
      cp("08_other_released");
    }
  } else {
    AsyncService::Config config;
    config.numWorkers = workers;
    AsyncService service{config};
    // give(handle) is "hand the model back" in whichever mode is selected.
    // `destroyed` records what release() reported, so the summary can say
    // whether the closure held.
    bool releaseReportedDestroyed = false;
    bool releaseCalled = false;
    auto give = [&](std::shared_ptr<TranslationModel> &handle, const char *what) {
      lifecycle::event("harness_release_handle scenario=%s what=%s mode=%s", scenario.c_str(), what,
                       releaseMode == ReleaseMode::Release ? "release" : "reset");
      if (releaseMode == ReleaseMode::Release) {
        releaseReportedDestroyed = service.release(std::move(handle));
        releaseCalled = true;
      } else {
        handle.reset();
      }
    };
    cp("01_service_created");
    auto model = service.createCompatibleModel(parse(configs[0]));
    std::shared_ptr<TranslationModel> second;
    if (scenario == "pivot") second = service.createCompatibleModel(parse(configs[1]));
    cp("02_model_loaded");

    if (scenario == "pivot") {
      std::vector<std::string> sources = input;
      std::vector<ResponseOptions> options(sources.size());
      auto responses = collectAll(std::move(sources), [&](size_t i, std::string &&text, auto callback) {
        service.pivot(model, second, std::move(text), std::move(callback), options[i]);
      });
      summaryHash = hex16(responseHash(responses));
      cp("03_translated");
      give(model, "first_model_only");
    } else if (scenario == "inflight") {
      auto responses = asyncTranslateAllReleasingMidflight(service, model, input, [&] {
        // Mid-flight: release() would block for the whole remaining corpus
        // (it waits for a worker turnaround while batches keep arriving), so
        // the in-flight case drops the handle and the closure runs at 04.
        lifecycle::event("harness_release_handle scenario=inflight what=midflight mode=reset");
        model.reset();
        cp("03a_released_midflight");
      });
      summaryHash = hex16(responseHash(responses));
      cp("03_translated");
    } else {
      summaryHash = hex16(responseHash(asyncTranslateAll(service, model, input)));
      cp("03_translated");
      give(model, "model");
    }
    if (scenario == "inflight" && releaseMode == ReleaseMode::Release) {
      // Nothing left to hand back -- the handle is already gone -- but the
      // service still holds worker-side references and caches.
      service.drain();
    }
    cp("04_released");
    sleepSeconds(1.0);
    cp("05_released_1s");
    sleepSeconds(3.0);
    cp("06_released_3s");

    if (scenario == "reload") {
      auto again = service.createCompatibleModel(parse(configs[0]));
      cp("07_reloaded");
      const std::string reloadHash = hex16(responseHash(asyncTranslateAll(service, again, input)));
      cp("08_reload_translated");
      extra += " reload_hash=" + reloadHash;
      extra += std::string(" reload_hash_match=") + (reloadHash == summaryHash ? "yes" : "NO");
      give(again, "reloaded_model");
      cp("09_reload_released");
    } else if (scenario == "switch") {
      auto other = service.createCompatibleModel(parse(configs[1]));
      cp("07_model_b_loaded");
      extra += " model_b_hash=" + hex16(responseHash(asyncTranslateAll(service, other, input)));
      cp("08_model_b_translated");
      give(other, "model_b");
      cp("09_model_b_released");
    } else if (scenario == "pivot") {
      // The second model is still resident; one sentence through it is the
      // next GEMM after the first model's teardown.
      std::vector<std::string> one{input.empty() ? std::string("hello") : input[0]};
      std::vector<ResponseOptions> options(1);
      collectAll(std::move(one), [&](size_t i, std::string &&text, auto callback) {
        service.translate(second, std::move(text), std::move(callback), options[i]);
      });
      cp("07_other_model_gemm");
      give(second, "second_model");
      cp("08_other_released");
    } else if (configs.size() == 2) {
      auto other = service.createCompatibleModel(parse(configs[1]));
      std::vector<std::string> one{input.empty() ? std::string("hello") : input[0]};
      std::vector<ResponseOptions> options(1);
      collectAll(std::move(one), [&](size_t i, std::string &&text, auto callback) {
        service.translate(other, std::move(text), std::move(callback), options[i]);
      });
      cp("07_other_model_gemm");
      give(other, "other_model");
      cp("08_other_released");
    }
    // inflight drops the handle by hand and only drains, so there is no
    // release() verdict to report there.
    if (releaseCalled) {
      extra += std::string(" release_reported_destroyed=") + (releaseReportedDestroyed ? "yes" : "no");
    }
  }

  cp("10_service_destroyed");
  std::fprintf(stderr,
               "[lifecycle] summary scenario=%s mode=%s workers=%zu sentences=%zu hash=%s dtor_seen_at=%s%s\n",
               scenario.c_str(), releaseMode == ReleaseMode::Release ? "release" : "reset", workers, input.size(),
               summaryHash.c_str(), cp.dtorSeenAt.c_str(), extra.c_str());
  std::fflush(stderr);
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  size_t workers = 0;  // 0 = BlockingService
  size_t cacheSize = 0;
  size_t repeat = 1;
  bool bench = false;
  bool cacheStatsWanted = false;
  bool memWanted = false;
  const char *dumpPrefix = nullptr;
  const char *lifecycleScenario = nullptr;
  const char *releaseModeFlag = "release";
  bool workersGiven = false;
  std::vector<const char *> configs;

  for (int i = 1; i < argc; ++i) {
    auto needValue = [&](const char *flag) {
      if (i + 1 >= argc) {
        std::cerr << flag << " needs a value\n";
        exit(2);
      }
      return std::stoul(argv[++i]);
    };
    if (std::strcmp(argv[i], "--workers") == 0) {
      workers = needValue("--workers");
      workersGiven = true;
    } else if (std::strcmp(argv[i], "--lifecycle") == 0) {
      if (i + 1 >= argc) { std::cerr << "--lifecycle needs a scenario\n" << kScenarioHelp; exit(2); }
      lifecycleScenario = argv[++i];
    } else if (std::strcmp(argv[i], "--release-mode") == 0) {
      if (i + 1 >= argc) { std::cerr << "--release-mode needs release|reset\n"; exit(2); }
      releaseModeFlag = argv[++i];
    } else if (std::strcmp(argv[i], "--cache-size") == 0) {
      cacheSize = needValue("--cache-size");
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      repeat = std::max<size_t>(1, needValue("--repeat"));
    } else if (std::strcmp(argv[i], "--bench") == 0) {
      bench = true;
    } else if (std::strcmp(argv[i], "--cache-stats") == 0) {
      cacheStatsWanted = true;
    } else if (std::strcmp(argv[i], "--mem") == 0) {
      memWanted = true;
    } else if (std::strcmp(argv[i], "--dump-prefix") == 0) {
      if (i + 1 >= argc) { std::cerr << "--dump-prefix needs a value\n"; exit(2); }
      dumpPrefix = argv[++i];
    } else {
      configs.push_back(argv[i]);
    }
  }
  if (configs.empty() || configs.size() > 2) {
    std::cerr << "usage: smoke [--workers N] [--cache-size N] [--repeat R] [--bench] [--lifecycle S] "
                 "<model-config.yml> [<second-config.yml>] < text-lines\n"
              << kScenarioHelp;
    return 2;
  }

  std::vector<std::string> corpus;
  for (std::string line; std::getline(std::cin, line);) corpus.push_back(line);
  const size_t sentences = corpus.size();
  const size_t words = countWords(corpus);

  if (lifecycleScenario != nullptr) {
    ReleaseMode releaseMode;
    if (std::strcmp(releaseModeFlag, "release") == 0) {
      releaseMode = ReleaseMode::Release;
    } else if (std::strcmp(releaseModeFlag, "reset") == 0) {
      releaseMode = ReleaseMode::Reset;
    } else {
      std::cerr << "--release-mode must be release or reset\n";
      return 2;
    }
    return runLifecycleScenario(lifecycleScenario, corpus, configs, workersGiven ? workers : 0, releaseMode);
  }

  auto emit = [&](const std::string &key, double value) {
    if (bench) std::cerr << "[bench] " << key << "=" << value << "\n";
  };

  // Memory ledger: one sample per stage, always printed (independent of --bench)
  // so a memory run does not have to carry the timing noise with it.
  auto emitMem = [&](const char *stage) {
    if (!memWanted) return;
    FootprintSample sample = footprintSample();
    fprintf(stderr, "[mem] %s footprint_mb=%.1f peak_rss_mb=%.1f\n", stage, sample.footprintMb,
            sample.peakMb);
  };

  if (bench) {
    // Same probe the JNI layer logs at service creation, plus the cache
    // parameters ruy's block_map tunes with (dummy values 32768/524288 mean
    // cpuinfo failed and ruy is blocking for a 512KB last-level cache).
    ruy::Context probe;
    ruy::CpuInfo cpuInfo;
    fprintf(stderr, "[bench] ruy_paths=0x%x dotprod=%d cache_local=%d cache_llc=%d\n",
            static_cast<int>(probe.get_runtime_enabled_paths()), cpuInfo.NeonDotprod() ? 1 : 0,
            cpuInfo.CacheParams().local_cache_size, cpuInfo.CacheParams().last_level_cache_size);
  }

  std::vector<Response> responses;
  auto runPasses = [&](auto translateOnce) {
    for (size_t pass = 0; pass < repeat; ++pass) {
      std::vector<std::string> sources = corpus;  // fresh copy, translate consumes it
      std::vector<ResponseOptions> options(sources.size());
      auto start = std::chrono::steady_clock::now();
      responses = translateOnce(std::move(sources), options);
      double ms = msSince(start);
      emit("pass" + std::to_string(pass) + "_ms", ms);
      if (pass == 0) {
        emit("sentences_per_s", sentences / (ms / 1000.0));
        emit("words_per_s", words / (ms / 1000.0));
      }
      // Determinism probes, after the timer stops: FNV-1a over the pass's
      // output, and optionally the full text of every pass to files.
      if (bench) {
        std::cerr << "[bench] pass" << pass << "_hash=" << hex16(responseHash(responses)) << "\n";
      }
      if (dumpPrefix) {
        std::ofstream out(std::string(dumpPrefix) + ".pass" + std::to_string(pass) + ".txt");
        for (auto &r : responses) out << r.target.text << "\n";
      }
    }
  };

  TranslationCache::Stats cacheStats;
  if (workers == 0) {
    BlockingService::Config config;
    config.cacheSize = cacheSize;
    BlockingService service{config};
    emitMem("baseline");

    auto loadStart = std::chrono::steady_clock::now();
    auto makeModel = [&](const char *path) {
      return std::make_shared<TranslationModel>(parseOptionsFromFilePath(path));
    };
    auto model = makeModel(configs[0]);
    auto second = (configs.size() == 2) ? makeModel(configs[1]) : nullptr;
    emit("load_ms", msSince(loadStart));
    emitMem("after_load");

    runPasses([&](std::vector<std::string> &&sources, const std::vector<ResponseOptions> &options) {
      return second ? service.pivotMultiple(model, second, std::move(sources), options)
                    : service.translateMultiple(model, std::move(sources), options);
    });
    if (cacheStatsWanted) cacheStats = service.cacheStats();
    emitMem("steady");
    model.reset();
    second.reset();
    emitMem("after_model_reset");
  } else {
    AsyncService::Config config;
    config.numWorkers = workers;
    config.cacheSize = cacheSize;
    AsyncService service{config};
    emitMem("baseline");

    auto loadStart = std::chrono::steady_clock::now();
    auto makeModel = [&](const char *path) {
      return service.createCompatibleModel(parseOptionsFromFilePath(path));
    };
    auto model = makeModel(configs[0]);
    auto second = (configs.size() == 2) ? makeModel(configs[1]) : nullptr;
    emit("load_ms", msSince(loadStart));
    emitMem("after_load");

    runPasses([&](std::vector<std::string> &&sources, const std::vector<ResponseOptions> &options) {
      return collectAll(std::move(sources), [&](size_t i, std::string &&text, auto callback) {
        if (second) {
          service.pivot(model, second, std::move(text), std::move(callback), options[i]);
        } else {
          service.translate(model, std::move(text), std::move(callback), options[i]);
        }
      });
    });
    if (cacheStatsWanted) cacheStats = service.cacheStats();
    emitMem("steady");
    model.reset();
    second.reset();
    emitMem("after_model_reset");
  }

  // The service outlives the model handles above: its batching pool keeps a
  // strong reference to every model it has ever seen
  // (aggregate_batching_pool.h aggregateQueue_), so "after_model_reset" is not
  // the real unload point -- this one is.
  emitMem("after_service_destroyed");

  if (cacheStatsWanted) {
    emit("cache_hits", static_cast<double>(cacheStats.hits));
    emit("cache_misses", static_cast<double>(cacheStats.misses));
  }
  emit("peak_rss_mb", peakRssMb());

  for (auto &response : responses) std::cout << response.target.text << "\n";
  return 0;
}
