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
//   --dump-prefix P write each pass's output to P.passN.txt (determinism diffs)
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
#include <vector>

#include <sys/resource.h>

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

double peakRssMb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0.0;
#ifdef __APPLE__
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
#endif
}

double msSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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

}  // namespace

int main(int argc, char *argv[]) {
  size_t workers = 0;  // 0 = BlockingService
  size_t cacheSize = 0;
  size_t repeat = 1;
  bool bench = false;
  bool cacheStatsWanted = false;
  const char *dumpPrefix = nullptr;
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
    } else if (std::strcmp(argv[i], "--cache-size") == 0) {
      cacheSize = needValue("--cache-size");
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      repeat = std::max<size_t>(1, needValue("--repeat"));
    } else if (std::strcmp(argv[i], "--bench") == 0) {
      bench = true;
    } else if (std::strcmp(argv[i], "--cache-stats") == 0) {
      cacheStatsWanted = true;
    } else if (std::strcmp(argv[i], "--dump-prefix") == 0) {
      if (i + 1 >= argc) { std::cerr << "--dump-prefix needs a value\n"; exit(2); }
      dumpPrefix = argv[++i];
    } else {
      configs.push_back(argv[i]);
    }
  }
  if (configs.empty() || configs.size() > 2) {
    std::cerr << "usage: smoke [--workers N] [--cache-size N] [--repeat R] [--bench] "
                 "<model-config.yml> [<second-config.yml>] < text-lines\n";
    return 2;
  }

  std::vector<std::string> corpus;
  for (std::string line; std::getline(std::cin, line);) corpus.push_back(line);
  const size_t sentences = corpus.size();
  const size_t words = countWords(corpus);

  auto emit = [&](const std::string &key, double value) {
    if (bench) std::cerr << "[bench] " << key << "=" << value << "\n";
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
        uint64_t h = 1469598103934665603ull;
        for (auto &r : responses) {
          for (unsigned char c : r.target.text) h = (h ^ c) * 1099511628211ull;
          h = (h ^ '\n') * 1099511628211ull;
        }
        char buf[17];
        snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
        std::cerr << "[bench] pass" << pass << "_hash=" << buf << "\n";
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

    auto loadStart = std::chrono::steady_clock::now();
    auto makeModel = [&](const char *path) {
      return std::make_shared<TranslationModel>(parseOptionsFromFilePath(path));
    };
    auto model = makeModel(configs[0]);
    auto second = (configs.size() == 2) ? makeModel(configs[1]) : nullptr;
    emit("load_ms", msSince(loadStart));

    runPasses([&](std::vector<std::string> &&sources, const std::vector<ResponseOptions> &options) {
      return second ? service.pivotMultiple(model, second, std::move(sources), options)
                    : service.translateMultiple(model, std::move(sources), options);
    });
    if (cacheStatsWanted) cacheStats = service.cacheStats();
  } else {
    AsyncService::Config config;
    config.numWorkers = workers;
    config.cacheSize = cacheSize;
    AsyncService service{config};

    auto loadStart = std::chrono::steady_clock::now();
    auto makeModel = [&](const char *path) {
      return service.createCompatibleModel(parseOptionsFromFilePath(path));
    };
    auto model = makeModel(configs[0]);
    auto second = (configs.size() == 2) ? makeModel(configs[1]) : nullptr;
    emit("load_ms", msSince(loadStart));

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
  }

  if (cacheStatsWanted) {
    emit("cache_hits", static_cast<double>(cacheStats.hits));
    emit("cache_misses", static_cast<double>(cacheStats.misses));
  }
  emit("peak_rss_mb", peakRssMb());

  for (auto &response : responses) std::cout << response.target.text << "\n";
  return 0;
}
