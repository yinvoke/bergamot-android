// Identical harness for v0.1.0 and v0.2.0. No engine changes.
// Usage: smoke workers config.yml [pivot-target.yml] < corpus.txt
// First-pass latency includes service/model creation and lazy weight loading.
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"
using namespace marian::bergamot;
using Clock = std::chrono::steady_clock;

double memoryMiB(const std::string& field) {
  std::ifstream input("/proc/self/status");
  for (std::string line; std::getline(input, line);) {
    if (line.rfind(field + ":", 0) == 0) {
      return std::stod(line.substr(field.size() + 1)) / 1024.0;
    }
  }
  throw std::runtime_error("missing memory field " + field);
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  const int workers = std::stoi(argv[1]);
  if (workers < 1) return 2;
  std::vector<std::string> corpus;
  for (std::string line; std::getline(std::cin, line);) corpus.push_back(line);
  if (corpus.empty()) return 2;
  std::cout << std::fixed << std::setprecision(3);
  const auto coldStart = Clock::now();
  AsyncService::Config config;
  config.numWorkers = workers;
  config.cacheSize = 0;
  AsyncService service{config};
  auto model = service.createCompatibleModel(parseOptionsFromFilePath(argv[2]));
  auto second = argc == 4 ? service.createCompatibleModel(parseOptionsFromFilePath(argv[3])) : nullptr;
  for (int pass = 0; pass < 3; ++pass) {
    auto sources = corpus;
    std::vector<Response> responses(corpus.size());
    std::mutex mutex;
    std::condition_variable done;
    size_t pending = corpus.size();
    const auto start = pass == 0 ? coldStart : Clock::now();
    for (size_t i = 0; i < sources.size(); ++i) {
      auto callback = [&, i](Response&& response) {
        std::lock_guard<std::mutex> lock(mutex);
        responses[i] = std::move(response);
        if (--pending == 0) done.notify_all();
      };
      if (second) service.pivot(model, second, std::move(sources[i]), std::move(callback), ResponseOptions{});
      else service.translate(model, std::move(sources[i]), std::move(callback), ResponseOptions{});
    }
    {
      std::unique_lock<std::mutex> lock(mutex);
      done.wait(lock, [&] { return pending == 0; });
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const double peak = memoryMiB("VmHWM");
    const double rss = memoryMiB("VmRSS");
    uint64_t hash = 14695981039346656037ULL;
    size_t bytes = 0;
    for (const auto& response : responses) {
      bytes += response.target.text.size();
      for (unsigned char c : response.target.text) { hash ^= c; hash *= 1099511628211ULL; }
      hash ^= '\n'; hash *= 1099511628211ULL;
    }
    std::cout << "{\"pass\":" << pass << ",\"workers\":" << workers
      << ",\"sentences\":" << corpus.size() << ",\"elapsed_ms\":" << ms
      << ",\"peak_rss_mib\":" << peak << ",\"rss_mib\":" << rss
      << ",\"output_bytes\":" << bytes << ",\"output_hash\":\"" << std::hex << hash
      << std::dec << "\"}" << std::endl;
  }
}
