#include "service.h"

#include <string>
#include <utility>

#if defined(__ANDROID__) || defined(__GLIBC__)
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#include "batch.h"
#include "byte_array_util.h"
#include "common/lifecycle.h"                // D0: worker/service lifecycle tracing
#include "definitions.h"
#include "tensors/cpu/integer_common.h"  // D0: releaseThreadPackingCaches()

namespace marian {
namespace bergamot {

namespace {

// Combines two responses with first.target == second.source mapping alignments etc accordingly.
// There are several constraints which are matched by only the pivoting workflow in <>Service source, therefore this
// function is not for external use and in a hidden namespace.
Response combine(Response &&first, Response &&second) {
  Response combined;

  // Compute alignment first using internal matrices and mappings.
  if (first.alignments.size()) {
    combined.alignments = remapAlignments(first, second);
  }

  combined.source = std::move(first.source);
  combined.target = std::move(second.target);
  combined.qualityScores = std::move(second.qualityScores);

  return combined;
}

std::optional<TranslationCache> makeOptionalCache(size_t size, size_t mutexBuckets) {
  return size > 0 ? std::make_optional<TranslationCache>(size, mutexBuckets) : std::nullopt;
}

// D0: hand freed pages back to the OS. Freeing to the allocator is not the
// same as shrinking RSS -- bionic's scudo and macOS libmalloc both keep
// per-thread magazines that otherwise only drain when the thread exits, so a
// release that frees tens of MB can leave RSS completely flat. Only called on
// the explicit release path (onTrimMemory-ish moments), never during
// translation: a purge costs milliseconds and re-faults pages afterwards.
void purgeAllocator() {
#if defined(__ANDROID__)
#if defined(M_PURGE_ALL)
  mallopt(M_PURGE_ALL, 0);
#elif defined(M_PURGE)
  mallopt(M_PURGE, 0);
#endif
#elif defined(__APPLE__)
  malloc_zone_pressure_relief(nullptr, 0);
#elif defined(__GLIBC__)
  malloc_trim(0);
#endif
}

}  // namespace

BlockingService::BlockingService(const BlockingService::Config &config)
    : config_(config),
      requestId_(0),
      batchingPool_(),
      cache_(makeOptionalCache(config.cacheSize, /*mutexBuckets = */ 1)),
      logger_(config.logger) {}

std::vector<Response> BlockingService::translateMultiple(std::shared_ptr<TranslationModel> translationModel,
                                                         std::vector<std::string> &&sources,
                                                         const std::vector<ResponseOptions> &responseOptions) {
  std::vector<HTML> htmls;
  for (size_t i = 0; i < sources.size(); i++) {
    htmls.emplace_back(std::move(sources[i]), responseOptions[i].HTML);
  }
  std::vector<Response> responses = translateMultipleRaw(translationModel, std::move(sources), responseOptions);
  for (size_t i = 0; i < responses.size(); i++) {
    htmls[i].restore(responses[i]);
  }

  return responses;
}

std::vector<Response> BlockingService::translateMultipleRaw(std::shared_ptr<TranslationModel> translationModel,
                                                            std::vector<std::string> &&sources,
                                                            const std::vector<ResponseOptions> &responseOptions) {
  std::vector<Response> responses;
  responses.resize(sources.size());

  for (size_t i = 0; i < sources.size(); i++) {
    auto callback = [i, &responses](Response &&response) { responses[i] = std::move(response); };  //
    Ptr<Request> request =
        translationModel->makeRequest(requestId_++, std::move(sources[i]), callback, responseOptions[i], cache_);
    batchingPool_.enqueueRequest(translationModel, request);
  }

  Batch batch;
  Ptr<TranslationModel> model{nullptr};
  while (batchingPool_.generateBatch(model, batch)) {
    model->translateBatch(/*deviceId=*/0, batch);
  }

  return responses;
}

bool BlockingService::release(std::shared_ptr<TranslationModel> &&model) {
  std::weak_ptr<TranslationModel> observer = model;
  if (lifecycle::enabled() && model) lifecycle::event("service_release id=%zu", model->modelId());
  model.reset();
  // translateMultipleRaw() drains the aggregate queue before returning, so
  // there is nothing to clear there; the packing caches, however, belong to
  // this very thread and would otherwise wait for a GEMM that may never come.
  batchingPool_.clear();
  marian::cpu::integer::releaseThreadPackingCaches();
  purgeAllocator();
  const bool destroyed = observer.expired();
  if (lifecycle::enabled()) lifecycle::event("service_release_done destroyed=%d", destroyed ? 1 : 0);
  return destroyed;
}

std::vector<Response> BlockingService::pivotMultiple(std::shared_ptr<TranslationModel> first,
                                                     std::shared_ptr<TranslationModel> second,
                                                     std::vector<std::string> &&sources,
                                                     const std::vector<ResponseOptions> &responseOptions) {
  std::vector<HTML> htmls;
  for (size_t i = 0; i < sources.size(); i++) {
    htmls.emplace_back(std::move(sources[i]), responseOptions[i].HTML);
  }

  // Translate source to pivots. This is same as calling translateMultiple.
  std::vector<Response> sourcesToPivots;
  sourcesToPivots = translateMultipleRaw(first, std::move(sources), responseOptions);

  // Translate pivots to targets, after we have outputs at pivot from first round. We cannot use translateMultiple here
  // because need consistency at pivot on both sides.
  std::vector<Response> pivotsToTargets;
  pivotsToTargets.resize(sourcesToPivots.size());

  for (size_t i = 0; i < sourcesToPivots.size(); i++) {
    AnnotatedText intermediate =
        sourcesToPivots[i].target;  // We cannot eliminate this copy, as we need two versions of intermediate. Holding
                                    // it in allows further use in makePivotRequest
    auto callback = [i, &pivotsToTargets](Response &&response) { pivotsToTargets[i] = std::move(response); };  //

    Ptr<Request> request =
        second->makePivotRequest(requestId_++, std::move(intermediate), callback, responseOptions[i], cache_);
    batchingPool_.enqueueRequest(second, request);
  }

  Batch batch;
  Ptr<TranslationModel> model{nullptr};
  while (batchingPool_.generateBatch(model, batch)) {
    model->translateBatch(/*deviceId=*/0, batch);
  }

  // Combine both sides. They're associated by indices.
  std::vector<Response> finalResponses;
  for (size_t i = 0; i < sourcesToPivots.size(); i++) {
    Response finalResponse = combine(std::move(sourcesToPivots[i]), std::move(pivotsToTargets[i]));
    finalResponses.push_back(std::move(finalResponse));
  }

  for (size_t i = 0; i < finalResponses.size(); i++) {
    htmls[i].restore(finalResponses[i]);
  }

  return finalResponses;
}

AsyncService::AsyncService(const AsyncService::Config &config)
    : requestId_(0),
      config_(config),
      safeBatchingPool_(),
      cache_(makeOptionalCache(config_.cacheSize, /*mutexBuckets=*/config_.numWorkers)),
      logger_(config.logger) {
  ABORT_IF(config_.numWorkers == 0, "Number of workers should be at least 1 in a threaded workflow");
  if (lifecycle::enabled()) lifecycle::event("service_ctor workers=%zu", config_.numWorkers);
  workers_.reserve(config_.numWorkers);
  safeBatchingPool_.setConsumerCount(config_.numWorkers);
  for (size_t cpuId = 0; cpuId < config_.numWorkers; cpuId++) {
    workers_.emplace_back([cpuId, this] {
      // Consumer thread main-loop. Note that this is an infinite-loop unless the monitor is explicitly told to
      // shutdown, which happens in the destructor for this class.
      Batch batch;
      Ptr<TranslationModel> translationModel{nullptr};
      size_t seenMaintenanceEpoch = 0;
      for (;;) {
        bool maintenanceDue = false;
        size_t sentences = safeBatchingPool_.generateBatch(seenMaintenanceEpoch, maintenanceDue, translationModel,
                                                           batch);
        if (maintenanceDue) {
          // D0: everything this thread holds on a model's behalf goes here.
          batch.clear();
          translationModel.reset();
          marian::cpu::integer::releaseThreadPackingCaches();
          purgeAllocator();
          if (lifecycle::enabled()) lifecycle::event("worker_maintenance cpu=%zu", cpuId);
          safeBatchingPool_.ackMaintenance();
          continue;
        }
        if (sentences == 0) break;  // shutdown
        translationModel->translateBatch(cpuId, batch);
        // D0: drop the batch's RequestSentences and this worker's owning model
        // reference before blocking again. Without this the last batch's model
        // stays alive for as long as the worker sits idle -- which is forever,
        // in an app that has stopped translating.
        batch.clear();
        translationModel.reset();
      }
      if (lifecycle::enabled()) lifecycle::event("worker_shutdown cpu=%zu", cpuId);
    });
  }
}

bool AsyncService::release(std::shared_ptr<TranslationModel> &&model) {
  std::weak_ptr<TranslationModel> observer = model;
  if (lifecycle::enabled() && model) lifecycle::event("service_release id=%zu", model->modelId());
  model.reset();
  safeBatchingPool_.runMaintenance();
  // The model itself was freed on whichever thread dropped the last reference
  // -- possibly this one -- so purge here too, not only on the workers.
  purgeAllocator();
  const bool destroyed = observer.expired();
  if (lifecycle::enabled()) lifecycle::event("service_release_done destroyed=%d", destroyed ? 1 : 0);
  return destroyed;
}

void AsyncService::drain() { safeBatchingPool_.runMaintenance(); }

void AsyncService::clear() { safeBatchingPool_.clear(); }

AsyncService::~AsyncService() {
  if (lifecycle::enabled()) lifecycle::event("service_dtor_enter workers=%zu", workers_.size());
  safeBatchingPool_.shutdown();
  for (std::thread &worker : workers_) {
    assert(worker.joinable());
    worker.join();
  }
  workers_.clear();
  if (lifecycle::enabled()) {
    lifecycle::event("service_dtor_done live_models=%ld live_graphs=%ld",
                     lifecycle::liveModels().load(std::memory_order_relaxed),
                     lifecycle::liveGraphs().load(std::memory_order_relaxed));
  }
}

void AsyncService::pivot(std::shared_ptr<TranslationModel> first, std::shared_ptr<TranslationModel> second,
                         std::string &&source, CallbackType clientCallback, const ResponseOptions &responseOptions) {
  Ptr<HTML> html = std::make_shared<HTML>(std::move(source), responseOptions.HTML);
  // This is callback chaining or CPS due to async.

  // We create a callback which feeds the result of first into a second translation (internalCallback), which is
  // supplied with a callback (joiningCallback) which merges both results and creates our final response.
  //

  auto internalCallback = [this, clientCallback, second, responseOptions, html](Response &&sourceToPivot) {
    // We cannot eliminate the following copy, as we need two versions of intermediate. Holding
    // it in a copy allows moving the response into the lambda below.

    AnnotatedText intermediate = sourceToPivot.target;

    // https://stackoverflow.com/a/65606554/4565794
    // Move semantics only work on mutable lambdas, and can only be done once. It's only once in our case, so issok.
    auto joiningCallback = [this, sourceToPivot = std::move(sourceToPivot), clientCallback,
                            html](Response &&pivotToTarget) mutable {
      // We have both Responses at this callback, sourceToPivot is moved in, second half will be available when
      // complete.
      Response finalResponse = combine(std::move(sourceToPivot), std::move(pivotToTarget));

      // Sentences should be consistent now, give way to client.
      html->restore(finalResponse);
      clientCallback(std::move(finalResponse));
    };

    // Second call.
    Ptr<Request> request =
        second->makePivotRequest(requestId_++, std::move(intermediate), joiningCallback, responseOptions, cache_);
    safeBatchingPool_.enqueueRequest(second, request);
  };

  // First call.
  translateRaw(first, std::move(source), internalCallback, responseOptions);
}

void AsyncService::translate(std::shared_ptr<TranslationModel> translationModel, std::string &&source,
                             CallbackType callback, const ResponseOptions &responseOptions) {
  // Producer thread, a call to this function adds new work items. If batches are available, notifies workers waiting.
  Ptr<HTML> html = std::make_shared<HTML>(std::move(source), responseOptions.HTML);
  auto internalCallback = [html, callback](Response &&response) {
    html->restore(response);
    callback(std::move(response));
  };

  translateRaw(translationModel, std::move(source), internalCallback, responseOptions);
}

void AsyncService::translateRaw(std::shared_ptr<TranslationModel> translationModel, std::string &&source,
                                CallbackType callback, const ResponseOptions &responseOptions) {
  // Producer thread, a call to this function adds new work items. If batches are available, notifies workers waiting.
  Ptr<Request> request =
      translationModel->makeRequest(requestId_++, std::move(source), callback, responseOptions, cache_);
  safeBatchingPool_.enqueueRequest(translationModel, request);
}

}  // namespace bergamot
}  // namespace marian
