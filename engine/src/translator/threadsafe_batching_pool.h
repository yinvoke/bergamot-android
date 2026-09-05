/* Thread-safe wrapper around BatchingPool or AggregateBatchingPool, made generic with templates. */
#ifndef SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_H_
#define SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_H_

#include <condition_variable>
#include <mutex>

#include "aggregate_batching_pool.h"
#include "batching_pool.h"
#include "common/options.h"
#include "definitions.h"
#include "translation_model.h"

namespace marian {
namespace bergamot {

/// The following mechanism operates in a multithreaded async-workflow guarding access to the pushes to the structure
/// keeping sentences bucketed by length and sorted by priority.
///
/// This is a wrap of a producer-consumer queue implemented as a monitor, where there is a mutex guarding the
/// underlying data structure (BatchingPoolType) and (worker/consumer) threads waiting on a condition variable and the
/// queuing thread producing and notifying waiting threads (consumers) through the same condition variable.
///
/// Originally written by for a single model (where items are produce: Request, consume: Batch), converted to
/// also work for multiple models where items are produce: (TranslationModel, Request), consume: (TranlsationModel,
/// Batch). This is accomplished by template parameter packs.
///
/// Requires BatchingPoolType to implement the following:
///
/// * produce: `size_t enqueueRequest(...)` (returns number elements produced)
/// * consume: `size_t generateBatch(...)` (returns number of elements available to be consumed)

template <class BatchingPoolType>
class ThreadsafeBatchingPool {
 public:
  template <class... Args>
  ThreadsafeBatchingPool(Args &&...args);
  ~ThreadsafeBatchingPool();

  template <class... Args>
  void enqueueRequest(Args &&...args);

  /// Consumer side. Blocks until there is a batch, a shutdown, or a
  /// maintenance request (D0).
  ///
  /// @param [in,out] seenMaintenanceEpoch: consumer-owned; the epoch this
  /// consumer has already serviced. Start it at 0.
  /// @param [out] maintenanceDue: set when this consumer owes a maintenance
  /// pass. The return value is then 0 and no batch was produced; run the pass
  /// and call ackMaintenance().
  /// @returns sentences in the produced batch; 0 means shutdown (or, with
  /// maintenanceDue set, a maintenance wake-up).
  template <class... Args>
  size_t generateBatch(size_t &seenMaintenanceEpoch, bool &maintenanceDue, Args &&...args);

  // Removes any pending requests from the batching pool.
  void clear();

  /// D0: how many consumers call generateBatch. Must be set before they start;
  /// runMaintenance() waits for exactly this many acknowledgements.
  void setConsumerCount(size_t consumers);

  /// D0, producer side. Wakes every consumer, has each run one maintenance
  /// pass (drop its cached model reference and its GEMM packing caches), and
  /// blocks until all of them have acknowledged. Also drops the aggregate
  /// queue's model references when nothing is pending.
  ///
  /// Bounded by one in-flight batch: a consumer inside translateBatch only
  /// notices on its way back round, but the maintenance check is ahead of the
  /// batch check, so queued work cannot starve it.
  ///
  /// MUST NOT be called from a consumer thread (i.e. from a translation
  /// callback): that consumer could never acknowledge and this would deadlock.
  void runMaintenance();

  /// Consumer side, after finishing a maintenance pass.
  void ackMaintenance();

  // Signals shut down of batching pool. After this no new requests can be enqueued,
  // but all enqueued requests will be processed. To prevent this from happening,
  // call `clear()` before `shutdown()`.
  void shutdown();

 private:
  BatchingPoolType backend_;

  // Number of sentences in backend_;
  size_t enqueued_;

  // Are we shutting down?
  bool shutdown_;

  // Lock on this object. Guards every member below as well as backend_,
  // enqueued_ and shutdown_.
  std::mutex mutex_;

  // Signaled when there are sentences to translate, on shutdown, and on a new
  // maintenance epoch.
  std::condition_variable work_;

  // D0 maintenance protocol.
  // Lock order, where both are taken: maintenanceSerial_ then mutex_. Never
  // the reverse; consumers only ever take mutex_.
  std::mutex maintenanceSerial_;         // serialises concurrent runMaintenance()
  std::condition_variable maintenanceDone_;  // signaled by ackMaintenance()
  size_t consumers_{0};
  size_t maintenanceEpoch_{0};
  size_t maintenanceAcked_{0};
};

}  // namespace bergamot
}  // namespace marian

#define SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_IMPL
#include "threadsafe_batching_pool.cpp"
#undef SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_IMPL

#endif  // SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_H_
