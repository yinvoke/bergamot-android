
#ifndef SRC_BERGAMOT_THREADSAFE_BATCHING_POOL_IMPL
#error "This is an impl file and must not be included directly!"
#endif

#include <cassert>

namespace marian {
namespace bergamot {

template <class BatchingPoolType>
template <class... Args>
ThreadsafeBatchingPool<BatchingPoolType>::ThreadsafeBatchingPool(Args &&...args)
    : backend_(std::forward<Args>(args)...), enqueued_(0), shutdown_(false) {}

template <class BatchingPoolType>
ThreadsafeBatchingPool<BatchingPoolType>::~ThreadsafeBatchingPool() {
  shutdown();
}

template <class BatchingPoolType>
template <class... Args>
void ThreadsafeBatchingPool<BatchingPoolType>::enqueueRequest(Args &&...args) {
  std::unique_lock<std::mutex> lock(mutex_);
  assert(!shutdown_);
  enqueued_ += backend_.enqueueRequest(std::forward<Args>(args)...);
  work_.notify_all();
}

template <class BatchingPoolType>
void ThreadsafeBatchingPool<BatchingPoolType>::clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  backend_.clear();
  enqueued_ = 0;
}

template <class BatchingPoolType>
void ThreadsafeBatchingPool<BatchingPoolType>::shutdown() {
  std::unique_lock<std::mutex> lock(mutex_);
  shutdown_ = true;
  work_.notify_all();
  // A runMaintenance() blocked on acknowledgements must not outlive shutdown.
  maintenanceDone_.notify_all();
}

template <class BatchingPoolType>
void ThreadsafeBatchingPool<BatchingPoolType>::setConsumerCount(size_t consumers) {
  std::unique_lock<std::mutex> lock(mutex_);
  consumers_ = consumers;
}

template <class BatchingPoolType>
void ThreadsafeBatchingPool<BatchingPoolType>::runMaintenance() {
  std::unique_lock<std::mutex> serial(maintenanceSerial_);
  std::unique_lock<std::mutex> lock(mutex_);
  if (consumers_ == 0 || shutdown_) return;
  // Only safe with nothing pending: the aggregate queue is how a model's
  // batching pool is reached at all, so dropping it while sentences are
  // enqueued would strand them. generateBatch() already clears at that point;
  // this covers the paths that do not go through it.
  if (enqueued_ == 0) backend_.clear();
  ++maintenanceEpoch_;
  maintenanceAcked_ = 0;
  work_.notify_all();
  maintenanceDone_.wait(lock, [this]() { return maintenanceAcked_ >= consumers_ || shutdown_; });
}

template <class BatchingPoolType>
void ThreadsafeBatchingPool<BatchingPoolType>::ackMaintenance() {
  std::unique_lock<std::mutex> lock(mutex_);
  ++maintenanceAcked_;
  maintenanceDone_.notify_all();
}

template <class BatchingPoolType>
template <class... Args>
size_t ThreadsafeBatchingPool<BatchingPoolType>::generateBatch(size_t &seenMaintenanceEpoch, bool &maintenanceDue,
                                                               Args &&...args) {
  std::unique_lock<std::mutex> lock(mutex_);
  work_.wait(lock, [this, &seenMaintenanceEpoch]() {
    return enqueued_ || shutdown_ || maintenanceEpoch_ != seenMaintenanceEpoch;
  });
  // Maintenance is checked before work on purpose: a release must not wait for
  // a backlog to drain.
  if (maintenanceEpoch_ != seenMaintenanceEpoch) {
    seenMaintenanceEpoch = maintenanceEpoch_;
    maintenanceDue = true;
    return 0;
  }
  maintenanceDue = false;
  size_t sentencesInBatch = backend_.generateBatch(std::forward<Args>(args)...);
  assert(sentencesInBatch > 0 || shutdown_);
  enqueued_ -= sentencesInBatch;
  // D0: with nothing left pending, the aggregate queue's owning references to
  // TranslationModels are dead weight -- it would otherwise hold them until
  // some later generateBatch found that model's pool empty, which only happens
  // once new work arrives. enqueueRequest() re-inserts, so this is safe.
  if (enqueued_ == 0) backend_.clear();
  return sentencesInBatch;
}

}  // namespace bergamot
}  // namespace marian
