#include "translation_model.h"

#include "batch.h"
#include "byte_array_util.h"
#include "cache.h"
#include "common/lifecycle.h"  // D0: model/backend lifecycle tracing
#include "common/logging.h"
#include "data/corpus.h"
#include "data/text_input.h"
#include "html.h"
#include "parser.h"
#include "tensors/cpu/integer_common.h"  // PATCH B: prepack generation counter
#include "translator/beam_search.h"

namespace marian {
namespace bergamot {

std::atomic<size_t> TranslationModel::modelCounter_ = 0;

TranslationModel::TranslationModel(const Config &options, MemoryBundle &&memory /*=MemoryBundle{}*/,
                                   size_t replicas /*=1*/)
    : modelId_(modelCounter_++),
      options_(options),
      memory_(std::move(memory)),
      vocabs_(options, std::move(memory_.vocabs)),
#if !defined(WASM)
      textProcessor_(options, vocabs_, std::move(memory_.ssplitPrefixFile)),
#elif defined(WASM)
      textProcessor_(vocabs_),
#endif // defined(WASM)
      batchingPool_(options),
      qualityEstimator_(createQualityEstimator(getQualityEstimatorModel(memory, options))) {
  ABORT_IF(replicas == 0, "At least one replica needs to be created.");
  backend_.resize(replicas);
  modelMemory_ = std::make_shared<std::vector<AlignedMemory>>(std::move(memory_.models));

  // Try to load shortlist from memory-bundle. If not available, try to load from options_;

  int srcIdx = 0, trgIdx = 1;
  // vocabs_->sources().front() is invoked as we currently only support one source vocab
  bool shared_vcb = (vocabs_.sources().front() == vocabs_.target());

  if (memory_.shortlist.size() > 0 && memory_.shortlist.begin() != nullptr) {
    bool check = options_->get<bool>("check-bytearray", false);
    shortlistGenerator_ = New<data::BinaryShortlistGenerator>(memory_.shortlist.begin(), memory_.shortlist.size(),
                                                              vocabs_.sources().front(), vocabs_.target(), srcIdx,
                                                              trgIdx, shared_vcb, check);
  } else if (options_->hasAndNotEmpty("shortlist")) {
    // Changed to BinaryShortlistGenerator to enable loading binary shortlist file
    // This class also supports text shortlist file
    shortlistGenerator_ = New<data::BinaryShortlistGenerator>(options_, vocabs_.sources().front(), vocabs_.target(),
                                                              srcIdx, trgIdx, shared_vcb);
  } else {
    // In this case, the loadpath does not load shortlist.
    shortlistGenerator_ = nullptr;
  }

  lifecycle::liveModels().fetch_add(1, std::memory_order_relaxed);
  if (lifecycle::enabled()) {
    lifecycle::event("model_ctor id=%zu replicas=%zu model_bytes=%zu shortlist_bytes=%zu", modelId_, replicas,
                     modelMemoryBytes(), memory_.shortlist.size());
  }
}

// PATCH B: see translation_model.h. Bumping here makes every worker thread drop
// its ruy prepacked-weight cache before any replacement model can be handed the
// freed weight addresses.
TranslationModel::~TranslationModel() {
  if (lifecycle::enabled()) {
    size_t loaded = 0;
    for (const auto &backend : backend_) loaded += backend.initialized ? 1 : 0;
    // model_bytes>0 here means some replica was never used, so loadBackend()
    // never got to drop the file image (D0, lazy-replica residue).
    lifecycle::event("model_dtor_enter id=%zu replicas=%zu backends_loaded=%zu model_bytes=%zu", modelId_,
                     backend_.size(), loaded, modelMemoryBytes());
  }
  marian::cpu::integer::bumpPrepackGeneration();
  // Release graphs + scorers here rather than leaving it to member destruction,
  // so the counters logged below already reflect the freed ExpressionGraphs.
  // Safe: scorers hold their own shared_ptr to shortlistGenerator_, and the
  // graph stopped referencing memory_.models once loadBackend ran forward().
  // memory_ is deliberately NOT cleared -- BinaryShortlistGenerator keeps a raw
  // pointer into memory_.shortlist, and member destruction order already frees
  // the generator first.
  backend_.clear();
  lifecycle::liveModels().fetch_sub(1, std::memory_order_relaxed);
  if (lifecycle::enabled()) {
    lifecycle::event("model_dtor_done id=%zu live_models=%ld live_graphs=%ld", modelId_,
                     lifecycle::liveModels().load(std::memory_order_relaxed),
                     lifecycle::liveGraphs().load(std::memory_order_relaxed));
  }
}

size_t TranslationModel::modelMemoryBytes() const {
  // The file image lives behind modelMemory_ (patch 0016), not memory_.models,
  // and stays pinned until every replica has loaded -- so a non-zero value in
  // the destructor means some replica was never used.
  size_t bytes = 0;
  if (auto memory = std::atomic_load(&modelMemory_)) {
    for (const auto &model : *memory) bytes += model.size();
  }
  return bytes;
}

void TranslationModel::loadBackend(size_t idx) {
  auto &graph = backend_[idx].graph;
  auto &scorerEnsemble = backend_[idx].scorerEnsemble;
  if (lifecycle::enabled()) {
    lifecycle::event("load_backend_begin id=%zu replica=%zu model_bytes=%zu", modelId_, idx, modelMemoryBytes());
  }

  marian::DeviceId device_(idx, DeviceType::cpu);
  graph = New<ExpressionGraph>(/*inference=*/true);  // set the graph to be inference only
  auto prec = options_->get<std::vector<std::string>>("precision", {"float32"});
  graph->setDefaultElementType(typeFromString(prec[0]));
  graph->setDevice(device_);
  graph->getBackend()->configureDevice(options_);
  graph->reserveWorkspaceMB(5);

  // Pin the model bytes for the duration of this load; concurrent loads each
  // hold their own reference, so the release below can never free bytes a
  // sibling worker is still reading.
  std::shared_ptr<std::vector<AlignedMemory>> modelMemory = std::atomic_load(&modelMemory_);

  // if the model memory is populated, then all models were of binary format
  if (modelMemory && modelMemory->size() >= 1) {
    const std::vector<const void *> container = std::invoke([&]() {
      std::vector<const void *> model_ptrs(modelMemory->size());
      for (size_t i = 0; i < modelMemory->size(); ++i) {
        const AlignedMemory &model = (*modelMemory)[i];

        ABORT_IF(model.size() == 0 || model.begin() == nullptr, "The provided memory is empty. Cannot load the model.");
        ABORT_IF(
            (uintptr_t)model.begin() % 256 != 0,
            "The provided memory is not aligned to 256 bytes and will crash when vector instructions are used on it.");
        if (options_->get<bool>("check-bytearray", false)) {
          ABORT_IF(!validateBinaryModel(model, model.size()),
                   "The binary file is invalid. Incomplete or corrupted download?");
        }

        model_ptrs[i] = model.begin();
        LOG(debug, "Loaded model {} of {} from memory", (i + 1), model_ptrs.size());
      }
      return model_ptrs;
    });

    scorerEnsemble = createScorers(options_, container);
  } else {
    // load npz format models, or a mixture of binary/npz formats
    scorerEnsemble = createScorers(options_);
    LOG(debug, "Loaded {} model(s) from file", scorerEnsemble.size());
  }

  for (auto& scorer : scorerEnsemble) {
    scorer->init(graph);
    if (shortlistGenerator_) {
      scorer->setShortlistGenerator(shortlistGenerator_);
    }
  }

  graph->forward();

  // At this point the ExpressionGraph has consumed the `std::vector<marian::io::Item>`
  // and converted them to `Tensor`s. This happens the first time that
  // `ExpressionGraph::forward` is called. It is relatively safe to clear the `Item`s owned
  // by the scorer since they will not practically be used again. This will free up memory
  // in the memory-constrained environment of the browser.
  for (auto scorer : scorerEnsemble) {
    scorer->clearItems();
  }

  // Similarly to the scorers, there is an extra copy of the model bytes. Drop
  // the shared reference only once every replica has loaded — an earlier
  // clear would force later replicas back to disk (or, unsynchronized, hand
  // them freed memory). Replicas that loaded concurrently keep their own pin
  // until they leave this function.
  if (backendsLoaded_.fetch_add(1) + 1 == backend_.size()) {
    const size_t releasedBytes = modelMemoryBytes();
    std::atomic_store(&modelMemory_, std::shared_ptr<std::vector<AlignedMemory>>());
    if (lifecycle::enabled()) {
      // The local pin above still holds the bytes until this function returns.
      lifecycle::event("model_memory_released id=%zu replica=%zu bytes=%zu", modelId_, idx, releasedBytes);
    }
  }
  if (lifecycle::enabled()) {
    lifecycle::event("load_backend_end id=%zu replica=%zu live_graphs=%ld", modelId_, idx,
                     lifecycle::liveGraphs().load(std::memory_order_relaxed));
  }
}

// Make request process is shared between Async and Blocking workflow of translating.
Ptr<Request> TranslationModel::makeRequest(size_t requestId, std::string &&source, CallbackType callback,
                                           const ResponseOptions &responseOptions,
                                           std::optional<TranslationCache> &cache) {
  Segments segments;
  AnnotatedText annotatedSource;

  textProcessor_.process(std::move(source), annotatedSource, segments);
  ResponseBuilder responseBuilder(responseOptions, std::move(annotatedSource), vocabs_, callback, *qualityEstimator_);
#if defined(WASM)
  responseBuilder.registerTargetLanguage(targetLanguage_);
#endif // defined(WASM)

  Ptr<Request> request =
      New<Request>(requestId, /*model=*/*this, std::move(segments), std::move(responseBuilder), cache);
  return request;
}

Ptr<Request> TranslationModel::makePivotRequest(size_t requestId, AnnotatedText &&previousTarget, CallbackType callback,
                                                const ResponseOptions &responseOptions,
                                                std::optional<TranslationCache> &cache) {
  Segments segments;

  textProcessor_.processFromAnnotation(previousTarget, segments);
  ResponseBuilder responseBuilder(responseOptions, std::move(previousTarget), vocabs_, callback, *qualityEstimator_);
#if defined(WASM)
  responseBuilder.registerTargetLanguage(targetLanguage_);
#endif // defined(WASM)

  Ptr<Request> request = New<Request>(requestId, *this, std::move(segments), std::move(responseBuilder), cache);
  return request;
}

Ptr<marian::data::CorpusBatch> TranslationModel::convertToMarianBatch(Batch &batch) {
  std::vector<data::SentenceTuple> batchVector;
  auto &sentences = batch.sentences();

  size_t batchSequenceNumber{0};
  for (auto &sentence : sentences) {
    data::SentenceTuple sentence_tuple(batchSequenceNumber);
    Segment segment = sentence.getUnderlyingSegment();
    sentence_tuple.push_back(segment);
    batchVector.push_back(sentence_tuple);

    ++batchSequenceNumber;
  }

  // Usually one would expect inputs to be [B x T], where B = batch-size and T = max seq-len among the sentences in the
  // batch. However, marian's library supports multi-source and ensembling through different source-vocabulary but same
  // target vocabulary. This means the inputs are 3 dimensional when converted into marian's library formatted batches.
  //
  // Consequently B x T projects to N x B x T, where N = ensemble size. This adaptation does not fully force the idea of
  // N = 1 (the code remains general, but N iterates only from 0-1 in the nested loop).

  size_t batchSize = batchVector.size();

  std::vector<size_t> sentenceIds;
  std::vector<int> maxDims;

  for (auto &example : batchVector) {
    if (maxDims.size() < example.size()) {
      maxDims.resize(example.size(), 0);
    }
    for (size_t i = 0; i < example.size(); ++i) {
      if (example[i].size() > static_cast<size_t>(maxDims[i])) {
        maxDims[i] = static_cast<int>(example[i].size());
      }
    }
    sentenceIds.push_back(example.getId());
  }

  using SubBatch = marian::data::SubBatch;
  std::vector<Ptr<SubBatch>> subBatches;
  for (size_t j = 0; j < maxDims.size(); ++j) {
    subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_.sources().at(j)));
  }

  std::vector<size_t> words(maxDims.size(), 0);
  for (size_t i = 0; i < batchSize; ++i) {
    for (size_t j = 0; j < maxDims.size(); ++j) {
      for (size_t k = 0; k < batchVector[i][j].size(); ++k) {
        subBatches[j]->data()[k * batchSize + i] = batchVector[i][j][k];
        subBatches[j]->mask()[k * batchSize + i] = 1.f;
        words[j]++;
      }
    }
  }

  for (size_t j = 0; j < maxDims.size(); ++j) {
    subBatches[j]->setWords(words[j]);
  }

  using CorpusBatch = marian::data::CorpusBatch;
  Ptr<CorpusBatch> corpusBatch = New<CorpusBatch>(subBatches);
  corpusBatch->setSentenceIds(sentenceIds);
  return corpusBatch;
}

void TranslationModel::translateBatch(size_t deviceId, Batch &batch) {
  auto &backend = backend_[deviceId];

  if (!backend.initialized) {
    loadBackend(deviceId);
    backend.initialized = true;
  }

  BeamSearch search(options_, backend.scorerEnsemble, vocabs_.target());
  Histories histories = search.search(backend.graph, convertToMarianBatch(batch));
  batch.completeBatch(histories);
}

}  // namespace bergamot
}  // namespace marian
