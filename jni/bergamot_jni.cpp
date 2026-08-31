// JNI glue for io.github.yinvoker.bergamot.NativeBridge.
// Thin by design: batch in, batch out, blocking from the caller's view.
// AsyncService workers give intra-batch parallelism; the Kotlin layer owns
// threading policy, lifecycle and cancellation.
#include <jni.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

namespace {

using marian::bergamot::AsyncService;
using marian::bergamot::Response;
using marian::bergamot::ResponseOptions;
using marian::bergamot::TranslationModel;

using ModelHandle = std::shared_ptr<TranslationModel>;

void throwJava(JNIEnv *env, const std::string &message) {
  jclass cls = env->FindClass("java/lang/RuntimeException");
  if (cls != nullptr) env->ThrowNew(cls, message.c_str());
}

std::string toStdString(JNIEnv *env, jstring value) {
  const char *chars = env->GetStringUTFChars(value, nullptr);
  std::string result(chars == nullptr ? "" : chars);
  if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
  return result;
}

std::vector<std::string> toStdStrings(JNIEnv *env, jobjectArray array) {
  jsize n = env->GetArrayLength(array);
  std::vector<std::string> result;
  result.reserve(n);
  for (jsize i = 0; i < n; ++i) {
    auto element = static_cast<jstring>(env->GetObjectArrayElement(array, i));
    result.push_back(toStdString(env, element));
    env->DeleteLocalRef(element);
  }
  return result;
}

jobjectArray toJavaStrings(JNIEnv *env, const std::vector<Response> &responses) {
  jclass stringClass = env->FindClass("java/lang/String");
  jobjectArray result = env->NewObjectArray(static_cast<jsize>(responses.size()), stringClass, nullptr);
  for (jsize i = 0; i < static_cast<jsize>(responses.size()); ++i) {
    jstring text = env->NewStringUTF(responses[i].target.text.c_str());
    env->SetObjectArrayElement(result, i, text);
    env->DeleteLocalRef(text);
  }
  return result;
}

/// Fans texts out to the AsyncService and blocks until every callback fired.
/// submit(i, text, callback) issues request i.
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

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_createService(JNIEnv *env, jobject, jint workers) {
  try {
    AsyncService::Config config;
    config.numWorkers = workers < 1 ? 1 : static_cast<size_t>(workers);
    return reinterpret_cast<jlong>(new AsyncService(config));
  } catch (const std::exception &e) {
    throwJava(env, std::string("createService failed: ") + e.what());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_destroyService(JNIEnv *, jobject, jlong service) {
  delete reinterpret_cast<AsyncService *>(service);
}

JNIEXPORT jlong JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_loadModel(JNIEnv *env, jobject, jlong service, jstring configYaml) {
  try {
    auto *svc = reinterpret_cast<AsyncService *>(service);
    auto options = marian::bergamot::parseOptionsFromString(toStdString(env, configYaml), /*validate=*/false);
    // createCompatibleModel sizes per-worker backends; a bare TranslationModel
    // has one and crashes (SIGBUS) as soon as worker id > 0 touches it.
    return reinterpret_cast<jlong>(new ModelHandle(svc->createCompatibleModel(options)));
  } catch (const std::exception &e) {
    throwJava(env, std::string("loadModel failed: ") + e.what());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_destroyModel(JNIEnv *, jobject, jlong model) {
  delete reinterpret_cast<ModelHandle *>(model);
}

JNIEXPORT jobjectArray JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_translate(JNIEnv *env, jobject, jlong service, jlong model,
                                                        jobjectArray texts, jboolean html) {
  try {
    auto *svc = reinterpret_cast<AsyncService *>(service);
    auto &handle = *reinterpret_cast<ModelHandle *>(model);
    ResponseOptions responseOptions;
    responseOptions.HTML = html;
    auto responses = collectAll(toStdStrings(env, texts), [&](size_t, std::string &&text, auto callback) {
      svc->translate(handle, std::move(text), std::move(callback), responseOptions);
    });
    return toJavaStrings(env, responses);
  } catch (const std::exception &e) {
    throwJava(env, std::string("translate failed: ") + e.what());
    return nullptr;
  }
}

JNIEXPORT jobjectArray JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_translatePivot(JNIEnv *env, jobject, jlong service, jlong first,
                                                             jlong second, jobjectArray texts, jboolean html) {
  try {
    auto *svc = reinterpret_cast<AsyncService *>(service);
    auto &firstHandle = *reinterpret_cast<ModelHandle *>(first);
    auto &secondHandle = *reinterpret_cast<ModelHandle *>(second);
    ResponseOptions responseOptions;
    responseOptions.HTML = html;
    auto responses = collectAll(toStdStrings(env, texts), [&](size_t, std::string &&text, auto callback) {
      svc->pivot(firstHandle, secondHandle, std::move(text), std::move(callback), responseOptions);
    });
    return toJavaStrings(env, responses);
  } catch (const std::exception &e) {
    throwJava(env, std::string("translatePivot failed: ") + e.what());
    return nullptr;
  }
}

}  // extern "C"
