// JNI glue for io.github.yinvoker.bergamot.NativeBridge.
// Thin by design: batch in, batch out, blocking. Threading, lifecycle and
// cancellation live in the Kotlin layer.
#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"

namespace {

using marian::bergamot::BlockingService;
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

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_createService(JNIEnv *env, jobject, jint cacheSize) {
  try {
    BlockingService::Config config;
    config.cacheSize = static_cast<size_t>(cacheSize);
    return reinterpret_cast<jlong>(new BlockingService(config));
  } catch (const std::exception &e) {
    throwJava(env, std::string("createService failed: ") + e.what());
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_destroyService(JNIEnv *, jobject, jlong service) {
  delete reinterpret_cast<BlockingService *>(service);
}

JNIEXPORT jlong JNICALL
Java_io_github_yinvoker_bergamot_NativeBridge_loadModel(JNIEnv *env, jobject, jstring configYaml) {
  try {
    auto options = marian::bergamot::parseOptionsFromString(toStdString(env, configYaml), /*validate=*/false);
    return reinterpret_cast<jlong>(new ModelHandle(std::make_shared<TranslationModel>(options)));
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
    auto *svc = reinterpret_cast<BlockingService *>(service);
    auto *handle = reinterpret_cast<ModelHandle *>(model);
    std::vector<std::string> sources = toStdStrings(env, texts);
    ResponseOptions responseOptions;
    responseOptions.HTML = html;
    std::vector<ResponseOptions> perText(sources.size(), responseOptions);
    std::vector<Response> responses = svc->translateMultiple(*handle, std::move(sources), perText);
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
    auto *svc = reinterpret_cast<BlockingService *>(service);
    auto *firstHandle = reinterpret_cast<ModelHandle *>(first);
    auto *secondHandle = reinterpret_cast<ModelHandle *>(second);
    std::vector<std::string> sources = toStdStrings(env, texts);
    ResponseOptions responseOptions;
    responseOptions.HTML = html;
    std::vector<ResponseOptions> perText(sources.size(), responseOptions);
    std::vector<Response> responses =
        svc->pivotMultiple(*firstHandle, *secondHandle, std::move(sources), perText);
    return toJavaStrings(env, responses);
  } catch (const std::exception &e) {
    throwJava(env, std::string("translatePivot failed: ") + e.what());
    return nullptr;
  }
}

}  // extern "C"
