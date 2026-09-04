# Native build entry. Consumed two ways:
#   - host (ARM only): engine + tools/smoke, driven by plain cmake
#   - Android: engine + jni, driven by AGP externalNativeBuild (see bergamot/)
cmake_minimum_required(VERSION 3.22)
project(bergamot_android CXX C)

# Old cmake_minimum_required() calls throughout the vendored tree.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Consumers outside engine/ (tools/smoke, jni/) include marian/bergamot/ssplit
# headers, whose contents are shaped by these macros. The vendored tree sets
# them directory-scoped only, so sibling directories must match them here or
# header code paths diverge from the compiled engine (raw x86 intrinsics on
# ARM, wrong string_view, etc.). Mirrors the flag set the engine itself is
# built with.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv8)")
  message(FATAL_ERROR "bergamot-android is ARM-only (see engine/UPSTREAM.md)")
endif()
add_compile_definitions(ARM FMA SSE COMPILE_CPU=1 USE_PTHREADS USE_SENTENCEPIECE _USE_INTERNAL_STRING_VIEW)
add_compile_options(-flax-vector-conversions -fno-strict-aliasing)

if(ANDROID)
  # bionic ships pthreads inside libc; the ancient FindThreads invocations in
  # vendored subprojects (ruy/cpuinfo) fail to detect that. Pre-seed the result.
  set(CMAKE_THREAD_LIBS_INIT "")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG OFF)
  set(Threads_FOUND TRUE)
endif()

add_subdirectory(engine)

option(BUILD_SMOKE "Also build the smoke CLI on Android (adb-shell benchmarking)" OFF)
option(BUILD_I8MM_BENCH "SDOT-vs-SMMLA microbench (i8mm backend go/no-go data)" OFF)
if(BUILD_I8MM_BENCH)
  add_subdirectory(tools/i8mm-bench)
endif()
if(ANDROID)
  add_subdirectory(jni)
  if(BUILD_SMOKE)
    add_subdirectory(tools/smoke)
  endif()
else()
  add_subdirectory(tools/smoke)
endif()
