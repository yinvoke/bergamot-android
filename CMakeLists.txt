# Native build entry. Consumed two ways:
#   - host (ARM only): engine + tools/smoke, driven by plain cmake
#   - Android: engine + jni, driven by AGP externalNativeBuild (see bergamot/)
cmake_minimum_required(VERSION 3.22)
project(bergamot_android CXX C)

# Old cmake_minimum_required() calls throughout the vendored tree.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

add_subdirectory(engine)

if(ANDROID)
  add_subdirectory(jni)
else()
  add_subdirectory(tools/smoke)
endif()
