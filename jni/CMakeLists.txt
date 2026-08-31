add_library(bergamot SHARED bergamot_jni.cpp)
target_include_directories(bergamot PRIVATE ${CMAKE_SOURCE_DIR}/engine/src ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy/third_party/cpuinfo/include)
target_link_libraries(bergamot bergamot-translator-source log)
