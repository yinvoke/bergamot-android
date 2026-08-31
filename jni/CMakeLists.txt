add_library(bergamot SHARED bergamot_jni.cpp)
target_include_directories(bergamot PRIVATE ${CMAKE_SOURCE_DIR}/engine/src)
target_link_libraries(bergamot bergamot-translator-source log)
