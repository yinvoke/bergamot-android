add_executable(smoke smoke.cpp)
target_include_directories(smoke PRIVATE ${CMAKE_SOURCE_DIR}/engine/src)
target_link_libraries(smoke bergamot-translator-source)
