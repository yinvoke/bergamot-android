# SMMLA GEMM test set. Registered with ctest; run it on an i8mm CPU with
# SMMLA_TEST_REQUIRE_I8MM=1 so a silent skip cannot pass as coverage.
add_executable(smmla_test smmla_test.cpp)
target_include_directories(smmla_test PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy/third_party/cpuinfo/include)
target_link_libraries(smmla_test bergamot-translator-source)
add_test(NAME smmla_test COMMAND smmla_test)

# Shape-level ruy-vs-SMMLA microbench; same include/link set, not a test.
add_executable(ab_bench ab-bench.cpp)
target_include_directories(ab_bench PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy/third_party/cpuinfo/include)
target_link_libraries(ab_bench bergamot-translator-source)
