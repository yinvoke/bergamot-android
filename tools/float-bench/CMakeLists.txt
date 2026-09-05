# Shape-level ruy-vs-small-GEMM microbench for the attention float matmuls.
# Not a test: it prints timings and asserts bit-identical results.
# It compares against ruy's float GEMM, so it only exists in builds that use
# it (Android, or a host configured with -DUSE_RUY_SGEMM=ON); the default host
# build links Apple Accelerate and has no GemmRuy to compare with.
if(NOT USE_RUY_SGEMM)
  message(STATUS "float_bench: skipped (needs the ruy float GEMM, -DUSE_RUY_SGEMM=ON)")
  return()
endif()
add_executable(float_bench float-batched-bench.cpp)
target_include_directories(float_bench PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy
  ${CMAKE_SOURCE_DIR}/engine/marian-fork/src/3rd_party/ruy/third_party/cpuinfo/include)
# prod_blas.h picks its backend from this; the bench must see the same one
# the engine was built with.
target_compile_definitions(float_bench PRIVATE USE_RUY_SGEMM=1)
target_link_libraries(float_bench bergamot-translator-source)
