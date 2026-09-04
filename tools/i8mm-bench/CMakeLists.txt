# Standalone microbench; needs +i8mm codegen, gates itself on HWCAP2 at runtime.
add_executable(i8mm-bench main.cpp)
target_compile_options(i8mm-bench PRIVATE -march=armv8.2-a+dotprod+i8mm)
