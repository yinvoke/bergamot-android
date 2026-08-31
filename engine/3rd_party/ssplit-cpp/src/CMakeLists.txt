# bergamot-android: library-only. Upstream's CLI (src/command/) is not vendored.
include_directories(ssplit)
add_library(ssplit STATIC ssplit/ssplit.cpp ssplit/regex.cpp)
target_link_libraries(ssplit ${EXT_LIBS})

set_target_properties(ssplit PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
