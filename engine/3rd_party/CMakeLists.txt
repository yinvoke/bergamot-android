# bergamot-android: trimmed from upstream (WASM branch removed).
set(COMPILE_TESTS OFF)

add_subdirectory(ssplit-cpp EXCLUDE_FROM_ALL)
get_property(INCLUDE_DIRECTORIES DIRECTORY ssplit-cpp/src PROPERTY INCLUDE_DIRECTORIES)
target_include_directories(ssplit PUBLIC ${INCLUDE_DIRECTORIES})
