# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

# The parent tests/ directory does add_definitions(-UNDEBUG) to keep asserts on
# in the test suite. For a performance benchmark we want a genuine optimized
# build (NDEBUG defined, asserts and Google Benchmark's internal CHECKs compiled
# out). That -UNDEBUG is inherited from the parent scope's COMPILE_OPTIONS and
# is emitted at the end of the compile flags, so it cannot be removed from a
# child directory. Re-assert -DNDEBUG here via add_compile_options so it is
# appended after the inherited -UNDEBUG: the compiler applies -D/-U left to
# right, so NDEBUG ends up defined. This runs before add_executable and
# FetchContent below, so both the benchmark and the Google Benchmark library
# inherit it.
add_compile_options(-DNDEBUG)

include(FetchContent)

FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebenchmark)

#
# bench-mul-mat-hsa

set(BENCH_TARGET bench-mul-mat-hsa)
add_executable(${BENCH_TARGET} ${BENCH_TARGET}.cpp)
target_link_libraries(${BENCH_TARGET} PRIVATE benchmark::benchmark_main ggml)
target_compile_features(${BENCH_TARGET} PRIVATE cxx_std_17)
