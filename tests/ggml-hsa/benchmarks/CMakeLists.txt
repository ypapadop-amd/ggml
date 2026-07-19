# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

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
