// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for the internal HSA_DEPAD kernel (the MUL_MAT de-pad post-amble): narrows each row
// from a padded f32 [d0pad, d1pad] temporary to a dense [d0, d1] destination, either plain
// (f32 -> f32) or fusing the per-layer cast (f32 -> bf16). Times the on-device dispatch (kernel
// build is cached; each call submits + waits, matching the per-op production path).

#include <benchmark/benchmark.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

#include <cstdint>
#include <vector>

// Benchmarks the de-pad post-amble via the internal test dispatch hook, with the padded f32 source
// [d0pad, d1pad] and dense destination [d0, d1] taken from state.range(0..3). DstType selects the
// plain (f32) or fused-cast (bf16) destination.
template <ggml_type DstType>
void bench_depad(benchmark::State & state) {
#ifndef GGML_USE_HSA
    state.SkipWithError("HSA backend not available.");
    return;
#else
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        state.SkipWithError("Backend creation failed.");
        return;
    }

    const std::int64_t d0    = state.range(0);
    const std::int64_t d1    = state.range(1);
    const std::int64_t d0pad = state.range(2);
    const std::int64_t d1pad = state.range(3);

    ggml_init_params params = {/*.mem_size   =*/ggml_tensor_overhead() * 2 + 1024,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor *  src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0pad, d1pad);
    ggml_tensor *  dst = ggml_new_tensor_2d(ctx, DstType, d0, d1);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        state.SkipWithError("Tensor buffer allocation failed.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    std::vector<float> src_host(d0pad * d1pad);
    for (std::int64_t i = 0; i < d0pad * d1pad; ++i) {
        src_host[i] = static_cast<float>(i) * 0.25f + 1.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    // warm up (also triggers the one-time JIT compile so it is not measured)
    if (ggml_hsa_test_dispatch_transform(backend, "HSA_DEPAD", src, dst) != GGML_STATUS_SUCCESS) {
        state.SkipWithError("Warm-up dispatch error.");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    for (auto _ : state) {
        if (ggml_hsa_test_dispatch_transform(backend, "HSA_DEPAD", src, dst) !=
            GGML_STATUS_SUCCESS) {
            state.SkipWithError("Dispatch error.");
            break;
        }
    }

    // Elements copied per iteration (the dense, non-padded sub-block).
    state.counters["elem/s"] = benchmark::Counter(
        static_cast<double>(d0 * d1), benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
#endif
}

// {d0, d1, d0pad, d1pad}: real MNIST MUL_MAT output shapes plus larger square shapes.
#define DEPAD_ARGS(bench)                                                                          \
    BENCHMARK(bench)                                                                               \
        ->Args({500, 500, 512, 512})   /* mnist C c1 */                                            \
        ->Args({10, 500, 128, 512})    /* mnist C c2 */                                            \
        ->Args({500, 4, 512, 128})     /* few wide rows */                                         \
        ->Args({1024, 1024, 1024, 1024})                                                           \
        ->Args({2048, 2048, 2048, 2048})                                                           \
        ->Args({4096, 4096, 4096, 4096})                                                           \
        ->UseRealTime()

DEPAD_ARGS(bench_depad<GGML_TYPE_F32>);
DEPAD_ARGS(bench_depad<GGML_TYPE_BF16>);
