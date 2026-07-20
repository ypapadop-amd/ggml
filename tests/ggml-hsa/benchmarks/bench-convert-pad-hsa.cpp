// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for the internal HSA_CONVERT_PAD kernel (the MUL_MAT convert+pad pre-amble):
// f32 [d0, d1] -> bf16 [d0pad, d1pad]. Times the on-device dispatch (kernel build is cached; each
// call submits + waits, matching the per-op production path). Parameterized by d0/d1/d0pad/d1pad.

#include <benchmark/benchmark.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

#include <cstdint>
#include <vector>

// Benchmarks the convert+pad pre-amble via the internal test dispatch hook, with the source
// f32 [d0, d1] and destination bf16 [d0pad, d1pad] taken from state.range(0..3).
void bench_convert_pad(benchmark::State & state) {
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
    ggml_tensor *  src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
    ggml_tensor *  dst = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d0pad, d1pad);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        state.SkipWithError("Tensor buffer allocation failed.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    std::vector<float> src_host(d0 * d1);
    for (std::int64_t i = 0; i < d0 * d1; ++i) {
        src_host[i] = static_cast<float>(i % 97) * 0.5f - 13.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));
    std::vector<uint16_t> dst_zero(d0pad * d1pad, 0);
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, ggml_nbytes(dst));

    // warm up (also triggers the one-time JIT compile so it is not measured)
    if (ggml_hsa_test_dispatch_transform(backend, "HSA_CONVERT_PAD", src, dst) !=
        GGML_STATUS_SUCCESS) {
        state.SkipWithError("Warm-up dispatch error.");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    for (auto _ : state) {
        if (ggml_hsa_test_dispatch_transform(backend, "HSA_CONVERT_PAD", src, dst) !=
            GGML_STATUS_SUCCESS) {
            state.SkipWithError("Dispatch error.");
            break;
        }
    }

    // Elements converted per iteration (the valid, non-padded sub-block).
    state.counters["elem/s"] = benchmark::Counter(
        static_cast<double>(d0 * d1), benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
#endif
}

// {d0, d1, d0pad, d1pad}: real MNIST MUL_MAT operand shapes plus larger square shapes.
BENCHMARK(bench_convert_pad)
    ->Args({784, 500, 800, 512})   // mnist A/B c1
    ->Args({500, 500, 512, 512})   // mnist B c2
    ->Args({500, 10, 512, 128})    // mnist A c2
    ->Args({784, 10, 800, 128})    // mnist A c3
    ->Args({1024, 1024, 1024, 1024})
    ->Args({2048, 2048, 2048, 2048})
    ->Args({4096, 4096, 4096, 4096})
    ->UseRealTime();
