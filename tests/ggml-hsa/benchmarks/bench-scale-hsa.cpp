// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_SCALE (out = a * s) across backends. SCALE is element-wise
// and memory-bound, so the reported metric is memory bandwidth. Shapes mirror the
// GPT-2 attention-score scaling by 1/sqrt(d_head): [n_kv, n_q, n_head] with
// n_head = 12 and n_kv == n_q == n_tokens.

#include <benchmark/benchmark.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

enum class BackendType {
    CPU,
    GPU,
    HSA,
};

std::vector<float> make_data(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i % 101) * 0.1f - 5.0f;
    }
    return v;
}

ggml_backend_t make_backend(BackendType type, benchmark::State & state) {
    switch (type) {
        case BackendType::CPU:
            return ggml_backend_cpu_init();
        case BackendType::GPU:
#ifdef GGML_USE_CUDA
            return ggml_backend_cuda_init(0);
#else
            state.SkipWithError("CUDA backend not available.");
            return nullptr;
#endif
        case BackendType::HSA:
#ifdef GGML_USE_HSA
            return ggml_backend_hsa_init(0);
#else
            state.SkipWithError("HSA backend not available.");
            return nullptr;
#endif
    }
    state.SkipWithError("Invalid backend type.");
    return nullptr;
}

} // namespace

// Benchmarks out = a * s, with a and out both [ne0, ne1, ne2, ne3] in ggml's
// ne[0]-fastest layout (state.range(0..3)). s is the GPT-2 1/sqrt(d_head) factor.
template <BackendType Backend>
void bench_scale(benchmark::State & state) {
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        if (!state.error_occurred()) {
            state.SkipWithError("Backend creation failed.");
        }
        return;
    }

    const std::int64_t ne0 = state.range(0);
    const std::int64_t ne1 = state.range(1);
    const std::int64_t ne2 = state.range(2);
    const std::int64_t ne3 = state.range(3);

    // GPT-2 scales attention scores by 1/sqrt(d_head); d_head = n_embd / n_head = 64.
    const float scale = 1.0f / std::sqrt(64.0f);

    const std::size_t tensor_count = 2;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * tensor_result = ggml_scale(ctx, tensor_a, scale);
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        state.SkipWithError("Operation not supported.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        state.SkipWithError("Tensor buffer allocation failed.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, tensor_count, /*grads*/ false);
    ggml_build_forward_expand(gf, tensor_result);

    const std::vector<float> A = make_data(ggml_nelements(tensor_a));
    ggml_backend_tensor_set(tensor_a, A.data(), 0, ggml_nbytes(tensor_a));

    // warm up (also triggers any one-time JIT compile so it's not measured)
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        state.SkipWithError("Warm-up graph compute error.");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    for (auto _ : state) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            state.SkipWithError("Graph compute error.");
            break;
        }
    }

    const double bytes_per_iter =
        static_cast<double>(ggml_nbytes(tensor_a) + ggml_nbytes(tensor_result));
    state.counters["bytes"] =
        benchmark::Counter(bytes_per_iter, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["elements"] = benchmark::Counter(
        static_cast<double>(ggml_nelements(tensor_a)), benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// GPT-2 attention-score shapes (ne0, ne1, ne2, ne3): [n_kv, n_q, n_head, 1],
// n_head = 12 and n_kv == n_q == n_tokens.
#define SCALE_SHAPES(BENCH)                                                                         \
    BENCH->Args({64, 64, 12, 1})                                                                    \
        ->Args({256, 256, 12, 1})                                                                   \
        ->Args({1024, 1024, 12, 1})                                                                 \
        ->UseRealTime()

SCALE_SHAPES(BENCHMARK(bench_scale<BackendType::CPU>));
SCALE_SHAPES(BENCHMARK(bench_scale<BackendType::HSA>));
SCALE_SHAPES(BENCHMARK(bench_scale<BackendType::GPU>));
