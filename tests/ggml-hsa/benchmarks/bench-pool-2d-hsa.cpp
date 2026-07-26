// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_POOL_2D (2x2 max pooling, stride 2) across backends.
// Pooling is memory-bound (read the window, write one output), so the reported
// metric is memory bandwidth. Shapes mirror the MNIST-CNN pooling layers, which
// halve each spatial dimension after conv1 ([28,28,8]) and conv2 ([14,14,16]).

#include "bench-hsa-common.hpp"

#include <cstdint>

// Benchmarks out = pool_2d(input[W,H,C,N], MAX, k=2, s=2, p=0). state.range = {W,H,C,N}.
template <BackendType Backend>
void bench_pool_2d(benchmark::State & state) {
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        if (!state.error_occurred()) {
            state.SkipWithError("Backend creation failed.");
        }
        return;
    }

    const std::int64_t W = state.range(0);
    const std::int64_t H = state.range(1);
    const std::int64_t C = state.range(2);
    const std::int64_t N = state.range(3);

    const std::size_t ctx_size = 4 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
    ggml_tensor * tensor_result =
        ggml_pool_2d(ctx, tensor_input, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        state.SkipWithError("Operation not supported.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, tensor_result);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        state.SkipWithError("Graph allocation failed.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    const std::vector<float> A = make_data(ggml_nelements(tensor_input));
    ggml_backend_tensor_set(tensor_input, A.data(), 0, ggml_nbytes(tensor_input));

    // warm up (also triggers any one-time JIT compile so it's not measured)
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        state.SkipWithError("Warm-up graph compute error.");
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
        static_cast<double>(ggml_nbytes(tensor_input) + ggml_nbytes(tensor_result));
    state.counters["bytes"] =
        benchmark::Counter(bytes_per_iter, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["elements"] = benchmark::Counter(
        static_cast<double>(ggml_nelements(tensor_input)),
        benchmark::Counter::kIsIterationInvariantRate);

    ggml_free(ctx);
    ggml_backend_free(backend);
}

// MNIST-CNN pooling inputs {W, H, C, N} (2x2 max, stride 2 -> halves W and H):
//   after conv1: [28,28,8]
//   after conv2: [14,14,16]
#define POOL_2D_SHAPES(BENCH)                                                                       \
    BENCH->Args({28, 28, 8, 500})                                                                   \
        ->Args({28, 28, 8, 1000})                                                                   \
        ->Args({14, 14, 16, 500})                                                                   \
        ->Args({14, 14, 16, 1000})                                                                  \
        ->UseRealTime()

POOL_2D_SHAPES(BENCHMARK(bench_pool_2d<BackendType::CPU>));
POOL_2D_SHAPES(BENCHMARK(bench_pool_2d<BackendType::HSA>));
POOL_2D_SHAPES(BENCHMARK(bench_pool_2d<BackendType::GPU>));
