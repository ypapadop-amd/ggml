// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_UNARY_OP_GELU across backends. GELU is element-wise and
// memory-bound, so the reported metric is memory bandwidth. Shapes mirror the
// GPT-2 MLP hidden activation: [4*n_embd, n_tokens] with 4*n_embd = 3072.

#include "bench-hsa-common.hpp"

#include <cstdint>

// Benchmarks out = gelu(a), with a and out both [ne0, ne1, ne2, ne3] in ggml's
// ne[0]-fastest layout, using state.range(0..3) for the four dims.
template <BackendType Backend>
void bench_gelu(benchmark::State & state) {
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        return;
    }

    const std::int64_t ne0 = state.range(0);
    const std::int64_t ne1 = state.range(1);
    const std::int64_t ne2 = state.range(2);
    const std::int64_t ne3 = state.range(3);

    const std::size_t tensor_count = 2;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * tensor_result = ggml_gelu(ctx, tensor_a);
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

    // Negative start so the input spans zero, exercising both sides of GELU.
    const std::vector<float> A = make_data(ggml_nelements(tensor_a), -50.0f);
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

// GPT-2 MLP hidden shapes (ne0, ne1, ne2, ne3): [4*n_embd, n_tokens, 1, 1],
// 4*n_embd = 3072.
#define GELU_SHAPES(BENCH)                                                                          \
    BENCH->Args({3072, 64, 1, 1})                                                                   \
        ->Args({3072, 256, 1, 1})                                                                   \
        ->Args({3072, 1024, 1, 1})                                                                  \
        ->UseRealTime()

GELU_SHAPES(BENCHMARK(bench_gelu<BackendType::CPU>));
GELU_SHAPES(BENCHMARK(bench_gelu<BackendType::HSA>));
GELU_SHAPES(BENCHMARK(bench_gelu<BackendType::GPU>));
