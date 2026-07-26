// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_MUL across backends, focused on the broadcast case GPT-2
// uses for the LayerNorm affine scale: out = a * w, with w a single [ne0] row
// (n_embd) broadcast over the token dimension. MUL is memory-bound, so the
// reported metric is memory bandwidth.

#include "bench-hsa-common.hpp"

#include <cstdint>

// Benchmarks out = a * w, with a [ne0, ne1, ne2, ne3] (state.range(0..3)) and w a
// single [ne0] row broadcast over the other dims (LayerNorm affine scale).
template <BackendType Backend>
void bench_mul(benchmark::State & state) {
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

    const std::size_t tensor_count = 3;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * tensor_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);  // broadcast affine
    ggml_tensor * tensor_result = ggml_mul(ctx, tensor_a, tensor_w);
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
    const std::vector<float> W = make_data(ggml_nelements(tensor_w));
    ggml_backend_tensor_set(tensor_a, A.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_w, W.data(), 0, ggml_nbytes(tensor_w));

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

    const double bytes_per_iter = static_cast<double>(
        ggml_nbytes(tensor_a) + ggml_nbytes(tensor_w) + ggml_nbytes(tensor_result));
    state.counters["bytes"] =
        benchmark::Counter(bytes_per_iter, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["elements"] = benchmark::Counter(
        static_cast<double>(ggml_nelements(tensor_result)),
        benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// GPT-2 LayerNorm affine shapes (ne0, ne1, ne2, ne3): [n_embd, n_tokens, 1, 1] * [n_embd].
#define MUL_SHAPES(BENCH)                                                                           \
    BENCH->Args({768, 64, 1, 1})                                                                    \
        ->Args({768, 256, 1, 1})                                                                    \
        ->Args({768, 1024, 1, 1})                                                                   \
        ->UseRealTime()

MUL_SHAPES(BENCHMARK(bench_mul<BackendType::CPU>));
MUL_SHAPES(BENCHMARK(bench_mul<BackendType::HSA>));
MUL_SHAPES(BENCHMARK(bench_mul<BackendType::GPU>));
