// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_GET_ROWS (row gather) across backends. GET_ROWS is a
// memory-bound gather (read one src0 row per index, write it to dst), so the
// reported metric is memory bandwidth over the gathered data. Shapes mirror the
// GPT-2 token-embedding (wte [n_embd, n_vocab]) and position-embedding
// (wpe [n_embd, n_ctx]) lookups: gather n_tokens rows.

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

#include <cstdint>
#include <vector>

namespace {

enum class BackendType {
    CPU,
    GPU,
    HSA,
};

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

// Benchmarks dst = get_rows(table, idx). table is [nc, nrows] f32 (state.range(0/1)),
// idx is [n_idx] i32 (state.range(2)); dst is [nc, n_idx] f32.
template <BackendType Backend>
void bench_get_rows(benchmark::State & state) {
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        if (!state.error_occurred()) {
            state.SkipWithError("Backend creation failed.");
        }
        return;
    }

    const std::int64_t nc    = state.range(0);
    const std::int64_t nrows = state.range(1);
    const std::int64_t n_idx = state.range(2);

    const std::size_t tensor_count = 3;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nrows);
    ggml_tensor * tensor_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_idx);
    ggml_tensor * tensor_result = ggml_get_rows(ctx, tensor_table, tensor_idx);
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

    std::vector<float> table(nc * nrows);
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<float>(i % 101) * 0.25f - 7.0f;
    }
    std::vector<std::int32_t> idx(n_idx);
    for (std::int64_t i = 0; i < n_idx; ++i) {
        idx[i] = static_cast<std::int32_t>((i * 7 + 3) % nrows);
    }
    ggml_backend_tensor_set(tensor_table, table.data(), 0, ggml_nbytes(tensor_table));
    ggml_backend_tensor_set(tensor_idx, idx.data(), 0, ggml_nbytes(tensor_idx));

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

    // Only the gathered rows are touched: read n_idx rows + write n_idx rows (plus
    // the small index array). Table size does not enter the moved-bytes count.
    const double bytes_per_iter = static_cast<double>(
        ggml_nbytes(tensor_result) * 2 + ggml_nbytes(tensor_idx));
    state.counters["bytes"] =
        benchmark::Counter(bytes_per_iter, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["elements"] = benchmark::Counter(
        static_cast<double>(ggml_nelements(tensor_result)),
        benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// GPT-2 embedding-lookup shapes (nc, nrows, n_idx):
//   token embedding: [n_embd, n_vocab] gather n_tokens  (768, 50257, N)
//   position embed.: [n_embd, n_ctx]   gather n_tokens  (768, 1024,  N)
#define GET_ROWS_SHAPES(BENCH)                                                                      \
    BENCH->Args({768, 50257, 64})                                                                   \
        ->Args({768, 50257, 256})                                                                   \
        ->Args({768, 50257, 1024})                                                                  \
        ->Args({768, 1024, 1024})                                                                   \
        ->UseRealTime()

GET_ROWS_SHAPES(BENCHMARK(bench_get_rows<BackendType::CPU>));
GET_ROWS_SHAPES(BENCHMARK(bench_get_rows<BackendType::HSA>));
GET_ROWS_SHAPES(BENCHMARK(bench_get_rows<BackendType::GPU>));
