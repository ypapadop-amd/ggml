// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_MUL_MAT across backends, parameterized by input
// dtype and M/N/K shape.

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
#include <numeric>
#include <vector>

namespace {

enum class BackendType {
    CPU,
    GPU,
    HSA,
};

template <typename T>
struct type_to_ggml_type;

template <>
struct type_to_ggml_type<float> {
    static constexpr ggml_type ggml_type_v = GGML_TYPE_F32;
};

template <>
struct type_to_ggml_type<ggml_bf16_t> {
    static constexpr ggml_type ggml_type_v = GGML_TYPE_BF16;
};

template <typename T>
constexpr ggml_type type_to_ggml_type_v = type_to_ggml_type<T>::ggml_type_v;

template <typename T>
std::vector<T> make_data(std::size_t n);

template <>
std::vector<float> make_data<float>(std::size_t n) {
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), 0.0f);
    return v;
}

template <>
std::vector<ggml_bf16_t> make_data<ggml_bf16_t>(std::size_t n) {
    std::vector<ggml_bf16_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = ggml_fp32_to_bf16(static_cast<float>(i));
    }
    return v;
}

} // namespace

// Benchmarks C = A * B^T, with A: [K, M], B: [K, N], C: [M, N] (all row-major
// in ggml's ne[0]-fastest convention), using state.range(0/1/2) for M/N/K.
template <BackendType Backend, typename T>
void bench_mul_mat(benchmark::State & state) {
    // initialize backend
    ggml_backend_t backend = {};
    switch (Backend) {
        case BackendType::CPU:
            backend = ggml_backend_cpu_init();
            break;
        case BackendType::GPU:
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
#else
            state.SkipWithError("CUDA backend not available.");
            return;
#endif
            break;
        case BackendType::HSA:
#ifdef GGML_USE_HSA
            backend = ggml_backend_hsa_init(0);
#else
            state.SkipWithError("HSA backend not available.");
            return;
#endif
            break;
        default:
            state.SkipWithError("Invalid backend type.");
            return;
    }
    if (backend == nullptr) {
        state.SkipWithError("Backend creation failed.");
        return;
    }

    const auto ggml_datatype = type_to_ggml_type_v<T>;
    const std::int64_t M = state.range(0);
    const std::int64_t N = state.range(1);
    const std::int64_t K = state.range(2);

    // create graph
    const std::size_t tensor_count = 3;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, ggml_datatype, K, M);
    ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, ggml_datatype, K, N);
    ggml_tensor * tensor_result = ggml_mul_mat(ctx, tensor_a, tensor_b);
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        state.SkipWithError("Operation not supported.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    // allocate space
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        state.SkipWithError("Tensor buffer allocation failed.");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    // build graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, tensor_count, /*grads*/ false);
    ggml_build_forward_expand(gf, tensor_result);

    // copy data in
    const std::vector<T> A = make_data<T>(tensor_a->ne[0] * tensor_a->ne[1]);
    const std::vector<T> B = make_data<T>(tensor_b->ne[0] * tensor_b->ne[1]);
    ggml_backend_tensor_set(tensor_a, A.data(), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, B.data(), 0, ggml_nbytes(tensor_b));

    // warm up (also triggers any one-time JIT compile so it's not measured)
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        state.SkipWithError("Warm-up graph compute error.");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    // execute
    for (auto _ : state) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            state.SkipWithError("Graph compute error.");
            break;
        }
    }

    const double flops_per_iter =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    state.counters["FLOPS"] =
        benchmark::Counter(flops_per_iter, benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

BENCHMARK(bench_mul_mat<BackendType::CPU, float>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();
BENCHMARK(bench_mul_mat<BackendType::CPU, ggml_bf16_t>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();

BENCHMARK(bench_mul_mat<BackendType::HSA, float>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();
BENCHMARK(bench_mul_mat<BackendType::HSA, ggml_bf16_t>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();

BENCHMARK(bench_mul_mat<BackendType::GPU, float>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();
BENCHMARK(bench_mul_mat<BackendType::GPU, ggml_bf16_t>)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->UseRealTime();
