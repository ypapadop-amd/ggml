// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_UNARY_OP_RELU across backends, parameterized by input dtype
// and 4D shape. RELU is element-wise and memory-bound, so the shapes mirror the
// realistic MNIST activation tensors and the reported metric is memory bandwidth.
//
// The NPU (HSA) backend ships both an IRON and a Triton RELU kernel; which one is
// used is selected at kernel-JIT time by the GGML_HSA_PREFER_TRITON environment
// variable (unset/0 = IRON, 1 = Triton). Run this binary twice with the HSA
// filter, flipping that variable, to compare the two paths (see repro-relu.sh).

#include "bench-hsa-common.hpp"

#include <cstdint>
#include <vector>

namespace {

template <typename T>
struct type_to_ggml_type;

template <>
struct type_to_ggml_type<float> {
    static constexpr ggml_type ggml_type_v = GGML_TYPE_F32;
};

template <typename T>
constexpr ggml_type type_to_ggml_type_v = type_to_ggml_type<T>::ggml_type_v;

// RELU spans zero, so seed the input with alternating-sign values to exercise both
// the pass-through and the clamp-to-zero branches (the exact values do not affect
// the memory-bound timing).
template <typename T>
std::vector<T> make_data(std::size_t n) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<T>((i % 2 == 0) ? -static_cast<float>(i) : static_cast<float>(i));
    }
    return v;
}

} // namespace

// Benchmarks out = relu(a), with a and out both [ne0, ne1, ne2, ne3] in ggml's
// ne[0]-fastest layout, using state.range(0..3) for the four dimensions.
template <BackendType Backend, typename T>
void bench_relu(benchmark::State & state) {
    // initialize backend
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        return;
    }

    const auto ggml_datatype = type_to_ggml_type_v<T>;
    const std::int64_t ne0 = state.range(0);
    const std::int64_t ne1 = state.range(1);
    const std::int64_t ne2 = state.range(2);
    const std::int64_t ne3 = state.range(3);

    // create graph
    const std::size_t tensor_count = 2;
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_4d(ctx, ggml_datatype, ne0, ne1, ne2, ne3);
    ggml_tensor * tensor_result = ggml_relu(ctx, tensor_a);
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
    const std::vector<T> A = make_data<T>(ggml_nelements(tensor_a));
    ggml_backend_tensor_set(tensor_a, A.data(), 0, ggml_nbytes(tensor_a));

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

    // RELU is memory-bound: one element read + one written per element.
    const double bytes_per_iter =
        2.0 * static_cast<double>(ggml_nelements(tensor_a)) * static_cast<double>(sizeof(T));
    state.counters["bytes"] =
        benchmark::Counter(bytes_per_iter, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["elements"] = benchmark::Counter(
        static_cast<double>(ggml_nelements(tensor_a)), benchmark::Counter::kIsIterationInvariantRate);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

// Realistic MNIST activation shapes (ne0, ne1, ne2, ne3):
//   {500, 500, 1, 1}    FC1 hidden activation           (0.25M elements)
//   {14, 14, 16, 500}   conv2 output (post-bias) x batch (1.57M elements)
//   {28, 28, 8, 500}    conv1 output (post-bias) x batch (3.14M elements)
#define RELU_SHAPES(BENCH)                                                                          \
    BENCH->Args({500, 500, 1, 1})                                                                   \
        ->Args({14, 14, 16, 500})                                                                   \
        ->Args({28, 28, 8, 500})                                                                    \
        ->UseRealTime()

RELU_SHAPES(BENCHMARK(bench_relu<BackendType::CPU, float>));
RELU_SHAPES(BENCHMARK(bench_relu<BackendType::HSA, float>));
RELU_SHAPES(BENCHMARK(bench_relu<BackendType::GPU, float>));
