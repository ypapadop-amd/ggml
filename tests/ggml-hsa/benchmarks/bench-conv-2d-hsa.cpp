// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Benchmark for GGML_OP_CONV_2D across backends. ggml_conv_2d expands to an
// im2col + mul_mat (+ reshape) subgraph; this times the whole convolution.
// CONV_2D is compute-bound, so the reported metric is FLOPS. Shapes mirror the
// MNIST-CNN layers: conv1 [28,28,1]*[3,3,1,8] and conv2 [14,14,8]*[3,3,8,16],
// with stride 1, padding 1 (same-size output), over a batch.

#include "bench-hsa-common.hpp"

#include <cstdint>

// Benchmarks out = conv_2d(kernel[3,3,IC,OC], input[W,H,IC,N]) with stride 1,
// padding 1, dilation 1. state.range = {W, H, IC, OC, N}.
template <BackendType Backend>
void bench_conv_2d(benchmark::State & state) {
    ggml_backend_t backend = make_backend(Backend, state);
    if (backend == nullptr) {
        return;
    }

    const std::int64_t W  = state.range(0);
    const std::int64_t H  = state.range(1);
    const std::int64_t IC = state.range(2);
    const std::int64_t OC = state.range(3);
    const std::int64_t N  = state.range(4);
    const std::int64_t KW = 3;
    const std::int64_t KH = 3;

    // conv_2d expands to several nodes (im2col, mul_mat, reshape); size the context
    // and graph generously and let ggml_gallocr allocate the intermediates.
    const std::size_t ctx_size =
        64 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, IC, OC);
    ggml_tensor * tensor_input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, N);
    ggml_tensor * tensor_result =
        ggml_conv_2d(ctx, tensor_kernel, tensor_input, 1, 1, 1, 1, 1, 1);
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

    const std::vector<float> K = make_data(ggml_nelements(tensor_kernel), 0.1f);
    const std::vector<float> A = make_data(ggml_nelements(tensor_input), -1.0f);
    ggml_backend_tensor_set(tensor_kernel, K.data(), 0, ggml_nbytes(tensor_kernel));
    ggml_backend_tensor_set(tensor_input, A.data(), 0, ggml_nbytes(tensor_input));

    // warm up (also triggers any one-time JIT compile so it's not measured; also
    // catches sub-ops like im2col/mul_mat that a backend may not support)
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

    // Output is [OW, OH, OC, N]; each output element is a dot product over KW*KH*IC.
    const double out_elems = static_cast<double>(tensor_result->ne[0]) *
                             static_cast<double>(tensor_result->ne[1]) *
                             static_cast<double>(tensor_result->ne[2]) *
                             static_cast<double>(tensor_result->ne[3]);
    const double flops_per_iter =
        2.0 * out_elems * static_cast<double>(KW * KH * IC);
    state.counters["FLOPS"] =
        benchmark::Counter(flops_per_iter, benchmark::Counter::kIsIterationInvariantRate);

    ggml_free(ctx);
    ggml_backend_free(backend);
}

// MNIST-CNN conv layers {W, H, IC, OC, N} with 3x3 kernels, stride 1, pad 1:
//   conv1: [28,28,1] -> 8 channels
//   conv2: [14,14,8] -> 16 channels
#define CONV_2D_SHAPES(BENCH)                                                                       \
    BENCH->Args({28, 28, 1, 8, 500})                                                                \
        ->Args({28, 28, 1, 8, 1000})                                                                \
        ->Args({14, 14, 8, 16, 500})                                                                \
        ->Args({14, 14, 8, 16, 1000})                                                               \
        ->UseRealTime()

CONV_2D_SHAPES(BENCHMARK(bench_conv_2d<BackendType::CPU>));
CONV_2D_SHAPES(BENCHMARK(bench_conv_2d<BackendType::HSA>));
CONV_2D_SHAPES(BENCHMARK(bench_conv_2d<BackendType::GPU>));
