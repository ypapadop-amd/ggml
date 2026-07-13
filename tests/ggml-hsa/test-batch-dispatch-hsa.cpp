// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

// Baseline / benchmark for batched packet dispatch on the HSA (AIE/NPU) backend.
//
// Builds a single graph containing a long chain of vector additions. None of the
// ops require host-side synchronization (native f32 element-wise), so the whole
// chain is dispatched back-to-back and drained once at the end. This isolates the
// per-dispatch doorbell cost that the multi-packet-per-doorbell work targets.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "ggml.h"

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

namespace {

std::vector<float> make_data(std::size_t n, float start) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = start + static_cast<float>(i % 16);
    }
    return v;
}

int run(ggml_backend_t backend, std::size_t N, std::size_t n_ops, int iters) {
    // tensors: two inputs + one result per op
    const std::size_t tensor_count = 2 + n_ops;

    const std::size_t alignment = ggml_backend_get_alignment(backend);
    const std::size_t input_bytes = 2 * GGML_PAD((N * sizeof(float)), alignment);
    const std::size_t buffer_size = input_bytes;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        ggml_backend_alloc_buffer(backend, buffer_size), ggml_backend_buffer_free};
    ggml_tallocr alloc = ggml_tallocr_new(buffer.get());
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};

    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params = {/*.mem_size   =*/ctx_size,
                               /*.mem_buffer =*/nullptr,
                               /*.no_alloc   =*/true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * tensor_a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, N);
    ggml_tensor * tensor_b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, N);
    if ((ggml_tallocr_alloc(&alloc, tensor_a) != GGML_STATUS_SUCCESS) ||
        (ggml_tallocr_alloc(&alloc, tensor_b) != GGML_STATUS_SUCCESS)) {
        std::cerr << "Could not allocate input tensors\n";
        return EXIT_FAILURE;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), tensor_count, /*grads*/ false);

    // chain: r = a + b, then r = r + b, ... n_ops times
    ggml_tensor * result = ggml_add(ctx.get(), tensor_a, tensor_b);
    for (std::size_t i = 1; i < n_ops; ++i) {
        result = ggml_add(ctx.get(), result, tensor_b);
    }

    if (!ggml_backend_supports_op(backend, result)) {
        std::cerr << "Operation not supported\n";
        return EXIT_FAILURE;
    }
    ggml_build_forward_expand(gf, result);
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        std::cerr << "Could not allocate graph\n";
        return EXIT_FAILURE;
    }

    const std::vector<float> A = make_data(N, 1.0f);
    const std::vector<float> B = make_data(N, 1.0f);
    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));

    // warm-up (JIT compile / cache population happens here)
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Warm-up execution failed\n";
        return EXIT_FAILURE;
    }

    ggml_time_init();
    // Split each iteration into the host-side dispatch phase (building/writing packets and ringing
    // doorbells, measured by graph_compute_async) and the device-drain phase (waiting for the queue
    // to empty, measured by synchronize). This isolates where the per-op time actually goes.
    int64_t dispatch_us = 0;
    int64_t drain_us = 0;
    const int64_t t0 = ggml_time_us();
    for (int it = 0; it < iters; ++it) {
        const int64_t d0 = ggml_time_us();
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "Execution failed\n";
            return EXIT_FAILURE;
        }
        const int64_t d1 = ggml_time_us();
        ggml_backend_synchronize(backend);
        const int64_t d2 = ggml_time_us();
        dispatch_us += d1 - d0;
        drain_us += d2 - d1;
    }
    const int64_t t1 = ggml_time_us();

    const double total_us = static_cast<double>(t1 - t0);
    const double per_iter_us = total_us / iters;
    const double per_op_us = per_iter_us / static_cast<double>(n_ops);
    const double dispatch_per_op = static_cast<double>(dispatch_us) / iters / n_ops;
    const double drain_per_op = static_cast<double>(drain_us) / iters / n_ops;

    std::vector<float> out(N);
    ggml_backend_tensor_get(result, std::data(out), 0, ggml_nbytes(result));

    // result[j] = A[j] + (n_ops * B[j])
    const float expected0 = A[0] + static_cast<float>(n_ops) * B[0];
    const bool ok = out[0] == expected0;

    std::cout << "N=" << N << "  n_ops=" << n_ops << "  iters=" << iters << '\n';
    std::cout << "total=" << total_us << " us"
              << "  per_iter=" << per_iter_us << " us"
              << "  per_op=" << per_op_us << " us\n";
    std::cout << "per_op breakdown: dispatch=" << dispatch_per_op << " us"
              << "  drain=" << drain_per_op << " us\n";
    std::cout << "check: out[0]=" << out[0] << " expected=" << expected0 << " -> "
              << (ok ? "OK" : "FAIL") << '\n';

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace

int main(int argc, char * argv[]) {
    // NOTE: with the current single-packet-per-doorbell dispatch the kernarg arena is
    // sized to the queue depth and only recycled on a queue-full stall, so a long
    // back-to-back chain overflows it. Default n_ops is kept at/under a typical queue
    // depth; raise it once multi-packet batching lands.
    std::size_t N = 1024;
    std::size_t n_ops = 32;
    int iters = 20;

    if (argc > 1) {
        N = static_cast<std::size_t>(std::atoll(argv[1]));
    }
    if (argc > 2) {
        n_ops = static_cast<std::size_t>(std::atoll(argv[2]));
    }
    if (argc > 3) {
        iters = std::atoi(argv[3]);
    }

    ggml_backend_t backend = {};

#ifdef GGML_USE_HSA
    std::cout << "Using HSA backend\n";
    backend = ggml_backend_hsa_init(0);
#endif

    if (backend == nullptr) {
        std::cerr << "Could not create backend\n";
        return EXIT_FAILURE;
    }

    const int ret = run(backend, N, n_ops, iters);

    ggml_backend_free(backend);
    return ret;
}
