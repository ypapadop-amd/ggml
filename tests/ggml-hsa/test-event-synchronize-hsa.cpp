// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Exercises event_synchronize: the CPU thread must block until the recorded work completes, so the
// read-back afterwards observes correct results.

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "ggml.h"

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

namespace {

int fail(const char * msg) {
    std::cerr << "FAIL: " << msg << '\n';
    return EXIT_FAILURE;
}

// Verifies that the device advertises event support.
bool device_reports_events(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    return props.caps.events;
}

int run(ggml_backend_t backend) {
    constexpr std::size_t N = 32;

    std::vector<float> A(N);
    std::vector<float> B(N);
    for (std::size_t i = 0; i < N; ++i) {
        A[i] = 10.0f + static_cast<float>(i);
        B[i] = 2.0f + static_cast<float>(i);
    }

    const std::size_t alignment = ggml_backend_get_alignment(backend);
    const std::size_t tensor_count = 3;
    const std::size_t buffer_size = tensor_count * GGML_PAD((N * sizeof(float)), alignment);
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
        return fail("could not allocate tensors");
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), tensor_count, /*grads*/ false);
    ggml_tensor * tensor_result = ggml_add(ctx.get(), tensor_a, tensor_b);
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        return fail("operation not supported");
    }
    ggml_build_forward_expand(gf, tensor_result);
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        return fail("could not allocate graph");
    }

    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    std::unique_ptr<ggml_backend_event, decltype(&ggml_backend_event_free)> event{
        ggml_backend_event_new(dev), ggml_backend_event_free};
    if (event == nullptr) {
        return fail("event_new returned nullptr");
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        return fail("graph compute failed");
    }

    ggml_backend_event_record(event.get(), backend);
    // CPU-side blocking wait: on return the work is guaranteed complete.
    ggml_backend_event_synchronize(event.get());

    std::vector<float> result(N);
    ggml_backend_tensor_get(tensor_result, std::data(result), 0, ggml_nbytes(tensor_result));
    for (std::size_t i = 0; i < N; ++i) {
        const float expected = A[i] + B[i];
        if (result[i] != expected) {
            std::cerr << "FAIL: result[" << i << "] = " << result[i] << ", expected " << expected
                      << '\n';
            return EXIT_FAILURE;
        }
    }

    std::cout << "event_synchronize produced correct results for " << N << " elements\n";
    return EXIT_SUCCESS;
}

} // namespace

int main() {
    ggml_backend_t backend = nullptr;

#ifdef GGML_USE_HSA
    backend = ggml_backend_hsa_init(0);
#endif

    if (backend == nullptr) {
        return fail("could not create HSA backend");
    }

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_guard{
        backend, ggml_backend_free};

    if (!device_reports_events(backend)) {
        return fail("device does not report event support");
    }

    return run(backend);
}
