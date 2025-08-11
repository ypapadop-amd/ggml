// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "ggml.h"

#include "ggml-cuda.h"
#include "ggml-hsa.h"

template<typename T>
auto create_data(std::size_t N, T value) {
    std::vector<T> v(N);
    std::iota(std::begin(v), std::end(v), value);
    return v;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (const auto & t : v) {
        os << ' ' << t;
    }
    return os << " ]";
}

#define ZERO_COPY

int main(int argc, char* argv[]) {
    std::size_t N = 32;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }

    // create data
    using value_type = float;
    constexpr auto ggml_type = GGML_TYPE_F32;
    const std::vector<value_type> A = create_data<value_type>(N, 11);
    const std::vector<value_type> B = create_data<value_type>(N, 12);
    const std::vector<value_type> C = create_data<value_type>(N, 10);

    // create HIP graph
    ggml_backend_t hip_backend = ggml_backend_cuda_init(0);
    if (hip_backend == nullptr) {
        std::cerr << "Could not create backend\n";
        return EXIT_FAILURE;
    }

    const std::size_t hip_tensor_count = 3;
    ggml_gallocr_t hip_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hip_backend));

    // allocate tensors on HIP memory
    const std::size_t hip_ctx_size =
        hip_tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(hip_tensor_count, false);
    ggml_init_params hip_params = {
        /*.mem_size   =*/ hip_ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true };
    ggml_context * hip_ctx = ggml_init(hip_params);
    ggml_tensor * tensor_a = ggml_new_tensor_1d(hip_ctx, ggml_type, N);
    ggml_set_name(tensor_a, "A");
    ggml_tensor * tensor_b = ggml_new_tensor_1d(hip_ctx, ggml_type, N);
    ggml_set_name(tensor_b, "B");

    // create graph
    ggml_cgraph * hip_gf = ggml_new_graph_custom(hip_ctx, hip_tensor_count, /*grads*/ false);

    // add operation
    ggml_tensor * hip_result = ggml_add(hip_ctx, tensor_a, tensor_b);
    ggml_set_name(hip_result, "A + B");

    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(hip_ctx, hip_backend);
    if (hip_buffer == nullptr) {
        std::cerr << "Could not allocate buffer for tensors\n";
        return EXIT_FAILURE;
    }

    ggml_build_forward_expand(hip_gf, hip_result);
    if (!ggml_gallocr_alloc_graph(hip_galloc, hip_gf)) {
        std::cerr << "Could not allocate graph\n";
        return EXIT_FAILURE;
    }

    // create HSA graph
    ggml_backend_t hsa_backend = ggml_backend_hsa_init(0);
    if (hsa_backend == nullptr) {
        std::cerr << "Could not create backend\n";
        return EXIT_FAILURE;
    }

    const std::size_t hsa_tensor_count = 3;
    ggml_gallocr_t hsa_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hsa_backend));

    // allocate tensors on HSA memory
    const std::size_t hsa_ctx_size =
        hsa_tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(hsa_tensor_count, false);
    ggml_init_params hsa_params = {
        /*.mem_size   =*/ hsa_ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true };
    ggml_context * hsa_ctx = ggml_init(hsa_params);
#ifdef ZERO_COPY
    ggml_tensor * tensor_hip_result = ggml_backend_hsa_tensor_alias(hsa_ctx, hip_result);
    ggml_set_name(tensor_hip_result, "A + B (alias)");
#else
    ggml_tensor * tensor_hip_result = ggml_new_tensor_1d(hsa_ctx, ggml_type, N);
    ggml_set_name(tensor_hip_result, "A + B (copy)");
#endif
    ggml_tensor * tensor_c = ggml_new_tensor_1d(hsa_ctx, ggml_type, N);
    ggml_set_name(tensor_c, "C");

    // create graph
    ggml_cgraph * hsa_gf = ggml_new_graph_custom(hsa_ctx, hsa_tensor_count, /*grads*/ false);

    // add operation
    ggml_tensor * hsa_result = ggml_sub(hsa_ctx, tensor_hip_result, tensor_c);
    ggml_set_name(hsa_result, "(A + B) - C");

    ggml_backend_buffer_t hsa_buffer = ggml_backend_alloc_ctx_tensors(hsa_ctx, hsa_backend);
    if (hsa_buffer == nullptr) {
        std::cerr << "Could not allocate buffer for tensors\n";
        return EXIT_FAILURE;
    }

    ggml_build_forward_expand(hsa_gf, hsa_result);
    if (!ggml_gallocr_alloc_graph(hsa_galloc, hsa_gf)) {
        std::cerr << "Could not allocate graph\n";
        return EXIT_FAILURE;
    }

    // copy data in
    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));
    ggml_backend_tensor_set(tensor_c, std::data(C), 0, ggml_nbytes(tensor_c));

    // execute HIP graph
    if (ggml_backend_graph_compute(hip_backend, hip_gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Execution failed\n";
        return EXIT_FAILURE;
    }

#ifndef ZERO_COPY
    // copy data from HIP to HSA graph
    ggml_backend_tensor_copy(hip_result, tensor_hip_result);
#endif

    // execute HSA graph
    if (ggml_backend_graph_compute(hsa_backend, hsa_gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Execution failed\n";
        return EXIT_FAILURE;
    }

    // copy data out and print
    std::vector<value_type> hip_result_data(N);
    ggml_backend_tensor_get(hip_result, std::data(hip_result_data), 0, ggml_nbytes(hip_result));

    // copy data out and print
    std::vector<value_type> hsa_result_data(N);
    ggml_backend_tensor_get(hsa_result, std::data(hsa_result_data), 0, ggml_nbytes(hsa_result));
    std::cout << "A           = " << A << '\n'
              << "B           = " << B << '\n'
              << "A + B       = " << hip_result_data << '\n'
              << "C           = " << C << '\n'
              << "(A + B) - C = " << hsa_result_data << '\n';

    // free resources
    ggml_gallocr_free(hsa_galloc);
    ggml_free(hsa_ctx);
    ggml_backend_buffer_free(hsa_buffer);
    ggml_backend_free(hsa_backend);

    ggml_gallocr_free(hip_galloc);
    ggml_free(hip_ctx);
    ggml_backend_buffer_free(hip_buffer);
    ggml_backend_free(hip_backend);

    return EXIT_SUCCESS;
}
