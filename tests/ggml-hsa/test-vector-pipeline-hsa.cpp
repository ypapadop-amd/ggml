// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Pipeline across two backends: HIP computes A + B, then the NPU (HSA) computes (A + B) - C, and
// the result is checked.
//
// The NPU reads the HIP result in place and writes its own result straight into HIP memory, with no
// copies. Needs a warm kernel cache for SUB f32; see "Running HIP and HSA in the Same Process" in
// src/ggml-hsa/README.md.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-hsa.h"
#include "ggml.h"

template <typename T>
auto create_data(std::size_t N, T value) {
    std::vector<T> v(N);
    std::iota(std::begin(v), std::end(v), value);
    return v;
}

template <typename T>
std::ostream & operator<<(std::ostream & os, const std::vector<T> & v) {
    os << "[";
    for (const auto & t : v) {
        os << ' ' << t;
    }
    return os << " ]";
}

int main(int argc, char * argv[]) {
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
        std::cerr << "Could not create HIP backend\n";
        return EXIT_FAILURE;
    }

    const std::size_t hip_tensor_count = 4;
    ggml_gallocr_t hip_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hip_backend));

    // allocate tensors on HIP memory
    const std::size_t hip_ctx_size = hip_tensor_count * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(hip_tensor_count, false);
    ggml_init_params hip_params = {
        /*.mem_size   =*/hip_ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * hip_ctx = ggml_init(hip_params);
    ggml_tensor * tensor_a = ggml_new_tensor_1d(hip_ctx, ggml_type, N);
    ggml_set_name(tensor_a, "A");
    ggml_tensor * tensor_b = ggml_new_tensor_1d(hip_ctx, ggml_type, N);
    ggml_set_name(tensor_b, "B");
    // HIP memory the NPU writes its result into
    ggml_tensor * tensor_d = ggml_new_tensor_1d(hip_ctx, ggml_type, N);
    ggml_set_name(tensor_d, "(A + B) - C (HIP)");

    ggml_cgraph * hip_gf = ggml_new_graph_custom(hip_ctx, hip_tensor_count, /*grads*/ false);
    ggml_tensor * hip_result = ggml_add(hip_ctx, tensor_a, tensor_b);
    ggml_set_name(hip_result, "A + B");
    ggml_build_forward_expand(hip_gf, hip_result);

    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(hip_ctx, hip_backend);
    if (hip_buffer == nullptr) {
        std::cerr << "Could not allocate HIP tensors\n";
        return EXIT_FAILURE;
    }
    if (!ggml_gallocr_alloc_graph(hip_galloc, hip_gf)) {
        std::cerr << "Could not allocate HIP graph\n";
        return EXIT_FAILURE;
    }

    // create HSA graph
    ggml_backend_t hsa_backend = ggml_backend_hsa_init(0);
    if (hsa_backend == nullptr) {
        std::cerr << "Could not create HSA backend\n";
        return EXIT_FAILURE;
    }

    const std::size_t hsa_tensor_count = 3;
    ggml_gallocr_t hsa_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(hsa_backend));

    // allocate tensors on HSA memory
    const std::size_t hsa_ctx_size = hsa_tensor_count * ggml_tensor_overhead() +
                                     ggml_graph_overhead_custom(hsa_tensor_count, false);
    ggml_init_params hsa_params = {
        /*.mem_size   =*/hsa_ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * hsa_ctx = ggml_init(hsa_params);
    ggml_tensor * tensor_hip_result = ggml_new_tensor_1d(hsa_ctx, ggml_type, N);
    // map the HIP buffer for the NPU and place A + B on it; the NPU then reads it in place
    ggml_backend_buffer_t hip_imported = ggml_backend_hsa_buffer_import(0, hip_buffer);
    if (hip_imported == nullptr) {
        std::cerr << "Could not import the HIP buffer\n";
        return EXIT_FAILURE;
    }
    if (ggml_backend_hsa_tensor_alloc_alias(hip_imported, tensor_hip_result, hip_result) !=
        GGML_STATUS_SUCCESS) {
        std::cerr << "Could not alias A + B\n";
        return EXIT_FAILURE;
    }
    ggml_set_name(tensor_hip_result, "A + B (alias)");
    ggml_tensor * tensor_c = ggml_new_tensor_1d(hsa_ctx, ggml_type, N);
    ggml_set_name(tensor_c, "C");

    ggml_cgraph * hsa_gf = ggml_new_graph_custom(hsa_ctx, hsa_tensor_count, /*grads*/ false);
    ggml_tensor * hsa_result = ggml_sub(hsa_ctx, tensor_hip_result, tensor_c);
    ggml_set_name(hsa_result, "(A + B) - C");
    ggml_build_forward_expand(hsa_gf, hsa_result);

    // the NPU writes (A + B) - C straight into D; must happen before the graph is allocated
    if (ggml_backend_hsa_tensor_alloc_alias(hip_imported, hsa_result, tensor_d) !=
        GGML_STATUS_SUCCESS) {
        std::cerr << "Could not place the NPU result in HIP memory\n";
        return EXIT_FAILURE;
    }

    ggml_backend_buffer_t hsa_buffer = ggml_backend_alloc_ctx_tensors(hsa_ctx, hsa_backend);
    if (hsa_buffer == nullptr) {
        std::cerr << "Could not allocate HSA tensors\n";
        return EXIT_FAILURE;
    }
    if (!ggml_gallocr_alloc_graph(hsa_galloc, hsa_gf)) {
        std::cerr << "Could not allocate HSA graph\n";
        return EXIT_FAILURE;
    }

    // copy data in
    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));
    ggml_backend_tensor_set(tensor_c, std::data(C), 0, ggml_nbytes(tensor_c));

    // execute HIP graph; ggml_backend_graph_compute waits for completion, so A + B is ready for
    // the NPU
    if (ggml_backend_graph_compute(hip_backend, hip_gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "HIP execution failed\n";
        return EXIT_FAILURE;
    }

    // execute HSA graph
    if (ggml_backend_graph_compute(hsa_backend, hsa_gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "HSA execution failed\n";
        return EXIT_FAILURE;
    }

    // copy data out and print
    std::vector<value_type> hip_result_data(N);
    ggml_backend_tensor_get(hip_result, std::data(hip_result_data), 0, ggml_nbytes(hip_result));
    std::vector<value_type> hsa_result_data(N);
    // read the NPU result back through HIP
    ggml_backend_tensor_get(tensor_d, std::data(hsa_result_data), 0, ggml_nbytes(tensor_d));
    std::cout << "A           = " << A << '\n'
              << "B           = " << B << '\n'
              << "A + B       = " << hip_result_data << '\n'
              << "C           = " << C << '\n'
              << "(A + B) - C = " << hsa_result_data << '\n';

    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < N; ++i) {
        mismatches +=
            (hip_result_data[i] != A[i] + B[i] || hsa_result_data[i] != A[i] + B[i] - C[i]);
    }
    if (mismatches != 0) {
        std::cerr << mismatches << " mismatch(es) out of " << N << '\n';
    }

    // free resources; the imported buffer goes before the HIP buffer it maps
    ggml_gallocr_free(hsa_galloc);
    ggml_free(hsa_ctx);
    ggml_backend_buffer_free(hsa_buffer);
    ggml_backend_buffer_free(hip_imported);
    ggml_backend_free(hsa_backend);

    ggml_gallocr_free(hip_galloc);
    ggml_free(hip_ctx);
    ggml_backend_buffer_free(hip_buffer);
    ggml_backend_free(hip_backend);

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
