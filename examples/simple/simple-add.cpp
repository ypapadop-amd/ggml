#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

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

int main(void) {
    // create data
    const std::size_t N = 256;
    const std::vector<std::int32_t> A = create_data<std::int32_t>(N, 0);
    const std::vector<std::int32_t> B = create_data<std::int32_t>(N, 1);

    // initialize GGML backend and allocators
    ggml_backend_t backend = {};

#ifdef GGML_USE_HSA
    std::cout << "Using HSA backend\n";
    backend = ggml_backend_hsa_init(0);
#endif

#ifdef GGML_USE_CUDA
    if (!backend) {
        std::cout << "Using CUDA backend\n";
        backend = ggml_backend_cuda_init(0); // init device 0
    }
#endif

    if (backend == nullptr) {
        std::cerr << "Could not create backend\n";
        return EXIT_FAILURE;
    }
    const std::size_t alignment = ggml_backend_get_alignment(backend);
    const std::size_t tensor_count = 3;
    const std::size_t buffer_size = tensor_count * GGML_PAD((N * sizeof(std::int32_t)), alignment);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    ggml_tallocr alloc = ggml_tallocr_new(buffer);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // allocate tensors on HSA memory
    const std::size_t ctx_size =
        tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false) + 64;
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor_a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * tensor_b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    if ((ggml_tallocr_alloc(&alloc, tensor_a) != GGML_STATUS_SUCCESS) ||
        (ggml_tallocr_alloc(&alloc, tensor_b) != GGML_STATUS_SUCCESS)) {
        std::cerr << "Could not allocate tensor\n";
        return EXIT_FAILURE;
    }

    // create graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, tensor_count, /*grads*/ false);

    // add vector-add operation
    ggml_tensor * tensor_result = ggml_add(ctx, tensor_a, tensor_b);
    if (!ggml_backend_supports_op(backend, tensor_result)) {
        std::cerr << "Operation not supported\n";
        return EXIT_FAILURE;
    }
    ggml_build_forward_expand(gf, tensor_result);
    ggml_gallocr_alloc_graph(galloc, gf);

    // copy data in (can be avoided if data created directly in tensors)
    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));

    // execute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Execution failed\n";
        return EXIT_FAILURE;
    }

    // copy data out and print
    std::vector<std::int32_t> result(N);
    ggml_backend_tensor_get(tensor_result, std::data(result), 0, ggml_nbytes(tensor_result));
    std::cout << "A =     " << A << '\n';
    std::cout << "B =     " << B << '\n';
    std::cout << "A + B = " << result << '\n';

    // free resources
    ggml_free(ctx);
    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);

    return EXIT_SUCCESS;
}
