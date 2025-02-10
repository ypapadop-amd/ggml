#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>

#include "ggml.h"
#include "ggml-hsa.h"

int main(void) {
    const std::size_t N = 256;
    std::array<std::int32_t, N> A = {};
    std::array<std::int32_t, N> B = {};

    std::iota(std::begin(A), std::end(A), 1);
    std::iota(std::begin(B), std::end(B), 10);

    ggml_backend_t backend = ggml_backend_hsa_init(0);
    const std::size_t alignment = ggml_backend_get_alignment(backend);
    const std::size_t tensor_count = 3;
    const std::size_t buffer_size = tensor_count * GGML_PAD((N * sizeof(std::int32_t)), alignment);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    ggml_tallocr alloc = ggml_tallocr_new(buffer);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    const std::size_t ctx_size = tensor_count * ggml_tensor_overhead() + ggml_graph_overhead_custom(tensor_count, false) + 64;
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * tensor_a = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * tensor_b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

    ggml_tallocr_alloc(&alloc, tensor_a);
    ggml_tallocr_alloc(&alloc, tensor_b);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, tensor_count, false);

    ggml_tensor * tensor_result = ggml_add(ctx, tensor_a, tensor_b);
    if (ggml_backend_supports_op(backend, tensor_result) == 0) {
        std::cout << "Operation not supported\n";
        return EXIT_FAILURE;
    }

    ggml_build_forward_expand(gf, tensor_result);

    ggml_gallocr_alloc_graph(galloc, gf);

    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::cout << "Execution failed\n";
    }

    std::array<std::int32_t, N> result = {};
    ggml_backend_tensor_get(tensor_result, std::data(result), 0, ggml_nbytes(tensor_result));

    std::cout << "add (" << result.size() << "):\n[";
    for (auto v : result) {
        std::cout << ' ' << v;
    }
    std::cout << " ]\n";

    ggml_free(ctx);
    ggml_gallocr_free(galloc);
    ggml_backend_free(backend);

    return 0;
}
