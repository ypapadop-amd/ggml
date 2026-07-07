// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

// Traits mapping C++ type to ggml enum
template <typename T>
struct cpp_to_ggml_type;

template <>
struct cpp_to_ggml_type<float> {
    static constexpr ggml_type value = GGML_TYPE_F32;
};

template <>
struct cpp_to_ggml_type<int32_t> {
    static constexpr ggml_type value = GGML_TYPE_I32;
};

template <>
struct cpp_to_ggml_type<ggml_bf16_t> {
    static constexpr ggml_type value = GGML_TYPE_BF16;
};

// Data creation
template <typename T>
T from_float(float f);

template <>
float from_float<float>(float f) {
    return f;
}

template <>
int32_t from_float<int32_t>(float f) {
    return static_cast<int32_t>(f);
}

template <>
ggml_bf16_t from_float<ggml_bf16_t>(float f) {
    return ggml_fp32_to_bf16(f);
}

template <typename T>
float to_float(T v);

template <>
float to_float<float>(float v) {
    return v;
}

template <>
float to_float<int32_t>(int32_t v) {
    return static_cast<float>(v);
}

template <>
float to_float<ggml_bf16_t>(ggml_bf16_t v) {
    return ggml_bf16_to_fp32(v);
}

template <typename T>
std::vector<T> make_data(std::size_t n, float start) {
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = from_float<T>(start + static_cast<float>(i));
    }
    return v;
}

template <typename T>
void print_vec(std::ostream & os, const std::vector<T> & v) {
    os << "[";
    for (const auto & t : v) {
        os << ' ' << to_float(t);
    }
    os << " ]";
}

template <typename T>
int run(ggml_backend_t backend, std::size_t N, const char * op) {
    constexpr ggml_type tensor_type = cpp_to_ggml_type<T>::value;

    const std::vector<T> A = make_data<T>(N, 10.0f);
    const std::vector<T> B = make_data<T>(N, 2.0f);

    const std::size_t alignment = ggml_backend_get_alignment(backend);
    const std::size_t tensor_count = 3;
    const std::size_t buffer_size = tensor_count * GGML_PAD((N * sizeof(T)), alignment);
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

    ggml_tensor * tensor_a = ggml_new_tensor_1d(ctx.get(), tensor_type, N);
    ggml_tensor * tensor_b = ggml_new_tensor_1d(ctx.get(), tensor_type, N);
    if ((ggml_tallocr_alloc(&alloc, tensor_a) != GGML_STATUS_SUCCESS) ||
        (ggml_tallocr_alloc(&alloc, tensor_b) != GGML_STATUS_SUCCESS)) {
        std::cerr << "Could not allocate tensor\n";
        return EXIT_FAILURE;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), tensor_count, /*grads*/ false);

    ggml_tensor * tensor_result = nullptr;
    if (std::strcmp(op, "+") == 0) {
        tensor_result = ggml_add(ctx.get(), tensor_a, tensor_b);
    } else if (std::strcmp(op, "-") == 0) {
        tensor_result = ggml_sub(ctx.get(), tensor_a, tensor_b);
    } else if (std::strcmp(op, "*") == 0) {
        tensor_result = ggml_mul(ctx.get(), tensor_a, tensor_b);
    } else if (std::strcmp(op, "/") == 0) {
        tensor_result = ggml_div(ctx.get(), tensor_a, tensor_b);
    } else {
        std::cerr << "Unknown operation \"" << op << "\".\n";
        return EXIT_FAILURE;
    }

    if (!ggml_backend_supports_op(backend, tensor_result)) {
        std::cerr << "Operation not supported\n";
        return EXIT_FAILURE;
    }
    ggml_build_forward_expand(gf, tensor_result);
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        std::cerr << "Could not allocate graph\n";
        return EXIT_FAILURE;
    }

    ggml_backend_tensor_set(tensor_a, std::data(A), 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, std::data(B), 0, ggml_nbytes(tensor_b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Execution failed\n";
        return EXIT_FAILURE;
    }

    std::vector<T> result(N);
    ggml_backend_tensor_get(tensor_result, std::data(result), 0, ggml_nbytes(tensor_result));
    std::cout << "A =     ";
    print_vec(std::cout, A);
    std::cout << '\n';
    std::cout << "B =     ";
    print_vec(std::cout, B);
    std::cout << '\n';
    std::cout << "A " << op << " B = ";
    print_vec(std::cout, result);
    std::cout << '\n';

    return EXIT_SUCCESS;
}

int main(int argc, char * argv[]) {
    std::size_t N = 32;
    const char * op = "+";
    std::string dtype = "f32";

    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        op = argv[2];
    }
    if (argc > 3) {
        dtype = argv[3];
    }

    std::cout << "dtype=" << dtype << "  N=" << N << "  op=" << op << '\n';

    ggml_backend_t backend = {};

#ifdef GGML_USE_HSA
    std::cout << "Using HSA backend\n";
    backend = ggml_backend_hsa_init(0);
#endif

#ifdef GGML_USE_CUDA
    if (!backend) {
        std::cout << "Using CUDA backend\n";
        backend = ggml_backend_cuda_init(0);
    }
#endif

    if (backend == nullptr) {
        std::cerr << "Could not create backend\n";
        return EXIT_FAILURE;
    }

    int ret;
    if (dtype == "f32") {
        ret = run<float>(backend, N, op);
    } else if (dtype == "i32") {
        ret = run<int32_t>(backend, N, op);
    } else if (dtype == "bf16") {
        ret = run<ggml_bf16_t>(backend, N, op);
    } else {
        std::cerr << "Unknown dtype \"" << dtype << "\". Use f32, i32, or bf16.\n";
        ret = EXIT_FAILURE;
    }

    ggml_backend_free(backend);
    return ret;
}
