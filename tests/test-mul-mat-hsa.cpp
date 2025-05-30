#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <iostream>

#ifdef GGML_USE_HSA
#define USE_NPU 1
#endif

template<typename>
struct fundamental_to_ggml_type;

template<>
struct fundamental_to_ggml_type<int16_t> {
    inline static const auto ggml_type = GGML_TYPE_I16;
};

template<>
struct fundamental_to_ggml_type<float> {
    inline static const auto ggml_type = GGML_TYPE_F32;
};

struct ggml_tensor * ggml_mul_mat_i16(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b) {
    const int64_t ne[4] = { a->ne[0], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I16, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct test_model {
    ggml_tensor * a;
    ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    ggml_context * ctx;
    std::vector<uint8_t> buf;
};

template<typename T>
void load_model(test_model & model, T * a, T * b, int M, int N, int K, bool use_gpu = false) {
    const auto ggml_type = fundamental_to_ggml_type<T>::ggml_type;

    size_t buffer_size = 0;
    buffer_size += (M * N) * ggml_type_size(ggml_type); // tensor a
    buffer_size += (N * K) * ggml_type_size(ggml_type); // tensor b
    buffer_size += 1024; // overhead

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 2;
    ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_HSA
    if (use_gpu) {
        fprintf(stderr, "%s: using HSA backend\n", __func__);
        model.backend = ggml_backend_hsa_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_hsa_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, ggml_type, K, M);
    printf("Matrix A: [%i, %i]\n", M, K);
    model.b = ggml_new_tensor_2d(model.ctx, ggml_type, N, K);
    printf("Matrix B: [%i, %i]\n", K, N);

    // create a allocator
    ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);
    ggml_tallocr_alloc(&alloc, model.b);

    // copy data directly to device
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));
}

ggml_cgraph * build_graph(test_model& model) {
    const size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    model.buf.resize(buf_size);

    ggml_init_params params0 = {
        /*.mem_size   =*/ model.buf.size(),
        /*.mem_buffer =*/ model.buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a context to build the graph
    ggml_context * ctx = ggml_init(params0);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // zT = x @ yT
#if USE_NPU
    // HACK: we don't need to transpose
    ggml_tensor * c = ggml_mul_mat_i16(ctx, model.a, model.b);
#else
    ggml_tensor * b_transposed = ggml_cont(ctx, ggml_transpose(ctx, model.b));
    ggml_tensor * c_transposed = ggml_mul_mat(ctx, model.a, b_transposed);
    ggml_tensor * c = ggml_cont(ctx, ggml_transpose(ctx, c_transposed));
#endif

    // z = (zT)T
    ggml_build_forward_expand(gf, c);

    // delete the context used to build the graph
    ggml_free(ctx);

    return gf;
}

ggml_tensor* compute(test_model & model, ggml_gallocr_t allocr) {
    ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        std::exit(-1);
    }

    ggml_graph_print(gf);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

template<typename T>
void print_matrix(const T * matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << ' ';
        }
        std::cout << '\n';
    }
}

template<typename T>
void make_eye(T * matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (i==j);
        }
    }
}

template<typename T, typename U>
void gemm(int M, int N, int K,
          const U * A,
          const U * B,
          T * C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

int main()
{
#if USE_NPU
    using value_type = int16_t;
#else
    using value_type = float;
#endif

    const int M = 32, N = 32, K = 32;

    ggml_time_init();

    // matrix A
    value_type matrixA[M * K] = {};
    std::iota(matrixA, std::next(matrixA, M * K), 0);

    // matrix B
    value_type matrixB[K * N] = {};
    make_eye(matrixB, K, N);

    matrixB[100] = 10;

    // matrix C
    value_type matrixC_naive[M * N] = {};
    gemm(M, N, K, matrixA, matrixB, matrixC_naive);

    std::cout << "\nA[" << M << ',' << K << "]\n";
    print_matrix(matrixA, M, K);
    std::cout << "\nB[" << K << ',' << N << "]\n";
    print_matrix(matrixB, K, N);
    std::cout << "\nC[" << M << ',' << N << "]\n";
    print_matrix(matrixC_naive, M, N);

#if USE_NPU
    const bool use_npu = true;
#else
    const bool use_npu = false;
#endif

    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, use_npu);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    //create the worst case graph for memory usage estimation
    ggml_cgraph * gf = build_graph(model);

    // compute the required memory
    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
    fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);

    ggml_tensor * result = compute(model, allocr);

    std::vector<value_type> matrixC(ggml_nelements(result));
    ggml_backend_tensor_get(result, matrixC.data(), 0, ggml_nbytes(result));

    printf("\nPerforming ggml_mul_mat test:\n");

    bool passed = true;
    for(int i = 0; i < M * N; i++) {
        if(matrixC_naive[i] != matrixC[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_mul_mat (%d): %s\n", (int) ggml_nelements(result), passed && (ggml_nelements(result) == M * N) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    std::cout << "\nC (GGML)\n";
    print_matrix(matrixC.data(), M, N);

    // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
