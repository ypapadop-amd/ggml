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
#include <chrono>
#include <typeinfo>

#ifdef GGML_USE_HSA
#define USE_NPU 1
#endif

template<typename>
struct fundamental_to_ggml_type;

template<>
struct fundamental_to_ggml_type<int8_t> {
    inline static const auto ggml_type = GGML_TYPE_I8;
};

template<>
struct fundamental_to_ggml_type<int16_t> {
    inline static const auto ggml_type = GGML_TYPE_I16;
};

template<>
struct fundamental_to_ggml_type<float> {
    inline static const auto ggml_type = GGML_TYPE_F32;
};

struct ggml_tensor * ggml_mul_mat_i8(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b) {
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I8, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct ggml_tensor * ggml_mul_mat_i16(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b) {
    GGML_ASSERT(!ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
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
void load_model(test_model & model, T * a, T * b, int M, int N, int K, bool use_gpu = true) {
    const auto ggml_type = fundamental_to_ggml_type<T>::ggml_type;

    size_t buffer_size = 0;
    buffer_size += (M * K) * ggml_type_size(ggml_type); // tensor a
    buffer_size += (N * K) * ggml_type_size(ggml_type); // tensor b
    // buffer_size += (M * N)*2 * ggml_type_size(ggml_type); // tensor out
    // buffer_size += 1024; // overhead

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
    printf("Matrix A: [%i, %i]\n", K, M);
    model.b = ggml_new_tensor_2d(model.ctx, ggml_type, K, N);
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

    // create a temporally context to build the graph
    ggml_context * ctx = ggml_init(params0);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    int num_runs_per_graph = 20;
    // Vector to store the output tensors
    std::vector<ggml_tensor*> outputs(num_runs_per_graph);

#if USE_NPU
    // Perform the first multiplication
    outputs[0] = ggml_mul_mat_i8(ctx, model.a, model.b);

    // Perform 99 additional multiplications, each writing to a new tensor
    for (int i = 1; i < num_runs_per_graph; ++i) {
        outputs[i] = ggml_mul_mat_i8(ctx, outputs[i - 1], model.b);
    }
#else
    // Perform the first multiplication
    outputs[0] = ggml_mul_mat(ctx, model.a, model.b);

    // Perform 99 additional multiplications, each writing to a new tensor
    for (int i = 1; i < num_runs_per_graph; ++i) {
        outputs[i] = ggml_mul_mat(ctx, outputs[i - 1], model.b);
    }
#endif

    // Add all tensors to the graph
    for (auto& tensor : outputs) {
        ggml_build_forward_expand(gf, tensor);
    }

    // Optionally dump the graph for debugging
    ggml_graph_dump_dot(gf, NULL, "debug.dot");
    // delete the temporally context used to build the graph
    ggml_free(ctx);
    // exit(-1);

    return gf;
}

ggml_tensor* compute(test_model & model, ggml_cgraph * gf, ggml_gallocr_t allocr) {

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    // ggml_graph_print(gf);

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
                C[m * K + k] += A[m * N + n] * B[n * K + k];
            }
        }
    }
}

int main()
{
    // if (!std::cout) {
    //     std::cerr << "std::cout is in a bad state!" << std::endl;
    // }
    // std::cout << "Through here" << std::endl;
#if USE_NPU
    using value_type = int8_t;
    const bool use_npu = true;
#else
    using value_type = float;
    // const bool use_gpu = true;
#endif

    // const int M = 6144, N = 1024, K = 1536;  // a conv2d expected matrix multiplication
    const int M = 1024, N = 1024, K = 1024;  // a conv2d expected matrix multiplication
 //MxN-KxM-KxN
    ggml_time_init();

    // matrix A
    value_type matrixA[M * K] = {};
    std::iota(matrixA, std::next(matrixA, M*K), 0);

    // matrix B
    value_type matrixB[N * K] = {};
    make_eye(matrixB, N, K);

    matrixB[N*K -1] = 10;

    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, use_npu);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    //create the worst case graph for memory usage estimation
    ggml_cgraph * gf = build_graph(model);

    // compute the required memory
    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
    // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    int num_runs_of_graphs = 10;
    std::vector<value_type> matrixC(ggml_nelements(model.a) * num_runs_of_graphs); // Allocate space for results

    std::string kernel_name = "mul_mat_aie2_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + typeid(matrixA[0]).name();
    std::cout << "Starting execution" << std::endl;

    std::chrono::duration<double, std::milli> total_elapsed_time(0);
    for (int i = 0; i < num_runs_of_graphs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ggml_tensor *result = compute(model, gf, allocr);
        
        // Retrieve the result of the current GEMM operation
        ggml_backend_tensor_get(result, matrixC.data() + i * ggml_nelements(result), 0, ggml_nbytes(result));
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
        total_elapsed_time += elapsed_time;
        
        std::cout << kernel_name << ",Execution time," << elapsed_time.count() << ",milliseconds" << std::endl;
    }

    std::cout << std::endl;
    std::cout << kernel_name << ",Average Execution time," << total_elapsed_time.count() / (float) num_runs_of_graphs << ",milliseconds" << std::endl;


    // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
