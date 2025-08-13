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


struct ggml_tensor * ggml_mul_mat_i16(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b) {
    const int64_t ne[4] = { a->ne[0], b->ne[1], b->ne[2], b->ne[3] };
    //struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I16, 4, ne);
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}


int main(void) {
    // initialize data of matrices to perform matrix multiplication
    //const int rows_A = 4, cols_A = 2;
    //float matrix_A[rows_A * cols_A] = {
    //    2, 8,
    //    5, 1,
    //    4, 2,
    //    8, 6
    //};
    const int rows_A = 8, cols_A = 8;
    float matrix_A[rows_A * cols_A] = {
        2, 8, 6, 7, 3, 2, 1, 0,
        5, 1, 2, 4, 1, 0, 8, 6,
        4, 2, 34, 0, 1, 2, 8, 7, 
        8, 6, 34, 10, 2, 3, 87, 3, 
        2, 8, 6, 7, 3, 2, 1, 0,
        5, 1, 2, 4, 1, 0, 8, 6,
        4, 2, 34, 0, 1, 2, 8, 7, 
        8, 6, 34, 10, 2, 3, 87, 3
        //2, 8, 6, 7, 3, 2, 1, 0,
        //5, 1, 2, 4, 1, 0, 8, 6,
        //4, 2, 34, 0, 1, 2, 8, 7, 
        //8, 6, 34, 10, 2, 3, 87, 3, 
        //2, 8, 6, 7, 3, 2, 1, 0,
        //5, 1, 2, 4, 1, 0, 8, 6,
        //4, 2, 34, 0, 1, 2, 8, 7, 
        //8, 6, 34, 10, 2, 3, 87, 3
    }; 
    //const int rows_B = 3, cols_B = 2;
    //float matrix_B[rows_B * cols_B] = {
    //    10, 5,
    //    9, 9,
    //    5, 4
    //};

    const int rows_B = 8, cols_B = 8;
    float matrix_B[rows_B * cols_B] = {
        10, 5, 2, 4, 7, 18, 4, 10,
        9, 9, 7, 5, 9, 12, 3, 2,
        3, 4, 0, 1, 32, 78, 6, 23,
        4, 9, 1, 0, 3, 46, 29, 9,
        10, 5, 2, 4, 7, 18, 4, 10,
        9, 9, 7, 5, 9, 12, 3, 2,
        3, 4, 0, 1, 32, 78, 6, 23,
        4, 9, 1, 0, 3, 46, 29, 9
        //10, 5, 2, 4, 7, 18, 4, 10,
        //9, 9, 7, 5, 9, 12, 3, 2,
        //3, 4, 0, 1, 32, 78, 6, 23,
        //4, 9, 1, 0, 3, 46, 29, 9,
        //10, 5, 2, 4, 7, 18, 4, 10,
        //9, 9, 7, 5, 9, 12, 3, 2,
        //3, 4, 0, 1, 32, 78, 6, 23,
        //4, 9, 1, 0, 3, 46, 29, 9
    };

//    const int rows_B = 4, cols_B = 4;
//    float matrix_B[rows_B * cols_B] = {
//        10, 5, 2, 4,
//        9, 9, 7, 5,
//        3, 4, 0, 1,
//        4, 9, 1, 0
//    };

    printf("Beginning regular CPU GGML matrix-mul kernel\n");

    // 1. Allocate `ggml_context` to store tensor data
    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
    ctx_size += rows_A * rows_B * ggml_type_size(GGML_TYPE_F32); // result
    ctx_size += 3 * ggml_tensor_overhead(); // metadata for 3 tensors
    ctx_size += ggml_graph_overhead(); // compute graph
    ctx_size += 1024; // some overhead (exact calculation omitted for simplicity)

    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 2. Create tensors and set data
    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);
    memcpy(tensor_a->data, matrix_A, ggml_nbytes(tensor_a));
    memcpy(tensor_b->data, matrix_B, ggml_nbytes(tensor_b));


    // 3. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // result = a*b^T
    // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
    // the result is transposed
    struct ggml_tensor * result = ggml_mul_mat(ctx, tensor_a, tensor_b);

    // Mark the "result" tensor to be computed
    ggml_build_forward_expand(gf, result);

    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // 5. Retrieve results (output tensors)
    float * result_data = (float *) result->data;
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
    
    // 6. Free memory
    ggml_free(ctx);



/* STARTING CODE FOR HSA KERNEL */

#ifdef GGML_USA_HSA
    printf("\n\n Beginning HSA GGML mat-mul kernel\n");
#elif defined(GGML_USE_CUDA)
    printf("\n\n Beginning CUDA GGML mat-mul kernel\n");
#endif

    // 1. Initialize backend
    ggml_backend_t backend = NULL;

#ifdef GGML_USE_HSA
    fprintf(stderr, "%s: using HSA backend\n", __func__);
    backend = ggml_backend_hsa_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_hsa_init() failed\n", __func__);
    }
#endif 
    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    // Calculate the size needed to allocate
    ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors
    // no need to allocate anything else!

    // 2. Allocate `ggml_context` to store tensor data
    struct ggml_init_params params_hsa = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    ctx = ggml_init(params_hsa);

    // Create tensors metadata (only there shapes and data type)
    struct ggml_tensor * tensor_a_hsa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b_hsa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    // 4. Allocate a `ggml_backend_buffer` to store all tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5. Copy tensor data from main memory (RAM) to backend buffer
    ggml_backend_tensor_set(tensor_a_hsa, matrix_A, 0, ggml_nbytes(tensor_a_hsa));
    ggml_backend_tensor_set(tensor_b_hsa, matrix_B, 0, ggml_nbytes(tensor_b_hsa));

    // 6. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf_hsa = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        // create a temporally context to build the graph
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);
        gf_hsa = ggml_new_graph(ctx_cgraph);

	/* OLD CODE FROM HUGGINGFACE TUTORIAL FOR CUDA BACKEND */
        // result = a*b^T
        // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
        // the result is transposed
        //struct ggml_tensor * result0 = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);

        // zT = x @ yT
//#ifdef GGML_USE_HSA
        // HACK: we don't need to transpose
        ggml_tensor * result0 = ggml_mul_mat_i16(ctx_cgraph, tensor_a_hsa, tensor_b_hsa);
//#else
        //ggml_tensor * b_transposed = ggml_cont(ctx_cgraph, ggml_transpose(ctx_cgraph, tensor_b_hsa));
        //ggml_tensor * result0 = ggml_mul_mat_i16(ctx_cgraph, tensor_a_hsa, b_transposed);
	//ggml_tensor * c_transposed = ggml_mul_mat(ctx_cgraph, tensor_a_hsa, b_transposed);
        //ggml_tensor * result0 = ggml_cont(ctx_cgraph, ggml_transpose(ctx_cgraph, c_transposed));
	//struct ggml_tensor * result0 = ggml_mul_mat(ctx_cgraph, tensor_a_hsa, tensor_b_hsa);
//#endif

        // Add "result" tensor and all of its dependencies to the cgraph
        ggml_build_forward_expand(gf_hsa, result0);
    }

    // 7. Create a `ggml_gallocr` for cgraph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf_hsa);

    // (we skip step 8. Optionally: schedule the cgraph using `ggml_backend_sched`)

    // 9. Run the computation
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, gf_hsa);

    // 10. Retrieve results (output tensors)
    // in this example, output tensor is always the last tensor in the graph
    struct ggml_tensor * result_hsa = ggml_graph_node(gf_hsa, -1);
    float * result_data_hsa = (float*) malloc(ggml_nbytes(result_hsa));
    // because the tensor data is stored in device buffer, we need to copy it back to RAM
    ggml_backend_tensor_get(result_hsa, result_data_hsa, 0, ggml_nbytes(result_hsa));
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result_hsa->ne[0], (int) result_hsa->ne[1]);
    for (int j = 0; j < result_hsa->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result_hsa->ne[0] /* cols */; i++) {
            printf(" %.2f", result_data_hsa[j * result_hsa->ne[0] + i]);
        }
    }
    printf(" ]\n");
    free(result_data_hsa);

    // 11. Free memory and exit
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}

