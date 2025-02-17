#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "ggml-hsa.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define N_THREADS 1

// Used to compare floats
#define EPSILON 0.0001

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {

    // Parallel vector_add
    struct ggml_tensor * a0;
    struct ggml_tensor * b0;

    // Parallel vector_add
    struct ggml_tensor * a1;
    struct ggml_tensor * b1;

    // the backends used in this compute
    std::vector<ggml_backend_t> backends;

    // Scheduler
    ggml_backend_sched_t sched;

    // the backend buffer to storage 
    ggml_backend_buffer_t cpu_buffer;
    ggml_backend_buffer_t gpu_buffer;
    ggml_backend_buffer_t hsa_buffer;

    // Allocators
    struct ggml_tallocr cpu_alloc;
    struct ggml_tallocr gpu_alloc;
    struct ggml_tallocr hsa_alloc;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a0, float * b0, float *a1, float*b1, int N) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_t gpu_backend;
    ggml_backend_t hsa_backend;
    ggml_backend_t cpu_backend;

    // Create a GPU backend
    gpu_backend = ggml_backend_cuda_init(0); // init device 0
    if (!gpu_backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }

    // Create our NPU backend
    hsa_backend = ggml_backend_hsa_init(0);

    // Create a CPU backend 
    cpu_backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu_backend, N_THREADS);

    // Pushing back all backends so it can be used by the scheduler
    model.backends.push_back(gpu_backend);
    model.backends.push_back(hsa_backend);
    model.backends.push_back(cpu_backend);
    
    // We are doing two GEMMs
    int num_tensors = 4;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a0 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, N);
    model.b0 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, N);
    model.a1 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, N);
    model.b1 = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, N);

    // Want to put A0 + B0 on GPU and A1 + B1 on NPU
    model.cpu_buffer = ggml_backend_alloc_buffer(cpu_backend, 8192);
    model.gpu_buffer = ggml_backend_alloc_buffer(gpu_backend, 8192);
    model.hsa_buffer = ggml_backend_alloc_buffer(hsa_backend, 8192);
    model.cpu_alloc = ggml_tallocr_new(model.cpu_buffer);
    model.gpu_alloc = ggml_tallocr_new(model.gpu_buffer);
    model.hsa_alloc = ggml_tallocr_new(model.hsa_buffer);
    ggml_tallocr_alloc(&model.gpu_alloc, model.a0);
    ggml_tallocr_alloc(&model.gpu_alloc, model.b0);
    ggml_tallocr_alloc(&model.hsa_alloc, model.a1);
    ggml_tallocr_alloc(&model.hsa_alloc, model.b1);
        
    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a0, a0, 0, ggml_nbytes(model.a0));
    ggml_backend_tensor_set(model.b0, b0, 0, ggml_nbytes(model.b0));
    ggml_backend_tensor_set(model.a1, a1, 0, ggml_nbytes(model.a1));
    ggml_backend_tensor_set(model.b1, b1, 0, ggml_nbytes(model.b1));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result = a + b
    struct ggml_tensor * result0 = ggml_add(ctx0, model.a0, model.b0);
    struct ggml_tensor * result1 = ggml_add(ctx0, model.a1, model.b1);
    struct ggml_tensor * result2 = ggml_add(ctx0, result0, result1);

    // build operations nodes
    ggml_build_forward_expand(gf, result2);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor *compute(simple_model & model) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    ggml_backend_sched_reset(model.sched);
    ggml_backend_sched_graph_compute(model.sched, gf);
    
    return ggml_graph_node(gf, -1);

}

int main(void) {
    ggml_time_init();

    // size of tensor
    int N = 256;

    // Initializing data for first vector add
    float *A0 = (float *)malloc(sizeof(float) * N);
    float *B0 = (float *)malloc(sizeof(float) * N);
    for(int i = 0; i < N; i++) {
      A0[i] = i;
      B0[i] = i + 10;
    }

    // Initializing data for the second vector add
    float *A1 = (float *)malloc(sizeof(float) * N);
    float *B1 = (float *)malloc(sizeof(float) * N);
    for(int i = 0; i < N; i++) {
      A1[i] = i + 1;
      B1[i] = i + 10;
    }

    simple_model model;
    load_model(model, A0, B0, A1, B1, N);

    model.sched = ggml_backend_sched_new(model.backends.data(), NULL, model.backends.size(), 6 /* graph size */, false);
    struct ggml_cgraph * gf = build_graph(model);

    // Allocating for the scheduler
    ggml_backend_sched_reserve(model.sched, gf);

    // perform computation
    struct ggml_tensor *output = compute(model);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(output));

    // bring the data from the backend memory
    ggml_backend_tensor_get(output, out_data.data(), 0, ggml_nbytes(output));

    // This is used to print a generic matrix
    int errors = 0;
    for (int i = 0; i < N; i++) {
      printf("%.0f ", out_data[i]);
      if ((out_data[i] - (A0[i] + A1[i] + B0[i] + B1[i]) ) > EPSILON) {
        printf("Expected is %f\n", A0[i] + A1[i] + B0[i] + B1[i]);
        printf("Result is %f\n", out_data[i]);
        errors++;
      }
    }
    printf("\n\n");

    if (!errors)
      printf("PASS!\n");
    else
      printf("FAIL\n");

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.gpu_buffer);
    ggml_backend_buffer_free(model.cpu_buffer);
    return 0;
}
