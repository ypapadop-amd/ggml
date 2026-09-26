#include "ggml.h"
#include "ggml-opt.h"

#include "mnist-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Builds the MNIST fully-connected forward pass in bf16 instead of f32.
//
// The stock mnist_model_build (mnist-common.cpp) keeps everything f32, so the HSA backend converts
// each MUL_MAT operand f32->bf16 and de-pads the result back to f32 between layers, round-tripping
// through f32 around every GEMM. Here the activations and weights are cast to bf16 in the graph so
// the intermediate ADD/RELU run in bf16 and the MUL_MAT operands arrive already bf16 (no per-layer
// input conversion). ggml always produces an f32 MUL_MAT result, so a cast back to bf16 is still
// needed before the second GEMM, and the final logits are cast to f32 (ggml_opt expects f32).
//
// Weights/biases are constant, but mnist_model_eval() re-executes the *entire* built graph once per
// physical batch (ggml_opt_epoch -> ggml_opt_eval -> ggml_backend_sched_graph_compute), so a
// ggml_cast node on a weight sitting in that graph gets redundantly recomputed every batch. Cast the
// weights/biases to bf16 exactly once here, into their own statically-allocated tensors, and build
// the eval graph directly against those instead of inserting cast nodes for them into ctx_compute.
static void mnist_model_build_bf16(mnist_model & model) {
    GGML_ASSERT(model.arch == "mnist-fc" && "bf16 eval only supports the fully-connected model");

    ggml_set_param(model.fc1_weight);
    ggml_set_param(model.fc1_bias);
    ggml_set_param(model.fc2_weight);
    ggml_set_param(model.fc2_bias);

    ggml_context * ctx = model.ctx_compute;

    // One-time precast: cast weights/biases to bf16 and copy the result into statically-allocated
    // bf16 tensors that live for the rest of the process (mirrors the ctx_static/buf_static pattern
    // mnist_model already uses for the f32 weights/biases).
    ggml_tensor * fc1_w_bf16;
    ggml_tensor * fc1_b_bf16;
    ggml_tensor * fc2_w_bf16;
    ggml_tensor * fc2_b_bf16;
    {
        struct ggml_init_params params_bf16 = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_bf16 = ggml_init(params_bf16); // intentionally never freed, tensors must outlive this function

        fc1_w_bf16 = ggml_new_tensor(ctx_bf16, GGML_TYPE_BF16, GGML_MAX_DIMS, model.fc1_weight->ne);
        fc1_b_bf16 = ggml_new_tensor(ctx_bf16, GGML_TYPE_BF16, GGML_MAX_DIMS, model.fc1_bias->ne);
        fc2_w_bf16 = ggml_new_tensor(ctx_bf16, GGML_TYPE_BF16, GGML_MAX_DIMS, model.fc2_weight->ne);
        fc2_b_bf16 = ggml_new_tensor(ctx_bf16, GGML_TYPE_BF16, GGML_MAX_DIMS, model.fc2_bias->ne);
        ggml_backend_alloc_ctx_tensors(ctx_bf16, model.backends[0]);

        struct ggml_init_params params_tmp = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead() + ggml_graph_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx_tmp = ggml_init(params_tmp);

        ggml_tensor * cast1 = ggml_cast(ctx_tmp, model.fc1_weight, GGML_TYPE_BF16);
        ggml_tensor * cast2 = ggml_cast(ctx_tmp, model.fc1_bias, GGML_TYPE_BF16);
        ggml_tensor * cast3 = ggml_cast(ctx_tmp, model.fc2_weight, GGML_TYPE_BF16);
        ggml_tensor * cast4 = ggml_cast(ctx_tmp, model.fc2_bias, GGML_TYPE_BF16);

        // Without this, the graph allocator sees no consumer for these nodes and is free to
        // recycle their memory for a later node in the same graph before we copy the results out.
        ggml_set_output(cast1);
        ggml_set_output(cast2);
        ggml_set_output(cast3);
        ggml_set_output(cast4);

        ggml_cgraph * gf = ggml_new_graph(ctx_tmp);
        ggml_build_forward_expand(gf, cast1);
        ggml_build_forward_expand(gf, cast2);
        ggml_build_forward_expand(gf, cast3);
        ggml_build_forward_expand(gf, cast4);

        ggml_backend_sched_reset(model.backend_sched);
        ggml_backend_sched_alloc_graph(model.backend_sched, gf);
        ggml_backend_sched_graph_compute(model.backend_sched, gf);

        ggml_backend_tensor_copy(cast1, fc1_w_bf16);
        ggml_backend_tensor_copy(cast2, fc1_b_bf16);
        ggml_backend_tensor_copy(cast3, fc2_w_bf16);
        ggml_backend_tensor_copy(cast4, fc2_b_bf16);

        ggml_backend_sched_reset(model.backend_sched);
        ggml_free(ctx_tmp);
    }

    // Layer 1: fc1 = relu(fc1_w @ images + fc1_bias), kept in bf16.
    // ggml_mul_mat yields f32; cast the result back to bf16 so ADD/RELU run in bf16.
    ggml_tensor * mm1  = ggml_cast(ctx, ggml_mul_mat(ctx, fc1_w_bf16, model.images), GGML_TYPE_BF16);
    ggml_tensor * fc1  = ggml_relu(ctx, ggml_add(ctx, mm1, fc1_b_bf16));

    // Layer 2: logits = fc2_w @ fc1 + fc2_bias.
    ggml_tensor * mm2    = ggml_cast(ctx, ggml_mul_mat(ctx, fc2_w_bf16, fc1), GGML_TYPE_BF16);
    ggml_tensor * logits_bf16 = ggml_add(ctx, mm2, fc2_b_bf16);

    // ggml_opt's loss/accuracy expects f32 logits.
    model.logits = ggml_cast(ctx, logits_bf16, GGML_TYPE_F32);

    ggml_set_name(model.logits, "logits");
    ggml_set_output(model.logits);
    GGML_ASSERT(model.logits->type == GGML_TYPE_F32);
    GGML_ASSERT(model.logits->ne[0] == MNIST_NCLASSES);
    GGML_ASSERT(model.logits->ne[1] == model.nbatch_physical);
    GGML_ASSERT(model.logits->ne[2] == 1);
    GGML_ASSERT(model.logits->ne[3] == 1);
}

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        exit(1);
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(GGML_TYPE_F32, GGML_TYPE_F32, MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTEST, MNIST_NBATCH_PHYSICAL);

    if (!mnist_image_load(argv[2], dataset)) {
        return 1;
    }
    if (!mnist_label_load(argv[3], dataset)) {
        return 1;
    }

    const int iex = rand() % MNIST_NTEST;
    mnist_image_print(stdout, dataset, iex);

    const std::string backend = argc >= 5 ? argv[4] : "";

    const int64_t t_start_us = ggml_time_us();
    mnist_model model = mnist_model_init_from_file(argv[1], backend, MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);
    mnist_model_build_bf16(model);
    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: loaded model in %.2lf ms\n", __func__, t_load_us / 1000.0);

    ggml_opt_result_t result_eval = mnist_model_eval(model, dataset);

    std::vector<int32_t> pred(MNIST_NTEST);
    ggml_opt_result_pred(result_eval, pred.data());
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, pred[iex]);

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result_eval, &loss, &loss_unc);
    fprintf(stdout, "%s: test_loss=%.6lf+-%.6lf\n", __func__, loss, loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result_eval, &accuracy, &accuracy_unc);
    fprintf(stdout, "%s: test_acc=%.2lf+-%.2lf%%\n", __func__, 100.0*accuracy, 100.0*accuracy_unc);

    ggml_opt_result_free(result_eval);

    return 0;
}
