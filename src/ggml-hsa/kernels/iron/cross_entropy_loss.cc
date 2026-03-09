// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include "aie_kernel_math.h"

extern "C" {

/**
 * @brief Computes cross-entropy loss using numerically stable log-softmax.
 *
 * Computes: loss = -sum(labels * log_softmax(logits))
 * where:   log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
 *
 * Three-pass algorithm for numerical stability:
 * 1. Find max(logits) to prevent overflow in exp().
 * 2. Compute sum_exp = sum(exp(logits - max)).
 * 3. Compute loss = -sum(labels * ((logits - max) - log(sum_exp))).
 *
 * @param[in]  logits   Input logits array of N elements (unnormalized scores).
 * @param[in]  labels   Target labels array of N elements (typically one-hot or probabilities).
 * @param[out] loss_out Single-element output array receiving the total loss.
 * @param[in]  N        Number of elements.
 */
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t N) {
    event0();

    constexpr int32_t VEC_SIZE = KERN_VEC_SIZE;
    const int32_t num_full_iters = N / VEC_SIZE;
    const int32_t tail_start = num_full_iters * VEC_SIZE;

    // ---------------------------------------------------------
    // Find max logit for numerical stability
    // ---------------------------------------------------------
    auto it_max_in = aie::cbegin_vector<VEC_SIZE>(logits);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int32_t i = 0; i < num_full_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_max_in++;
        v_max = aie::max(v_max, logit_vec);
    }

    float global_max = aie::reduce_max(v_max);

    // Scalar tail loop for remaining elements
    for (int32_t i = tail_start; i < N; i++) {
        if (logits[i] > global_max) global_max = logits[i];
    }

    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // ---------------------------------------------------------
    // Compute sum_exp = sum(exp(logits - max))
    // ---------------------------------------------------------
    auto it_logits = aie::cbegin_vector<VEC_SIZE>(logits);
    aie::accum<accfloat, VEC_SIZE> v_sum_exp_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int32_t i = 0; i < num_full_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_logits++;
        aie::vector<float, VEC_SIZE> x = aie::sub(logit_vec, v_global_max);
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);
        v_sum_exp_accum = aie::add(v_sum_exp_accum, exp_val);
    }

    aie::vector<float, VEC_SIZE> v_sum_exp = v_sum_exp_accum.to_vector<float>();
    float sum_exp = aie::reduce_add(v_sum_exp);

    // Scalar tail loop for remaining elements
    for (int32_t i = tail_start; i < N; i++) {
        sum_exp += scalar_exp(logits[i] - global_max);
    }

    // ---------------------------------------------------------
    // Compute log(sum_exp) using range-reduced scalar log
    // ---------------------------------------------------------
    float log_sum_exp = scalar_log(sum_exp);

    // ---------------------------------------------------------
    // Compute cross entropy loss in log-space
    //
    // log_softmax(x_i) = (x_i - max) - log(sum_exp)
    // loss = -sum( labels * log_softmax )
    // ---------------------------------------------------------
    it_logits = aie::cbegin_vector<VEC_SIZE>(logits);
    auto it_labels = aie::cbegin_vector<VEC_SIZE>(labels);

    aie::accum<accfloat, VEC_SIZE> v_loss_accum = aie::zeros<accfloat, VEC_SIZE>();
    aie::vector<float, VEC_SIZE> v_log_sum_exp = aie::broadcast<float, VEC_SIZE>(log_sum_exp);

    for (int32_t i = 0; i < num_full_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_logits++;
        aie::vector<float, VEC_SIZE> label_vec = *it_labels++;

        // log_softmax = (logits - max) - log(sum_exp)
        aie::vector<float, VEC_SIZE> log_softmax = aie::sub(logit_vec, v_global_max);
        log_softmax = aie::sub(log_softmax, v_log_sum_exp);

        // Accumulate: labels * log_softmax
        aie::accum<accfloat, VEC_SIZE> product = aie::mul(log_softmax, label_vec);
        v_loss_accum = aie::add(v_loss_accum, product.to_vector<float>());
    }

    // Reduce to scalar loss
    aie::vector<float, VEC_SIZE> v_loss = v_loss_accum.to_vector<float>();
    float total_loss = aie::reduce_add(v_loss);

    // Scalar tail loop for remaining elements
    for (int32_t i = tail_start; i < N; i++) {
        float log_softmax_i = (logits[i] - global_max) - log_sum_exp;
        total_loss += labels[i] * log_softmax_i;
    }

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;

    event1();
}

} // extern "C"
