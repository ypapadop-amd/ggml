// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

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
 * @param[in]  N        Number of elements (arbitrary length).
 */
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t N) {
    event0();

    // Pass 1: Find max logit for numerical stability
    auto global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; i++) {
        if (logits[i] > global_max) {
            global_max = logits[i];
        }
    }

    // Pass 2: Compute sum_exp = sum(exp(logits - max))
    float sum_exp = 0.0f;
    for (int32_t i = 0; i < N; i++) {
        float x = logits[i] - global_max;
        sum_exp += scalar_exp(x);
    }

    // Compute log(sum_exp) using range-reduced scalar log
    const auto log_sum_exp = scalar_log(sum_exp);

    // Pass 3: Compute cross entropy loss in log-space
    // log_softmax(x_i) = (x_i - max) - log(sum_exp)
    // loss = -sum( labels * log_softmax )
    float total_loss = 0.0f;
    for (int32_t i = 0; i < N; i++) {
        float log_softmax = (logits[i] - global_max) - log_sum_exp;
        total_loss += labels[i] * log_softmax;
    }

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;

    event1();
}

} // extern "C"
