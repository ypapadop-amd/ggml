// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 16
#endif

// =============================================================================
// Scalar natural logarithm using IEEE 754 range reduction + atanh series
// =============================================================================
inline float scalar_log(float x) {
    if (x <= 0.0f)
        return -88.0f;

    int32_t bits = reinterpret_cast<int32_t &>(x);
    int32_t e_int = ((bits >> 23) & 0xFF) - 127;
    float e = static_cast<float>(e_int);

    int32_t m_bits = (bits & 0x007FFFFF) | 0x3F800000;
    float m = reinterpret_cast<float &>(m_bits);

    float z = (m - 1.0f) / (m + 1.0f);
    float z2 = z * z;

    float poly = 0.0909090909f;       // 1/11
    poly = poly * z2 + 0.1111111111f; // 1/9
    poly = poly * z2 + 0.1428571429f; // 1/7
    poly = poly * z2 + 0.2000000000f; // 1/5
    poly = poly * z2 + 0.3333333333f; // 1/3
    poly = poly * z2 + 1.0f;

    float ln_m = 2.0f * (z * poly);

    constexpr float ln2 = 0.6931471805599453f;
    return ln_m + (e * ln2);
}

extern "C" {
#ifdef COMPILE_GGML_OP_CROSS_ENTROPY_LOSS

// =============================================================================
// Cross entropy loss kernel
//
// Computes: loss = -sum(labels * log_softmax(logits))
// where    log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
//
// Three-pass algorithm:
//   1: Find max(logits) for numerical stability
//   2: Compute sum_exp = sum(exp(logits - max))
//   3: Compute loss = -sum(labels * ((logits - max) - log(sum_exp)))
// =============================================================================
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t N) {
    event0();

    constexpr int VEC_SIZE = KERN_VEC_SIZE;
    const int num_iters = N / VEC_SIZE;

    // ---------------------------------------------------------
    // Find max logit for numerical stability
    // ---------------------------------------------------------
    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_max_in++;
        v_max = aie::max(v_max, logit_vec);
    }

    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // ---------------------------------------------------------
    // Compute sum_exp = sum(exp(logits - max))
    // ---------------------------------------------------------
    auto it_logits = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    aie::accum<accfloat, VEC_SIZE> v_sum_exp_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_logits++;
        aie::vector<float, VEC_SIZE> x = aie::sub(logit_vec, v_global_max);
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);
        v_sum_exp_accum = aie::add(v_sum_exp_accum, exp_val);
    }

    aie::vector<float, VEC_SIZE> v_sum_exp = v_sum_exp_accum.to_vector<float>();
    float sum_exp = aie::reduce_add(v_sum_exp);

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
    it_logits = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    auto it_labels = aie::cbegin_vector<VEC_SIZE>((float *)labels);

    aie::accum<accfloat, VEC_SIZE> v_loss_accum = aie::zeros<accfloat, VEC_SIZE>();
    aie::vector<float, VEC_SIZE> v_log_sum_exp = aie::broadcast<float, VEC_SIZE>(log_sum_exp);

    for (int i = 0; i < num_iters; i++) {
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

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;

    event1();
}

#endif // COMPILE_GGML_OP_CROSS_ENTROPY_LOSS

} // extern "C"
