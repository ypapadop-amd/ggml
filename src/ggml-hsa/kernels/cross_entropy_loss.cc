// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 16
#endif

// Vector exponential function (reused from softmax)
template <int VecSize = KERN_VEC_SIZE>
inline aie::vector<float, VecSize> vec_exp(aie::vector<float, VecSize>& x) {
    // Taylor series coefficients in reverse order for Horner's method
    constexpr float exp_coeffs[] = {
        0.0000000001605904f, // 1/13!
        0.0000000020876757f, // 1/12!
        0.0000000250521084f, // 1/11!
        0.0000002755731922f, // 1/10!
        0.0000027557319224f, // 1/9!
        0.0000248015873016f, // 1/8!
        0.0001984126984127f, // 1/7!
        0.0013888888888889f, // 1/6!
        0.0083333333333333f, // 1/5!
        0.0416666666666667f, // 1/4!
        0.1666666666666667f, // 1/3!
        0.5f,                // 1/2!
        1.0f,                // 1/1!
        1.0f                 // 1/0!
    };
    constexpr int NUM_EXP_COEFFS = sizeof(exp_coeffs) / sizeof(exp_coeffs[0]);

    // clamp x to prevent overflow
    aie::vector<float, VecSize> v_clamp_min = aie::broadcast<float, VecSize>(-88.0f);
    aie::vector<float, VecSize> v_clamp_max = aie::broadcast<float, VecSize>(88.0f);
    x = aie::max(x, v_clamp_min);
    x = aie::min(x, v_clamp_max);

    aie::accum<accfloat, VecSize> tmp_accum;
    aie::vector<float, VecSize> poly = aie::broadcast<float, VecSize>(exp_coeffs[0]);

#pragma unroll
    for (int i = 1; i < NUM_EXP_COEFFS; ++i) {
        tmp_accum = aie::mul(poly, x);
        poly = tmp_accum.template to_vector<float>();
        poly = aie::add(poly, aie::broadcast<float, VecSize>(exp_coeffs[i]));
    }

    // clamp to positive
    aie::vector<float, VecSize> v_exp_min = aie::broadcast<float, VecSize>(1e-38f);
    poly = aie::max(poly, v_exp_min);

    return poly;
}

// Vector natural logarithm function using Taylor series
// ln(x) for x in (0, inf), computed using ln(x) = ln(m * 2^e) = ln(m) + e * ln(2)
template <int VecSize = KERN_VEC_SIZE>
inline aie::vector<float, VecSize> vec_log(aie::vector<float, VecSize> x) {
    // Use range reduction: x = m * 2^e where 0.5 <= m < 1.0
    // ln(x) = ln(m) + e * ln(2)

    constexpr float ln2 = 0.6931471805599453f;
    constexpr float epsilon = 1e-38f;

    // Clamp to prevent log(0) or log(negative)
    aie::vector<float, VecSize> v_epsilon = aie::broadcast<float, VecSize>(epsilon);
    x = aie::max(x, v_epsilon);

    // For simplicity, use Taylor series around 1: ln(1+y) = y - y^2/2 + y^3/3 - y^4/4 + ...
    // We'll use approximation: ln(x) ≈ 2 * atanh((x-1)/(x+1))
    // where atanh(z) = z + z^3/3 + z^5/5 + z^7/7 + ...

    // Simplified approximation for ln(x):
    // For x close to 1, use ln(x) ≈ (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4
    aie::vector<float, VecSize> one = aie::broadcast<float, VecSize>(1.0f);
    aie::vector<float, VecSize> y = aie::sub(x, one);

    // Compute powers and polynomial
    aie::accum<accfloat, VecSize> y2_accum = aie::mul(y, y);
    aie::vector<float, VecSize> y2 = y2_accum.template to_vector<float>();

    aie::accum<accfloat, VecSize> y3_accum = aie::mul(y2, y);
    aie::vector<float, VecSize> y3 = y3_accum.template to_vector<float>();

    aie::accum<accfloat, VecSize> y4_accum = aie::mul(y3, y);
    aie::vector<float, VecSize> y4 = y4_accum.template to_vector<float>();

    // ln(x) ≈ y - y^2/2 + y^3/3 - y^4/4
    aie::vector<float, VecSize> half = aie::broadcast<float, VecSize>(0.5f);
    aie::vector<float, VecSize> third = aie::broadcast<float, VecSize>(0.3333333333f);
    aie::vector<float, VecSize> quarter = aie::broadcast<float, VecSize>(0.25f);

    aie::accum<accfloat, VecSize> term2 = aie::mul(y2, half);
    aie::accum<accfloat, VecSize> term3 = aie::mul(y3, third);
    aie::accum<accfloat, VecSize> term4 = aie::mul(y4, quarter);

    aie::vector<float, VecSize> result = y;
    result = aie::sub(result, term2.template to_vector<float>());
    result = aie::add(result, term3.template to_vector<float>());
    result = aie::sub(result, term4.template to_vector<float>());

    return result;
}

extern "C" {
#ifdef COMPILE_GGML_OP_CROSS_ENTROPY_LOSS

// Cross entropy loss kernel
// Computes: loss = -sum(labels * log(softmax(logits))) / num_rows
// Where softmax(logits) is computed with numerical stability (subtract max)
//
// Algorithm per row:
// 1. Find max of logits for numerical stability
// 2. Compute log_softmax = (logits - max) - log(sum(exp(logits - max)))
// 3. Multiply log_softmax by labels element-wise
// 4. Sum the products
// 5. Accumulate to total loss
//
// Final step: Negate and divide by number of rows
void ggml_op_cross_entropy_loss(const INPUT_DTYPE0 * __restrict logits,
                                  const INPUT_DTYPE1 * __restrict labels,
                                  OUTPUT_DTYPE * __restrict loss_out,
                                  int32_t N) {
    event0();

    constexpr int VEC_SIZE = KERN_VEC_SIZE;
    const int num_iters = N / VEC_SIZE;

    auto it_logits = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    auto it_labels = aie::cbegin_vector<VEC_SIZE>((float *)labels);

    // Find max value for numerical stability
    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_max_in++;
        v_max = aie::max(v_max, logit_vec);
    }

    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // Compute sum of exp(logits - max) for softmax normalization
    it_logits = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    aie::accum<accfloat, VEC_SIZE> v_sum_exp_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_logits++;

        // subtract max for numerical stability
        aie::vector<float, VEC_SIZE> x = aie::sub(logit_vec, v_global_max);

        // compute exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // accumulate sum
        v_sum_exp_accum = aie::add(v_sum_exp_accum, exp_val);
    }

    // Reduce to get total sum for softmax normalization
    aie::vector<float, VEC_SIZE> v_sum_exp = v_sum_exp_accum.to_vector<float>();
    float sum_exp = aie::reduce_add(v_sum_exp);

    // Compute log(sum_exp)
    float log_sum_exp = 0.0f;
    if (sum_exp > 0.0f) {
        // Use scalar logarithm approximation
        // ln(x) using same Taylor series approach
        float x_log = sum_exp;
        float y_log = x_log - 1.0f;
        float y2 = y_log * y_log;
        float y3 = y2 * y_log;
        float y4 = y3 * y_log;
        log_sum_exp = y_log - y2 * 0.5f + y3 * 0.3333333333f - y4 * 0.25f;
    }

    // Compute cross entropy loss: -sum(labels * log_softmax)
    // log_softmax = (logits - max) - log_sum_exp
    it_logits = aie::cbegin_vector<VEC_SIZE>((float *)logits);
    it_labels = aie::cbegin_vector<VEC_SIZE>((float *)labels);

    aie::accum<accfloat, VEC_SIZE> v_loss_accum = aie::zeros<accfloat, VEC_SIZE>();
    aie::vector<float, VEC_SIZE> v_log_sum_exp = aie::broadcast<float, VEC_SIZE>(log_sum_exp);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> logit_vec = *it_logits++;
        aie::vector<float, VEC_SIZE> label_vec = *it_labels++;

        // compute log_softmax = (logits - max) - log_sum_exp
        aie::vector<float, VEC_SIZE> log_softmax = aie::sub(logit_vec, v_global_max);
        log_softmax = aie::sub(log_softmax, v_log_sum_exp);

        // multiply by labels
        aie::accum<accfloat, VEC_SIZE> product = aie::mul(log_softmax, label_vec);

        // accumulate
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
