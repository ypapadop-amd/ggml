// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 16
#endif

// =============================================================================
// Range-reduced vectorized exponential: exp(x) = 2^n * exp(r)
//
// Cody-Waite splitting of ln(2) into high and low parts minimises
// rounding error in step 2. A degree-9 polynomial on [0, ln2) gives
// ~6e-9 peak error, well within float32's ~1.2e-7 epsilon.
// =============================================================================
template <int VecSize = KERN_VEC_SIZE>
inline aie::vector<float, VecSize> vec_exp(aie::vector<float, VecSize> & x) {
    constexpr float log2e = 1.4426950408889634f; // log2(e)
    // Cody-Waite split of ln(2) for accurate range reduction
    constexpr float ln2_hi = 6.93145751953125e-1f;      // upper bits, exact in float
    constexpr float ln2_lo = 1.4286068203094172321e-6f; // residual

    // Clamp to representable range of exp() in float32
    x = aie::max(x, aie::broadcast<float, VecSize>(-88.0f));
    x = aie::min(x, aie::broadcast<float, VecSize>(88.0f));

    // Compute t = x * log2(e)
    aie::accum<accfloat, VecSize> t_acc = aie::mul(x, log2e);
    aie::vector<float, VecSize> t = t_acc.template to_vector<float>();

    // n = floor(t) via magic-number rounding
    constexpr float magic = 12582912.0f; // 1.5 * 2^23
    aie::vector<float, VecSize> v_magic = aie::broadcast<float, VecSize>(magic);
    aie::vector<float, VecSize> n_f = aie::sub(aie::add(t, v_magic), v_magic);

    // Adjust round-to-nearest -> floor: if n_f > t, we rounded up
    auto overshot = aie::lt(t, n_f);
    n_f = aie::sub(n_f, aie::select(aie::broadcast<float, VecSize>(0.0f),
                                    aie::broadcast<float, VecSize>(1.0f), overshot));

    // r = x - n * ln(2) with Cody-Waite precision
    aie::accum<accfloat, VecSize> hi_acc = aie::mul(n_f, ln2_hi);
    aie::vector<float, VecSize> r = aie::sub(x, hi_acc.template to_vector<float>());
    aie::accum<accfloat, VecSize> lo_acc = aie::mul(n_f, ln2_lo);
    r = aie::sub(r, lo_acc.template to_vector<float>());

    // Evaluate exp(r) using Horner's method (degree 13)
    //
    // exp(r) ~ 1 + r + r^2/2! + r^3/3! + ... + r^13/13!
    // Coefficients in high-to-low order for Horner evaluation:
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

    aie::vector<float, VecSize> poly = aie::broadcast<float, VecSize>(exp_coeffs[0]);
    aie::accum<accfloat, VecSize> tmp;

#pragma unroll
    for (int i = 1; i < NUM_EXP_COEFFS; ++i) {
        tmp = aie::mul(poly, r);
        poly = aie::add(tmp.template to_vector<float>(),
                        aie::broadcast<float, VecSize>(exp_coeffs[i]));
    }

    // Compute 2^n via IEEE 754 bit construction
    n_f = aie::max(n_f, aie::broadcast<float, VecSize>(-126.0f));
    n_f = aie::min(n_f, aie::broadcast<float, VecSize>(127.0f));

    // aie::to_fixed<int32_t>(n_f, 23) = floor(n_f * 2^23) = n << 23
    auto n_shifted = aie::to_fixed<int32_t>(n_f, 23);
    // (n + 127) << 23 = (n << 23) + (127 << 23)
    auto scale_bits = aie::add(n_shifted, aie::broadcast<int32_t, VecSize>(0x3F800000));
    aie::vector<float, VecSize> scale = scale_bits.template cast_to<float>();

    // Reconstruct exp(x) = exp(r) * 2^n
    aie::accum<accfloat, VecSize> result_acc = aie::mul(poly, scale);
    aie::vector<float, VecSize> result = result_acc.template to_vector<float>();

    // Clamp to positive minimum (avoid exact zero from underflow)
    result = aie::max(result, aie::broadcast<float, VecSize>(1e-38f));

    return result;
}

// =============================================================================
// Vector natural logarithm using IEEE 754 range reduction + atanh series
// =============================================================================
template <int VecSize = KERN_VEC_SIZE>
inline aie::vector<float, VecSize> vec_log(aie::vector<float, VecSize> x) {
    constexpr float ln2 = 0.6931471805599453f;
    constexpr float epsilon = 1e-38f;

    aie::vector<float, VecSize> v_epsilon = aie::broadcast<float, VecSize>(epsilon);
    x = aie::max(x, v_epsilon);

    // 1. IEEE 754 Range Reduction
    auto v_bits = x.template cast_to<int32_t>();

    auto v_shifted = v_bits >> 23;
    auto v_e_int = aie::sub(aie::bit_and(v_shifted, aie::broadcast<int32_t, VecSize>(0xFF)),
                            aie::broadcast<int32_t, VecSize>(127));
    aie::vector<float, VecSize> v_e = aie::to_float(v_e_int);

    auto v_m_bits = aie::bit_or(aie::bit_and(v_bits, aie::broadcast<int32_t, VecSize>(0x007FFFFF)),
                                aie::broadcast<int32_t, VecSize>(0x3F800000));
    aie::vector<float, VecSize> v_m = v_m_bits.template cast_to<float>();

    // 2. atanh substitution: z = (m - 1) / (m + 1)
    aie::vector<float, VecSize> v_one = aie::broadcast<float, VecSize>(1.0f);
    aie::vector<float, VecSize> num = aie::sub(v_m, v_one);
    aie::vector<float, VecSize> den = aie::add(v_m, v_one);
    aie::vector<float, VecSize> z = aie::mul(num, aie::inv(den));

    aie::accum<accfloat, VecSize> z2_accum = aie::mul(z, z);
    aie::vector<float, VecSize> z2 = z2_accum.template to_vector<float>();

    // 3. Polynomial: 1 + z^2/3 + z^4/5 + z^6/7 + z^8/9 + z^10/11
    aie::vector<float, VecSize> poly = aie::broadcast<float, VecSize>(0.0909090909f); // 1/11

    aie::accum<accfloat, VecSize> acc = aie::mul(poly, z2);
    poly = aie::add(acc.template to_vector<float>(),
                    aie::broadcast<float, VecSize>(0.1111111111f)); // 1/9

    acc = aie::mul(poly, z2);
    poly = aie::add(acc.template to_vector<float>(),
                    aie::broadcast<float, VecSize>(0.1428571429f)); // 1/7

    acc = aie::mul(poly, z2);
    poly = aie::add(acc.template to_vector<float>(),
                    aie::broadcast<float, VecSize>(0.2000000000f)); // 1/5

    acc = aie::mul(poly, z2);
    poly = aie::add(acc.template to_vector<float>(),
                    aie::broadcast<float, VecSize>(0.3333333333f)); // 1/3

    acc = aie::mul(poly, z2);
    poly = aie::add(acc.template to_vector<float>(), aie::broadcast<float, VecSize>(1.0f));

    // ln_m = 2 * z * poly
    aie::vector<float, VecSize> v_two = aie::broadcast<float, VecSize>(2.0f);
    acc = aie::mul(z, poly);
    aie::vector<float, VecSize> ln_m =
        aie::mul(acc.template to_vector<float>(), v_two).template to_vector<float>();

    // 4. Reconstruct: ln(x) = ln_m + e * ln(2)
    acc = aie::mul(v_e, ln2);
    aie::vector<float, VecSize> result = aie::add(ln_m, acc.template to_vector<float>());

    return result;
}

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
void ggml_op_cross_entropy_loss(const INPUT_DTYPE0 * __restrict logits,
                                const INPUT_DTYPE1 * __restrict labels,
                                OUTPUT_DTYPE * __restrict loss_out,
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
