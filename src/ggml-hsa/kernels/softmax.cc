// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

template <int VEC_SIZE>
inline aie::vector<float, VEC_SIZE> vec_exp(aie::vector<float, VEC_SIZE> x) {
    // Taylor series coefficients for exp(x)

    constexpr float C0 = 1.0f;
    constexpr float C1 = 1.0f;
    constexpr float C2 = 0.5f;                 // 1/2!
    constexpr float C3 = 0.1666666666666667f;  // 1/3!
    constexpr float C4 = 0.0416666666666667f;  // 1/4!
    constexpr float C5 = 0.0083333333333333f;  // 1/5!
    constexpr float C6 = 0.0013888888888889f;  // 1/6!
    constexpr float C7 = 0.0001984126984127f;  // 1/7!
    constexpr float C8 = 0.0000248015873016f;  // 1/8!
    constexpr float C9 = 0.0000027557319224f;  // 1/9!
    constexpr float C10 = 0.0000002755731922f; // 1/10!
    constexpr float C11 = 0.0000000250521084f; // 1/11!

    // Clamp x to prevent overflow (exp(-88) ≈ 0, exp(88) ≈ inf)
    aie::vector<float, VEC_SIZE> v_clamp_min = aie::broadcast<float, VEC_SIZE>(-88.0f);
    aie::vector<float, VEC_SIZE> v_clamp_max = aie::broadcast<float, VEC_SIZE>(88.0f);
    x = aie::max(x, v_clamp_min);
    x = aie::min(x, v_clamp_max);

    /*
    Compute exp(x) using Horner's method:

    exp(x) ≈ C0 + x * (
             C1 + x * (
                 C2 + x * (
                     C3 + x * (
                         C4 + x * (
                             C5 + x * (
                                 C6 + x * (
                                     C7 + x * (
                                         C8 + x * (
                                             C9 + x * (
                                                 C10 + x * C11
                                             )
                                         )
                                     )
                                 )
                             )
                         )
                     )
                 )
             )
         )
    */

    aie::accum<accfloat, VEC_SIZE> tmp_accum;
    aie::vector<float, VEC_SIZE> poly = aie::broadcast<float, VEC_SIZE>(C11);

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C10));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C9));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C8));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C7));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C6));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C5));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C4));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C3));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C2));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C1));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    aie::vector<float, VEC_SIZE> result = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C0));

    // Clamp to positive (exp is always positive)
    aie::vector<float, VEC_SIZE> v_exp_min = aie::broadcast<float, VEC_SIZE>(1e-38f);
    result = aie::max(result, v_exp_min);

    return result;
}

extern "C" {
#ifdef COMPILE_GGML_OP_SOFTMAX
// Softmax without mask or positional encoding
void ggml_op_softmax(const INPUT_DTYPE * __restrict in,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N,
                     float scale,
                     float max_bias) {
    event0();

    constexpr int VEC_SIZE = 16;
    const int num_iters = N >> 4; // N / 16

    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);

    // Find max value for numerical stability

    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        v_max = aie::max(v_max, scaled_input);
    }

    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // Compute exp(x - max) and sum

    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;

        // Apply scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // Subtract max for numerical stability
        aie::vector<float, VEC_SIZE> x = aie::sub(scaled_input, v_global_max);

        // Compute exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // Accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);

        // Store exp values
        *it_exp_out++ = exp_val;
    }

    // Normalize by dividing by sum

    aie::vector<float, VEC_SIZE> v_sum_vec = v_sum_accum.to_vector<float>();
    float sum_total = aie::reduce_add(v_sum_vec);
    float sum_inv = aie::inv(sum_total);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> in_elems = *it_scale_in++;
        aie::accum<accfloat, VEC_SIZE> out_accum = aie::mul(in_elems, sum_inv);
        *it_soft_out++ = out_accum.to_vector<float>();
    }

    event1();
}

#endif // COMPILE_GGML_OP_SOFTMAX

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MAX
// Softmax with mask tensor
void ggml_op_softmax_with_mask(const INPUT_DTYPE * __restrict in,
                               const MASK_DTYPE * __restrict mask,
                               OUTPUT_DTYPE * __restrict out,
                               int32_t N,
                               float scale,
                               float max_bias) {}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MAX

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_POS
// Softmax with mask and positional encoding tensors
void ggml_op_softmax_with_mask_and_pos(const INPUT_DTYPE * __restrict in,
                                       const MASK_DTYPE * __restrict mask,
                                       const POS_DTYPE * __restrict pos,
                                       OUTPUT_DTYPE * __restrict out,
                                       int32_t N,
                                       float scale,
                                       float max_bias) {}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_POS

} // extern "C"