// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 16
#endif

template <int VecSize = KERN_VEC_SIZE>
inline aie::vector<float, VecSize> vec_exp(aie::vector<float, VecSize> & x) {
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

// Scalar 2^x using range reduction
// Horner's method suffer from precision loss for large input values
// We apply range reduction technique, which splits x into integer (i) and
// fractional (f) parts where i = floor(x) and f is between 0 and 1.
// Formula: 2^x = 2^i * 2^f
inline float pow2(float x) {
    // split x into integer and fractional parts
    int i = (int)x;
    if (x < (float)i) {
        i--;
    }
    float f = x - (float)i;

    constexpr float pow2_coeffs[] = {
        0.0000000070549116f, // ln(2)^10 / 10!
        0.0000001017808600f, // ln(2)^9 / 9!
        0.0000013215486790f, // ln(2)^8 / 8!
        0.0000152525277765f, // ln(2)^7 / 7!
        0.0001540353039338f, // ln(2)^6 / 6!
        0.0013333558146428f, // ln(2)^5 / 5!
        0.0096181291076285f, // ln(2)^4 / 4!
        0.0555041086648216f, // ln(2)^3 / 3!
        0.2402265069591007f, // ln(2)^2 / 2!
        0.6931471805599453f, // ln(2)^1 / 1!
        1.0f                 // ln(2)^0 / 0!
    };
    constexpr int NUM_POW2_COEFFS = sizeof(pow2_coeffs) / sizeof(pow2_coeffs[0]);

    // compute 2^f using Horner's method for Taylor series of exp(f * ln(2))
    float exp_f = pow2_coeffs[0];

#pragma unroll
    for (int j = 1; j < NUM_POW2_COEFFS; ++j) {
        exp_f = exp_f * f + pow2_coeffs[j];
    }

    // this takes a couple of cycles to compute 2^i using IEEE 754 bit manipulation
    // IEEE 754 float: 2^i is represented as exponent = 127 + i, mantissa = 0
    // create the integer representation of 2^i
    int32_t bits = (127 + i) << 23;
    // cast the bits directly to float
    float scale = reinterpret_cast<float &>(bits);

    return exp_f * scale;
}

// floor of log2 for positive integers
// by finding the index of the most significant bit
inline uint32_t floor_log2(uint32_t x) {
    uint32_t result = 0;
    while (x > 1) {
        x >>= 1;
        result++;
    }
    return result;
}

// ALiBi slope computation
inline float compute_alibi_slope(float max_bias, int32_t n_head, int32_t head_idx) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    uint32_t n_head_log2 = 1u << floor_log2((uint32_t)n_head);

    // compute base values m0 and m1
    float m0 = pow2(-max_bias / n_head_log2);
    float m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (head_idx < n_head_log2) {
        // slope = m0^(head_idx+1) via repeated multiplication
        slope = m0;
        for (uint32_t j = 0; j < head_idx; ++j) {
            slope *= m0;
        }
    } else {
        // slope = m1^(2*(head_idx - n_head_log2) + 1)
        uint32_t exp = 2 * (head_idx - n_head_log2) + 1;
        slope = m1;
        for (uint32_t j = 1; j < exp; ++j) {
            slope *= m1;
        }
    }

    return slope;
}

extern "C" {
#ifdef GGML_OP_SOFT_MAX
// Softmax without mask or sink
void ggml_op_soft_max(const INPUT_DTYPE * __restrict in,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      float scale,
                      float max_bias) {
    event0();

    constexpr int VEC_SIZE = KERN_VEC_SIZE;
    const int num_iters = N / VEC_SIZE;

    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);

    // find max value for numerical stability

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

    // compute exp(x - max) and sum

    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;

        // apply scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // subtract max for numerical stability
        aie::vector<float, VEC_SIZE> x = aie::sub(scaled_input, v_global_max);

        // compute exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);

        // store exp values
        *it_exp_out++ = exp_val;
    }

    // normalize by dividing by sum

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

#endif // GGML_OP_SOFT_MAX

#ifdef GGML_OP_SOFT_MAX_WITH_MASK
// Softmax with mask tensor
void ggml_op_soft_max_with_mask(const INPUT_DTYPE * __restrict in,
                                const MASK_DTYPE * __restrict mask,
                                OUTPUT_DTYPE * __restrict out,
                                int32_t N,
                                float scale,
                                float max_bias,
                                int32_t n_head,
                                int32_t tile_idx,
                                int32_t rows_per_head) {
    event0();

    constexpr int VEC_SIZE = KERN_VEC_SIZE;
    const int num_iters = N / VEC_SIZE;

    // compute ALiBi slope
    uint32_t head_idx = (uint32_t)(tile_idx / rows_per_head);
    float slope = compute_alibi_slope(max_bias, (uint32_t)n_head, head_idx);

    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);

    // find max(scale * in + slope * mask)

    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_max_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_max_mask++;

        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // scaled_mask = mask * slope (ALiBi)
        aie::accum<accfloat, VEC_SIZE> mask_accum = aie::mul(mask_vec, slope);
        aie::vector<float, VEC_SIZE> scaled_mask = mask_accum.to_vector<float>();

        // masked_input = scaled_input + scaled_mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, scaled_mask);

        v_max = aie::max(v_max, masked_input);
    }

    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // compute exp(scale * in + slope * mask - max) and accumulate sum

    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_mask++;

        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // scaled_mask = mask * slope (ALiBi)
        aie::accum<accfloat, VEC_SIZE> mask_accum = aie::mul(mask_vec, slope);
        aie::vector<float, VEC_SIZE> scaled_mask = mask_accum.to_vector<float>();

        // masked_input = scaled_input + scaled_mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, scaled_mask);

        // x = masked_input - max (numerical stability)
        aie::vector<float, VEC_SIZE> x = aie::sub(masked_input, v_global_max);

        // exp_val = exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);

        // store exp values for normalization pass
        *it_exp_out++ = exp_val;
    }

    // normalize by dividing by sum

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

#endif // GGML_OP_SOFT_MAX_WITH_MASK

#ifdef GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS
// Softmax with mask and sink tensors
void ggml_op_soft_max_with_mask_and_sinks(const INPUT_DTYPE * __restrict in,
                                          const MASK_DTYPE * __restrict mask,
                                          const SINK_DTYPE * __restrict sinks,
                                          OUTPUT_DTYPE * __restrict out,
                                          int32_t N,
                                          int32_t tile_idx,
                                          int32_t rows_per_head,
                                          float scale,
                                          float max_bias) {
    event0();

    constexpr int VEC_SIZE = KERN_VEC_SIZE;
    const int num_iters = N / VEC_SIZE;

    // calculate which head this tile belongs to and get the sink value
    int32_t head_idx = tile_idx / rows_per_head;
    float sink_val = sinks[head_idx];

    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);

    // find max value for numerical stability
    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_max_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_max_mask++;

        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);

        v_max = aie::max(v_max, masked_input);
    }

    // reduce to scalar max, then include sink in max calculation
    float global_max = aie::reduce_max(v_max);
    global_max = (global_max > sink_val) ? global_max : sink_val;
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // compute exp(scale * in + mask - max) and accumulate sum
    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_mask++;

        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);

        // x = masked_input - max (for numerical stability)
        aie::vector<float, VEC_SIZE> x = aie::sub(masked_input, v_global_max);

        // exp_val = exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);

        // store exp values for normalization pass
        *it_exp_out++ = exp_val;
    }

    // reduce sum across vector lanes
    aie::vector<float, VEC_SIZE> v_sum_vec = v_sum_accum.to_vector<float>();
    float sum_total = aie::reduce_add(v_sum_vec);

    // compute exp(sink - max) using vec_exp and add to sum
    float sink_shifted = sink_val - global_max;
    aie::vector<float, VEC_SIZE> sink_vec = aie::broadcast<float, VEC_SIZE>(sink_shifted);
    aie::vector<float, VEC_SIZE> sink_exp_vec = vec_exp<VEC_SIZE>(sink_vec);
    float sink_exp = sink_exp_vec.get(0);

    sum_total += sink_exp;
    float sum_inv = aie::inv(sum_total);

    // normalize by multiplying with 1/sum
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> in_elems = *it_scale_in++;
        aie::accum<accfloat, VEC_SIZE> out_accum = aie::mul(in_elems, sum_inv);
        *it_soft_out++ = out_accum.to_vector<float>();
    }

    event1();
}
#endif // GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

} // extern "C"
