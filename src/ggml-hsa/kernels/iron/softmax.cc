// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 8
#endif

/**
 * @brief Computes 2^x using range reduction for improved precision.
 *
 * Horner's method suffers from precision loss for large input values.
 * This function applies range reduction by splitting x into integer (i) and
 * fractional (f) parts where i = floor(x) and f is in [0, 1).
 * Formula: 2^x = 2^i * 2^f
 *
 * The fractional part 2^f is computed using a degree-10 Taylor series
 * approximation of exp(f * ln(2)) via Horner's method. The integer part 2^i
 * is computed using IEEE 754 bit manipulation.
 *
 * @param[in] x The exponent value.
 *
 * @return The computed value of 2^x.
 */
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

/**
 * @brief Computes floor(log2(x)) for positive integers.
 *
 * Finds the position of the most significant bit set in x.
 *
 * @param[in] x The input value (must be > 0 for meaningful result).
 *
 * @return The floor of log base 2 of x. Returns 0 for x <= 1.
 */
inline uint32_t floor_log2(uint32_t x) {
    uint32_t result = 0;
    while (x > 1) {
        x >>= 1;
        result++;
    }
    return result;
}

/**
 * @brief Computes the ALiBi (Attention with Linear Biases) slope for a head.
 *
 * ALiBi applies position-dependent biases in attention using geometric slopes.
 * The slope for each head is computed based on the head index and total heads.
 *
 * For heads 0 to n_head_log2-1: slope = m0^(head_idx+1)
 * For heads >= n_head_log2:     slope = m1^(2*(head_idx - n_head_log2) + 1)
 *
 * where m0 = 2^(-max_bias/n_head_log2) and m1 = 2^(-max_bias/2/n_head_log2)
 *
 * @param[in] max_bias  Maximum bias value. If <= 0, returns 1.0.
 * @param[in] n_head    Total number of attention heads.
 * @param[in] head_idx  Index of the current head (0-based).
 *
 * @return The computed ALiBi slope for this head.
 */
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
/**
 * @brief Computes softmax without mask or sink tensors.
 *
 * Implements the numerically stable softmax: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 *
 * Algorithm:
 * 1. Find global maximum of scaled inputs for numerical stability.
 * 2. Compute exp(scale*x - max) for each element and accumulate sum.
 * 3. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in       Input tensor of size N elements.
 * @param[out] out      Output tensor of size N elements (can alias in).
 * @param[in]  N        Number of elements in the row (must be divisible by KERN_VEC_SIZE).
 * @param[in]  scale    Scale factor applied to input before softmax.
 * @param[in]  max_bias Unused in this variant (kept for API consistency).
 */
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
/**
 * @brief Computes softmax with a mask tensor and ALiBi position biases.
 *
 * Implements: softmax(scale*x + slope*mask) where slope is computed via ALiBi.
 *
 * Algorithm:
 * 1. Compute ALiBi slope for current head based on tile_idx and rows_per_head.
 * 2. Find global maximum of (scale*input + slope*mask) for numerical stability.
 * 3. Compute exp(scale*x + slope*mask - max) and accumulate sum.
 * 4. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements (e.g., causal attention mask).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row (must be divisible by KERN_VEC_SIZE).
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Maximum ALiBi bias value.
 * @param[in]  n_head        Total number of attention heads.
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 */
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
/**
 * @brief Computes softmax with mask and sink (attention sink) tensors.
 *
 * This variant includes a "sink" value per attention head that participates
 * in the softmax normalization but is not stored in the output. Used for
 * streaming attention where early tokens act as attention sinks.
 *
 * Algorithm:
 * 1. Get sink value for current head based on tile_idx.
 * 2. Find global maximum including both (scale*input + mask) and sink.
 * 3. Compute exp(scale*x + mask - max) and accumulate sum.
 * 4. Add exp(sink - max) to sum for proper normalization.
 * 5. Normalize output by dividing each exp value by total sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements.
 * @param[in]  sinks         Per-head sink values array (indexed by head_idx).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row (must be divisible by KERN_VEC_SIZE).
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Unused in this variant.
 */
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
