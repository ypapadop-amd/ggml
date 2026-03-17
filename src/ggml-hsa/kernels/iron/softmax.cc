// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

/**
 * @brief Computes the ALiBi slope for a head.
 */
inline float compute_alibi_slope(float max_bias, int32_t n_head, int32_t head_idx) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    uint32_t n_head_log2 = 1u << floor_log2(static_cast<uint32_t>(n_head));

    float m0 = pow2(-max_bias / n_head_log2);
    float m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (static_cast<uint32_t>(head_idx) < n_head_log2) {
        slope = m0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(head_idx); ++j) {
            slope *= m0;
        }
    } else {
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
 * Pure scalar implementation for correctness testing.
 * Implements: softmax(x_i) = exp(scale*x_i - max) / sum(exp(scale*x_j - max))
 */
void ggml_op_soft_max(const INPUT_DTYPE * __restrict in,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      float scale,
                      float max_bias) {
    event0();

    const float * input = reinterpret_cast<const float *>(in);
    float * output = reinterpret_cast<float *>(out);

    // Step 1: Find max for numerical stability
    float global_max = -3.4028235e+38f;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Step 3: Normalize
    float sum_inv = 1.0f / sum_total;
    for (int i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}
#endif // GGML_OP_SOFT_MAX

#ifdef GGML_OP_SOFT_MAX_WITH_MASK
/**
 * @brief Computes softmax with mask tensor and ALiBi biases.
 *
 * Pure scalar implementation.
 * Implements: softmax(scale*x + slope*mask)
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

    const float * input = reinterpret_cast<const float *>(in);
    const float * mask_input = reinterpret_cast<const float *>(mask);
    float * output = reinterpret_cast<float *>(out);

    // Compute ALiBi slope
    int32_t head_idx = tile_idx / rows_per_head;
    float slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Step 1: Find max for numerical stability
    float global_max = -3.4028235e+38f;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Step 3: Normalize
    float sum_inv = 1.0f / sum_total;
    for (int i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}
#endif // GGML_OP_SOFT_MAX_WITH_MASK

#ifdef GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS
/**
 * @brief Computes softmax with mask and sink tensors.
 *
 * Pure scalar implementation.
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

    const float * input = reinterpret_cast<const float *>(in);
    const float * mask_input = reinterpret_cast<const float *>(mask);
    float * output = reinterpret_cast<float *>(out);

    // Get sink value for this head
    int32_t head_idx = tile_idx / rows_per_head;
    float sink_val = sinks[head_idx];

    // Step 1: Find max for numerical stability (including sink)
    float global_max = sink_val;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale + mask_input[i];
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = input[i] * scale + mask_input[i] - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Add sink contribution to sum
    float sink_exp = scalar_exp(sink_val - global_max);
    sum_total += sink_exp;

    // Step 3: Normalize
    float sum_inv = 1.0f / sum_total;
    for (int i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}
#endif // GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

} // extern "C"
