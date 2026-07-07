// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

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

    const uint32_t n_head_log2 = 1u << floor_log2(static_cast<uint32_t>(n_head));
    const auto m0 = pow2(-max_bias / n_head_log2);
    const auto m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (static_cast<uint32_t>(head_idx) < n_head_log2) {
        // slope = m0^(head_idx+1) via repeated multiplication
        slope = m0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(head_idx); ++j) {
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
 * @param[in]  N        Number of elements in the row.
 * @param[in]  scale    Scale factor applied to input before softmax.
 * @param[in]  max_bias Unused in this variant (kept for API consistency).
 */
void ggml_op_soft_max(const INPUT_DTYPE * __restrict in,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      float scale,
                      float max_bias) {
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const float * input = reinterpret_cast<const float *>(in);
    float * output = reinterpret_cast<float *>(out);

    // Step 1: Find max for numerical stability
    float global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Step 3: Normalize
    float sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; ++i) {
        output[i] *= sum_inv;
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
 * @param[in]  N             Number of elements in the row.
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
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // Compute ALiBi slope
    const auto head_idx = tile_idx / rows_per_head;
    const auto slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Step 1: Find max for numerical stability
    float global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Step 3: Normalize
    float sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; ++i) {
        output[i] *= sum_inv;
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
 * @param[in]  N             Number of elements in the row.
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

    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<SINK_DTYPE, float>::value, "SINK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // Get sink value for this head
    const auto head_idx = tile_idx / rows_per_head;
    const auto sink_val = sinks[head_idx];

    // Step 1: Find max for numerical stability (including sink)
    float global_max = sink_val;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i];
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Add sink contribution to sum
    const auto sink_exp = scalar_exp(sink_val - global_max);
    sum_total += sink_exp;

    // Step 3: Normalize
    const auto sum_inv = 1.0f / sum_total;
    for (int i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}

#endif // GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

} // extern "C"
