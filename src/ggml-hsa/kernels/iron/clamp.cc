// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file clamp.cc
 * @brief Clamp operation for AIE kernels.
 */

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Clamps each element to a specified range: out[i] = clamp(in[i], min, max).
 *
 * For each element, if the value is less than min_val, it is set to min_val.
 * If greater than max_val, it is set to max_val. Otherwise, it is unchanged.
 *
 * @param[in]  in      Input array of N elements.
 * @param[out] out     Output array of N elements.
 * @param[in]  N       Number of elements to process.
 * @param[in]  min_val Minimum allowed value (inclusive).
 * @param[in]  max_val Maximum allowed value (inclusive).
 */
void ggml_op_clamp(
    const INPUT_DTYPE * in, OUTPUT_DTYPE * out, int32_t N, float min_val, float max_val) {
    for (int32_t i = 0; i < N; ++i) {
        if (in[i] < static_cast<INPUT_DTYPE>(min_val)) {
            out[i] = static_cast<OUTPUT_DTYPE>(min_val);
        } else if (in[i] > static_cast<INPUT_DTYPE>(max_val)) {
            out[i] = static_cast<OUTPUT_DTYPE>(max_val);
        } else {
            out[i] = static_cast<OUTPUT_DTYPE>(in[i]);
        }
    }
}

} // extern "C"
