// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file scale.cc
 * @brief Scale and bias operation for AIE kernels.
 */

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Applies scale and bias to each element: out[i] = in[i] * scale + bias.
 *
 * @param[in]  in    Input array of N float elements.
 * @param[out] out   Output array of N float elements.
 * @param[in]  N     Number of elements to process.
 * @param[in]  scale Multiplicative scale factor.
 * @param[in]  bias  Additive bias term.
 */
void ggml_op_scale(
    const float * __restrict in, float * __restrict out, int32_t N, float scale, float bias) {
    for (int i = 0; i < N; ++i) {
        out[i] = in[i] * scale + bias;
    }
}

} // extern "C"