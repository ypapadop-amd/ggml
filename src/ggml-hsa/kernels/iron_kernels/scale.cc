// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file scale.cc
 * @brief Scale and bias operation for AIE kernels.
 */

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"
#include "transform.h"

extern "C" {

/**
 * @brief Applies scale and bias to each element: out[i] = in[i] * scale + bias.
 *
 * Vectorized over 512-bit registers with a scalar tail for the remainder. scale/bias
 * are runtime args, not compile-time constants, since op_params can vary per dispatch
 * of the same compiled kernel.
 *
 * @param[in]  in    Input array of N float elements.
 * @param[out] out   Output array of N float elements.
 * @param[in]  N     Number of elements to process.
 * @param[in]  scale Multiplicative scale factor.
 * @param[in]  bias  Additive bias term.
 */
void ggml_op_scale(
    const float * __restrict in, float * __restrict out, int32_t N, float scale, float bias) {
    // scale.py streams the largest tile dividing the element count, which need not be a whole
    // number of vectors; aligned accesses only when the builder says it is.
    transform_vector_n<tile_vector_aligned_v<float>>(
        out, N,
        [scale, bias](auto v) {
            return aie::add(aie::mul(v, scale).template to_vector<float>(), bias);
        },
        [scale, bias](float v) { return v * scale + bias; }, in);
}

} // extern "C"
