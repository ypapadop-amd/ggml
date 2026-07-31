// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file scale.cc
 * @brief Scale and bias operation for AIE kernels.
 */

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

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
    event0();

    constexpr int32_t V = 512 / (sizeof(float) * 8);
    const int32_t vend = (N / V) * V;

    // Aligned load/store: scale.py streams whole power-of-two tiles, so in/out are
    // vector-aligned (unlike row-tiled kernels, e.g. binary_ops ADD bias, whose
    // per-row stride isn't). MIN_ITERATION_COUNT(1): N (== tile_size) is at least V.
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<float, V> v = aie::load_v<V>(in + i);
        aie::vector<float, V> r = aie::mul(v, scale).to_vector<float>();
        aie::store_v(out + i, aie::add(r, bias));
    }

    for (int32_t i = vend; i < N; ++i) {
        out[i] = in[i] * scale + bias;
    }

    event1();
}

} // extern "C"
