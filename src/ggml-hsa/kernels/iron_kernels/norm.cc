// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file norm.cc
 * @brief Layer normalization (GGML_OP_NORM) over dim 0 for AIE kernels.
 */

#include <stdint.h>

#include <cstring>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Normalizes one row over dim 0.
 *
 * Computes y = (x - mean) / sqrt(variance + eps) where mean and variance are the
 * population statistics over the N row elements, matching
 * ggml_compute_forward_norm_f32 (variance divides by N, not N-1).
 *
 * @param[in]  in   Input row of N float elements.
 * @param[out] out  Output row of N float elements (may alias @p in).
 * @param[in]  N        Row length (nc = ne00).
 * @param[in]  eps_bits Raw IEEE-754 bits of the eps float (reinterpreted below).
 *                      Passed as int32 to avoid the peano-compat IR pass mangling
 *                      hex float immediates.
 */
void ggml_op_norm(const float * __restrict in,
                  float * __restrict out,
                  int32_t N,
                  int32_t eps_bits) {
    event0();

    float eps;
    std::memcpy(&eps, &eps_bits, sizeof(float));

    float sum = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        sum += in[i];
    }
    const float mean = sum / static_cast<float>(N);

    // Accumulate the variance from the centered values (read-only over in); NORM is
    // memory-bound, so we avoid materializing the centered row into out and reading it
    // back — pass 2 below writes the final normalized value straight from in.
    float variance = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        const float v = in[i] - mean;
        variance += v * v;
    }
    variance /= static_cast<float>(N);

    // Reciprocal sqrt via exp/log: 1/sqrt(a) = exp(-0.5 * log(a)). Reuses the
    // scalar_exp/scalar_log helpers, which compile cleanly on the AIE scalar path
    // (the aie::invsqrt intrinsic does not).
    const float scale = scalar_exp(-0.5f * scalar_log(variance + eps));
    for (int32_t i = 0; i < N; ++i) {
        out[i] = (in[i] - mean) * scale;
    }

    event1();
}

} // extern "C"
