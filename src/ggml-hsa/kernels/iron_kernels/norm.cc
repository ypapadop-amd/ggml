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

    // All three passes are vectorized with a scalar tail. N is the row length, a runtime
    // argument, so the tail covers any row that is not a whole number of vectors. Unaligned
    // load/store because a row base is not guaranteed vector-aligned.
    constexpr int32_t V = 512 / (sizeof(float) * 8);
    const int32_t vend = (N / V) * V;

    // Pass 1: sum(in).
    float sum = 0.0f;
    if (vend > 0) {
        aie::vector<float, V> vsum = aie::zeros<float, V>();
        for (int32_t i = 0; i < vend; i += V) {
            vsum = aie::add(vsum, aie::load_unaligned_v<V>(in + i));
        }
        sum = aie::reduce_add(vsum);
    }
    for (int32_t i = vend; i < N; ++i) {
        sum += in[i];
    }
    const float mean = sum / static_cast<float>(N);

    // Accumulate the variance from the centered values (read-only over in); NORM is
    // memory-bound, so we avoid materializing the centered row into out and reading it
    // back — pass 2 below writes the final normalized value straight from in.
    float variance = 0.0f;
    const aie::vector<float, V> vmean = aie::broadcast<float, V>(mean);
    if (vend > 0) {
        aie::vector<float, V> vvar = aie::zeros<float, V>();
        for (int32_t i = 0; i < vend; i += V) {
            const aie::vector<float, V> c = aie::sub(aie::load_unaligned_v<V>(in + i), vmean);
            vvar = aie::add(vvar, aie::mul(c, c).template to_vector<float>());
        }
        variance = aie::reduce_add(vvar);
    }
    for (int32_t i = vend; i < N; ++i) {
        const float v = in[i] - mean;
        variance += v * v;
    }
    variance /= static_cast<float>(N);

    // Reciprocal sqrt via exp/log: 1/sqrt(a) = exp(-0.5 * log(a)). Reuses the
    // scalar_exp/scalar_log helpers, which compile cleanly on the AIE scalar path
    // (the aie::invsqrt intrinsic does not). Evaluated once per row, so it stays scalar.
    const float scale = scalar_exp(-0.5f * scalar_log(variance + eps));

    // Pass 3: out = (in - mean) * scale.
    const aie::vector<float, V> vscale = aie::broadcast<float, V>(scale);
    for (int32_t i = 0; i < vend; i += V) {
        const aie::vector<float, V> c = aie::sub(aie::load_unaligned_v<V>(in + i), vmean);
        aie::store_unaligned_v(out + i, aie::mul(c, vscale).template to_vector<float>());
    }
    for (int32_t i = vend; i < N; ++i) {
        out[i] = (in[i] - mean) * scale;
    }

    event1();
}

} // extern "C"
