// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

// Pooling op selector (matches enum ggml_op_pool in include/ggml.h).
constexpr int32_t GGML_OP_POOL_MAX = 0;
constexpr int32_t GGML_OP_POOL_AVG = 1;

extern "C" {

/**
 * @brief Reduces each k1 x k0 window of one input channel-plane to an output element.
 *
 * Mirrors ggml_compute_forward_pool_2d for a single channel-plane. Padding is
 * handled by skipping out-of-bounds taps rather than gathering a padded
 * buffer: for MAX this is equivalent to -inf padding, and for AVG the divisor
 * is still the full k0*k1 window area (not the count of in-bounds taps),
 * matching the GGML CPU reference bit-for-bit.
 *
 * @param[in]  in   Input channel-plane of iw * ih elements (row-major, width fastest).
 * @param[out] out  Output channel-plane of ow * oh elements.
 * @param[in]  iw   Input width.
 * @param[in]  ih   Input height.
 * @param[in]  ow   Output width.
 * @param[in]  oh   Output height.
 * @param[in]  k0   Kernel width.
 * @param[in]  k1   Kernel height.
 * @param[in]  s0   Stride along width.
 * @param[in]  s1   Stride along height.
 * @param[in]  p0   Padding along width.
 * @param[in]  p1   Padding along height.
 * @param[in]  op   Pooling op: GGML_OP_POOL_MAX or GGML_OP_POOL_AVG.
 */
void ggml_op_pool_2d(const INPUT_DTYPE * __restrict in,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t iw,
                     int32_t ih,
                     int32_t ow,
                     int32_t oh,
                     int32_t k0,
                     int32_t k1,
                     int32_t s0,
                     int32_t s1,
                     int32_t p0,
                     int32_t p1,
                     int32_t op) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const int32_t offset0 = -p0;
    const int32_t offset1 = -p1;

    const bool is_max = (op == GGML_OP_POOL_MAX);

    for (int32_t oy = 0; oy < oh; ++oy) {
        for (int32_t ox = 0; ox < ow; ++ox) {
            const int32_t ix = offset0 + ox * s0;
            const int32_t iy = offset1 + oy * s1;

            float res;
            if (is_max) {
                res = std::numeric_limits<float>::lowest();
                for (int32_t ky = 0; ky < k1; ++ky) {
                    const int32_t y = iy + ky;
                    if (y < 0 || y >= ih) {
                        continue;
                    }
                    const auto * srow = in + static_cast<int32_t>(y) * iw;
                    for (int32_t kx = 0; kx < k0; ++kx) {
                        const int32_t x = ix + kx;
                        if (x < 0 || x >= iw) {
                            continue;
                        }
                        const auto v = static_cast<float>(srow[x]);
                        res = (v > res) ? v : res;
                    }
                }
            } else {
                res = 0.0f;
                for (int32_t ky = 0; ky < k1; ++ky) {
                    const int32_t y = iy + ky;
                    if (y < 0 || y >= ih) {
                        continue;
                    }
                    const auto * srow = in + static_cast<int32_t>(y) * iw;
                    for (int32_t kx = 0; kx < k0; ++kx) {
                        const int32_t x = ix + kx;
                        if (x < 0 || x >= iw) {
                            continue;
                        }
                        res += static_cast<float>(srow[x]);
                    }
                }
                res *= 1.0f / static_cast<float>(k0 * k1);
            }

            out[oy * ow + ox] = res;
        }
    }

    event1();
}

} // extern "C"
