// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file clamp.cc
 * @brief Clamp operation for AIE kernels.
 */

#include <type_traits>

#include "ggml-aie.hpp"
#include "transform.h"

extern "C" {

/**
 * @brief Clamps each element to a specified range: out[i] = max(min(in[i], max_val), min_val).
 *
 * The bounds apply in the CPU backend's order, MAX(MIN(x, max), min), so a range with
 * min_val > max_val gives min_val everywhere, as it does there.
 *
 * f32 compares on @ref float_order_key instead of as floats: the aie2 vector unit has no f32
 * min/max (aie::min/max subtract in the accumulator and test the sign after a bf16 conversion) and
 * the scalar unit has no float compare at all (a runtime call). The key orders non-NaN floats as
 * `<` does, -0 equal to +0, and the result is selected from the input or bound bits, so every
 * non-NaN result is the CPU's bit for bit. A NaN with the sign bit clear clamps to max_val, as on
 * the CPU; one with it set clamps to min_val, where the CPU gives max_val.
 *
 * @param[in]  in      Input array of N elements.
 * @param[out] out     Output array of N elements.
 * @param[in]  N       Number of elements to process.
 * @param[in]  min_val Minimum allowed value (inclusive).
 * @param[in]  max_val Maximum allowed value (inclusive).
 */
void ggml_op_clamp(const INPUT_DTYPE * __restrict in,
                   OUTPUT_DTYPE * __restrict out,
                   int32_t N,
                   float min_val,
                   float max_val) {
    // Aligned accesses only when clamp.py guaranteed the tile is a whole number of vectors.
    constexpr bool aligned = tile_vector_aligned_v<OUTPUT_DTYPE>;
    if constexpr (std::is_same_v<INPUT_DTYPE, float> && std::is_same_v<OUTPUT_DTYPE, float>) {
        const std::int32_t klo = float_order_key(min_val);
        const std::int32_t khi = float_order_key(max_val);
        transform_vector_n<aligned>(
            out, N,
            [=](auto v) {
                const auto k = float_order_keys<static_cast<int>(decltype(v)::size())>(v);
                const auto m = aie::select(max_val, v, aie::lt(k, khi));
                return aie::select(min_val, m, aie::gt(aie::min(k, khi), klo));
            },
            [=](float v) {
                const std::int32_t k = float_order_key(v);
                const float m = k < khi ? v : max_val;
                return (k < khi ? k : khi) > klo ? m : min_val;
            },
            in);
    } else {
        const auto lo = static_cast<INPUT_DTYPE>(min_val);
        const auto hi = static_cast<INPUT_DTYPE>(max_val);
        transform_vector_n<aligned>(
            out, N, [=](auto v) { return aie::max(aie::min(v, hi), lo); },
            [=](INPUT_DTYPE v) {
                const INPUT_DTYPE m = v < hi ? v : hi;
                return static_cast<OUTPUT_DTYPE>(m > lo ? m : lo);
            },
            in);
    }
}

} // extern "C"
