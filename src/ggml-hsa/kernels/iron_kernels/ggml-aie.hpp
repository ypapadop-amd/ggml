// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

/**
 * @file ggml-aie.hpp
 * @brief Common type definitions and utilities for AIE kernels.
 *
 * This header provides type aliases and type traits used across AIE kernels.
 */

#include <cstdint>
#include <type_traits>

#include "aie_api/aie.hpp"

using i8 = std::int8_t;   ///< Signed 8-bit integer type alias.
using i16 = std::int16_t; ///< Signed 16-bit integer type alias.
using i32 = std::int32_t; ///< Signed 32-bit integer type alias.
using bf16 = bfloat16;    ///< Brain floating-point 16-bit type alias.
using f32 = float;        ///< 32-bit floating-point type alias.

/**
 * @brief Type trait to check if a type is a floating-point type.
 *
 * This extends std::is_floating_point to also recognize bfloat16 as a
 * floating-point type, which is commonly used in AIE computations.
 *
 * @tparam T The type to check.
 *
 * Usage:
 * @code
 * static_assert(is_floating_point<float>::value);    // true
 * static_assert(is_floating_point<bfloat16>::value); // true
 * static_assert(!is_floating_point<int>::value);     // int is not floating-point
 * @endcode
 */
template <typename T>
struct is_floating_point
    : public std::integral_constant<bool,
                                    std::is_floating_point_v<T> || std::is_same_v<T, bfloat16>> {};

/**
 * @brief Helper variable template for is_floating_point.
 *
 * @tparam T The type to check.
 *
 * @return true if T is a floating-point type (including bfloat16), false otherwise.
 */
template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief Converts one f32 element to bf16 bits (round-to-nearest-even, NaN -> quiet).
 *
 * Replicates the host @c ggml_compute_fp32_to_bf16 integer arithmetic bit-for-bit, so vectorized
 * and scalar paths (and the host reference) all agree. Returns the raw 16-bit pattern; the caller
 * stores it into a bf16. Used for the scalar tail of any f32 -> bf16 kernel.
 *
 * @param[in] v The f32 value to convert.
 * @return The bf16 bit pattern as a uint16_t.
 */
inline std::uint16_t convert_f32_to_bf16_scalar(f32 v) {
    union {
        f32 f;
        std::uint32_t u;
    } bits;
    bits.f = v;
    if ((bits.u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<std::uint16_t>((bits.u >> 16) | 64u);
    }
    return static_cast<std::uint16_t>((bits.u + (0x7fffu + ((bits.u >> 16) & 1u))) >> 16);
}

/**
 * @brief Converts a vector of @p V f32 lanes to bf16 (round-to-nearest-even, NaN -> quiet).
 *
 * Lane-wise replica of @c convert_f32_to_bf16_scalar: applies the exact RNE integer arithmetic of
 * @c ggml_compute_fp32_to_bf16 rather than relying on hardware rounding/NaN handling, so the result
 * is bit-identical to the scalar path and the host reference. The bf16 bits land in the low 16 of
 * each u32 lane; @c filter_even grabs those low halves.
 *
 * @tparam V Vector width (number of f32 lanes).
 * @param[in] fv The f32 vector to convert.
 * @return The converted bf16 vector.
 */
template <int V>
inline aie::vector<bf16, V> convert_f32_to_bf16_vector(const aie::vector<f32, V> & fv) {
    const aie::vector<std::uint32_t, V> u = aie::vector_cast<std::uint32_t>(fv);
    const aie::vector<std::uint32_t, V> hi16 = aie::logical_downshift(u, 16);

    // NaN: (u >> 16) | 64
    const aie::vector<std::uint32_t, V> nan_val = aie::bit_or(64u, hi16);

    // RNE: (u + (0x7fff + ((u >> 16) & 1))) >> 16
    const aie::vector<std::uint32_t, V> lsb = aie::bit_and(1u, hi16);
    const aie::vector<std::uint32_t, V> rounded = aie::add(u, aie::add(lsb, 0x7fffu));
    const aie::vector<std::uint32_t, V> rne_val = aie::logical_downshift(rounded, 16);

    // nan_mask ? nan_val : rne_val   (select(v1, v2, m) == m ? v2 : v1)
    const auto nan_mask = aie::gt(aie::bit_and(0x7fffffffu, u), 0x7f800000u);
    const aie::vector<std::uint32_t, V> res32 = aie::select(rne_val, nan_val, nan_mask);

    const aie::vector<std::uint16_t, V> res16 = aie::filter_even(aie::vector_cast<std::uint16_t>(res32));
    return aie::vector_cast<bf16>(res16);
}
