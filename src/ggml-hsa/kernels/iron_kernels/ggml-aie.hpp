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
 * static_assert(!is_floating_point<int>::value);     // true
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
