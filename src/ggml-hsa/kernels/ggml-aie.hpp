// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <aie_api/aie.hpp>

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using bf16 = bfloat16;
using f32 = float;

template <typename T>
struct is_floating_point
    : public std::integral_constant<bool,
                                    std::is_floating_point_v<T> || std::is_same_v<T, bfloat16>> {};

template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;
