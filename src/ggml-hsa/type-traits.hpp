// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <cstdint>

#include "ggml.h"
#include "ggml-impl.h"

/**
 * @brief @ref ggml_type traits.
 */
template <ggml_type T>
struct ggml_hsa_type_traits;

template <>
struct ggml_hsa_type_traits<GGML_TYPE_F32> {
    static constexpr ggml_type ggml_type = GGML_TYPE_F32;
    using type = float;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_F16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_F16;
    using type = ggml_fp16_t;
    static constexpr bool is_fundamental = false;
    static constexpr auto to_fp32 = [](ggml_fp16_t v) -> float { return GGML_FP16_TO_FP32(v); };
    static constexpr auto from_fp32 = [](float v) -> ggml_fp16_t { return GGML_FP32_TO_FP16(v); };
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I8> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I8;
    using type = std::int8_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I16;
    using type = std::int16_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I32> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I32;
    using type = std::int32_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_BF16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_BF16;
    using type = ggml_bf16_t;
    static constexpr bool is_fundamental = false;
    static constexpr auto to_fp32 = [](ggml_bf16_t v) -> float { return GGML_BF16_TO_FP32(v); };
    static constexpr auto from_fp32 = [](float v) -> ggml_bf16_t { return GGML_FP32_TO_BF16(v); };
};
