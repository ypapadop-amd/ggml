// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Shared helpers for the HSA backend tests: f32/bf16 element encode/decode/round used to build
// inputs and compute references for the convert / convert_pad / depad ops.

#pragma once

#include <cstdint>

#include "ggml.h"

namespace hsa_test {

// Writes the float value @p v into @p bytes at element @p idx, encoded as @p type (f32 or bf16).
inline void store_val(ggml_type type, void * bytes, int64_t idx, float v) {
    if (type == GGML_TYPE_F32) {
        static_cast<float *>(bytes)[idx] = v;
    } else {
        static_cast<uint16_t *>(bytes)[idx] = ggml_fp32_to_bf16(v).bits;
    }
}

// Reads element @p idx from @p bytes (encoded as @p type) as a float.
inline float load_val(ggml_type type, const void * bytes, int64_t idx) {
    if (type == GGML_TYPE_F32) {
        return static_cast<const float *>(bytes)[idx];
    }
    return ggml_bf16_to_fp32(ggml_bf16_t{static_cast<const uint16_t *>(bytes)[idx]});
}

// Rounds @p v through @p type, returning the value the destination would hold.
inline float cast_val(ggml_type type, float v) {
    if (type == GGML_TYPE_F32) {
        return v;
    }
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
}

} // namespace hsa_test
