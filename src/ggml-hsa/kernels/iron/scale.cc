// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {
void ggml_op_scale(
    const float * __restrict in, float * __restrict out, int32_t N, float scale, float bias) {
    for (int i = 0; i < N; ++i) {
        out[i] = in[i] * scale + bias;
    }
}
} // extern "C"