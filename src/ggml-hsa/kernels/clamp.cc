// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {
void ggml_op_clamp(
    const float * __restrict in, float * __restrict out, int32_t N, float min, float max) {
    for (int i = 0; i < N; ++i) {
        if (in[i] < min) {
            out[i] = min;
        } else if (in[i] > max) {
            out[i] = max;
        } else {
            out[i] = in[i];
        }
    }
}
} // extern "C"