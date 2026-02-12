// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {

void ggml_op_clamp(
    const INPUT_DTYPE * in, OUTPUT_DTYPE * out, int32_t N, float min_val, float max_val) {
    for (int32_t i = 0; i < N; ++i) {
        if (in[i] < static_cast<INPUT_DTYPE>(min_val)) {
            out[i] = static_cast<OUTPUT_DTYPE>(min_val);
        } else if (in[i] > static_cast<INPUT_DTYPE>(max_val)) {
            out[i] = static_cast<OUTPUT_DTYPE>(max_val);
        } else {
            out[i] = static_cast<OUTPUT_DTYPE>(in[i]);
        }
    }
}

} // extern "C"
