// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {

void ggml_op_clamp(
    const INPUT_DTYPE * in, OUTPUT_DTYPE * out, int32_t N, float min_val, float max_val) {
    for (int32_t i = 0; i < N; ++i) {
        INPUT_DTYPE val = in[i];
        if (val < static_cast<INPUT_DTYPE>(min_val)) {
            val = static_cast<INPUT_DTYPE>(min_val);
        } else if (val > static_cast<INPUT_DTYPE>(max_val)) {
            val = static_cast<INPUT_DTYPE>(max_val);
        }
        out[i] = val;
    }
}

} // extern "C"
