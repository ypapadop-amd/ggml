// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

extern "C" {

#ifdef GGML_OP_ARGMAX

/**
 * Argmax operation: finds the index of the maximum value in a row.
 *
 * Single-pass algorithm that tracks both max value and its index.
 *
 * @param in Input row
 * @param out Output index
 * @param N Actual row length (number of valid elements)
 */
void ggml_op_argmax(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    event0();

    if (N > 0) {
        auto max_val = in[0];
        int32_t argmax_idx = 0;

        for (int32_t i = 1; i < N; i++) {
            if (in[i] > max_val) {
                max_val = in[i];
                argmax_idx = i;
            }
        }

        out[0] = static_cast<OUTPUT_DTYPE>(argmax_idx);
    }

    event1();
}

#endif // GGML_OP_ARGMAX

} // extern "C"
