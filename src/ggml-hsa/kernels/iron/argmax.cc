// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef KERN_VEC_SIZE
#define KERN_VEC_SIZE 16
#endif

extern "C" {

#ifdef GGML_OP_ARGMAX

/**
 * Argmax operation: finds the index of the maximum value in a row.
 *
 * Single-pass algorithm that tracks both max value and its index.
 * Uses vectorized operations for the aligned portion, scalar for remainder.
 *
 * @param in Input row (F32 array, tile may be padded beyond N)
 * @param out Output index (I32 scalar)
 * @param N Actual row length (number of valid elements)
 */
void ggml_op_argmax(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t N) {
    event0();

    // Simple scalar implementation that tracks index during max-finding.
    // This avoids floating-point comparison issues in a two-pass approach.
    float max_val = -3.4028235e+38f;
    int32_t argmax_idx = 0;

    for (int i = 0; i < N; i++) {
        float val = in[i];
        if (val > max_val) {
            max_val = val;
            argmax_idx = i;
        }
    }

    out[0] = argmax_idx;
    event1();
}

#endif // GGML_OP_ARGMAX

} // extern "C"
