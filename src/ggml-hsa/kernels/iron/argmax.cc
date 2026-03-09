// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file argmax.cc
 * @brief Argmax operation for AIE kernels.
 */

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Finds the index of the maximum value in an input array.
 *
 * Single-pass algorithm that tracks both the maximum value and its index.
 * If multiple elements have the same maximum value, returns the index of
 * the first occurrence.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array containing the index of the maximum element.
 *                 Only the first element (out[0]) is written.
 * @param[in]  N   Number of elements to search. If N <= 0, no output is written.
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

} // extern "C"
