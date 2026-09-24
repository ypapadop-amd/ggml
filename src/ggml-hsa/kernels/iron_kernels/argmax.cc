// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file argmax.cc
 * @brief Argmax operation for AIE kernels.
 */

#include <limits>

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
 * @param[out] out Index of the max element, written to out[0].
 * @param[in]  N   Number of elements to search.
 */
void ggml_op_argmax(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    event0();

    // Transcribed from ggml_vec_argmax_f32 (ggml-cpu/vec.h), which is the reference this has to
    // match. Two things in it are easy to get wrong, and the previous body got both:
    //
    //   max = MAX(max, x[i]); if (max == x[i]) { idx = i; }
    //
    // The index is updated on *equality*, not on strictly-greater, so on a tie the reference
    // keeps the LAST index holding the maximum. The previous `if (in[i] > max_val)` kept the
    // first, so a row like [0,1,9,3,4,3,2,9,0,-1] returned 2 where the reference returns 7.
    //
    // And the running maximum starts at -inf rather than in[0]. Since MAX(a,b) is (a > b ? a : b),
    // a NaN operand is not propagated -- it is replaced by the next element -- so a leading NaN
    // does not poison the scan. Seeding with in[0] meant a NaN there made every later comparison
    // false and the result was always 0.
    //
    // See test-argmax-hsa for the rows that pin these down.
    if (N > 0) {
        auto max_val = -std::numeric_limits<INPUT_DTYPE>::infinity();
        int32_t argmax_idx = 0;

        for (int32_t i = 0; i < N; i++) {
            max_val = (max_val > in[i]) ? max_val : in[i];
            if (max_val == in[i]) {
                argmax_idx = i;
            }
        }

        out[0] = static_cast<OUTPUT_DTYPE>(argmax_idx);
    }

    event1();
}

} // extern "C"
