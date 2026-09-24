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
 * Single-pass algorithm that tracks both the maximum value and its index, matching
 * @c ggml_vec_argmax_f32 (ggml-cpu/vec.h) exactly:
 *
 * The reference is @c max = MAX(max, x[i]) followed by @c if @c (max @c == @c x[i]) @c idx @c =
 * @c i, where @c MAX(a,b) is @c (a @c > @c b @c ? @c a @c : @c b). Two consequences, both of
 * which callers have to live with because they are the reference's:
 *
 * - On a tie the index is updated on @b equality, not strictly-greater, so the @b last element
 *   holding the maximum wins.
 * - A NaN @b does become the running maximum: @c a @c > @c NaN is false, so @c MAX returns the
 *   NaN. The next element then displaces it the same way. A NaN therefore @b discards whatever
 *   maximum preceded it, and the result is decided by the elements @b after the last NaN --
 *   this is not "the largest non-NaN element". @c [1, @c NaN, @c -1] returns 2, not 0. A row
 *   whose last NaN is also its last element keeps the index last set before it, or 0 if none
 *   ever was, which is also what an all-NaN row reports.
 *
 * Floating-point input only: the running maximum is seeded with -infinity, which an integer
 * type cannot represent. @c argmax.py rejects any other input dtype, matching ggml, whose CPU
 * implementation supports @c GGML_TYPE_F32 and aborts on everything else.
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
    // And the running maximum starts at -inf rather than in[0]. That matters for NaN. Since
    // MAX(a,b) is (a > b ? a : b) and a > NaN is false, a NaN becomes the running maximum and is
    // then displaced by the next element -- so a NaN discards the maximum before it, and never
    // compares equal to itself, so it never sets the index. Seeding with in[0] instead meant a
    // NaN there made every later comparison false and the answer was always 0, where the
    // reference goes on to scan the rest of the row.
    //
    // See test-argmax-hsa for the rows that pin these down.
    //
    // The -inf seed is why this kernel is floating-point only: for an integer INPUT_DTYPE
    // numeric_limits<T>::infinity() is 0, so an all-negative row would never update the running
    // maximum and would report index 0. argmax.py rejects non-f32 input, and this catches it at
    // compile time if that check is ever loosened. numeric_limits<T>::lowest() is not a fix --
    // it would break the all- -inf row, which the reference resolves to the last index.
    static_assert(std::numeric_limits<INPUT_DTYPE>::has_infinity,
                  "ggml_op_argmax seeds its running maximum with -infinity to match "
                  "ggml_vec_argmax_f32; INPUT_DTYPE must be a floating-point type");

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
