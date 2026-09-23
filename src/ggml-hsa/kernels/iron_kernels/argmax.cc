// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file argmax.cc
 * @brief Argmax operation for AIE kernels.
 */

#include <cstdint>

#include <aie_api/aie.hpp>

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

    // The scalar scan below carries a loop-carried dependency and a data-dependent branch per
    // element, which is why it showed up at 1.42 us/image of a 34.67 us MNIST baseline (4.1%)
    // despite only covering a 10-element row. When the row length is known at compile time and
    // fits one vector, do it branch-free instead: one reduce_max, one compare, one bit scan.
    //
    // The design streams exactly one row per tile (see argmax.py), and the tile is exactly N
    // elements, so a vector load off `in` would read past the buffer. Stage the row into a
    // padded, vector-aligned scratch buffer. The padding lanes hold in[0], a value already in
    // the row: they cannot raise the maximum, and if in[0] *is* the maximum then lane 0 is set
    // too and the first-set-lane scan below still returns 0.
#if defined(ARGMAX_N) && (ARGMAX_N) > 0 && (ARGMAX_N) <= 16
    constexpr int32_t Nv = ARGMAX_N;
    constexpr int32_t V = 16;
    (void)N;

    alignas(64) INPUT_DTYPE xs[V];
    for (int32_t i = 0; i < Nv; i++) {
        xs[i] = in[i];
    }
    for (int32_t i = Nv; i < V; i++) {
        xs[i] = in[0];
    }

    const auto xv = aie::load_v<V>(xs);
    const auto max_val = aie::reduce_max(xv);

    // First lane equal to the maximum. Matches the scalar scan's strictly-greater comparison,
    // which keeps the first occurrence on ties.
    const auto eq = aie::eq(xv, aie::broadcast<INPUT_DTYPE, V>(max_val));

    // Index of the lowest set bit, by binary search over the 16-bit lane mask. Not
    // __builtin_ctz: that lowers to G_CTTZ_ZERO_UNDEF, which Peano cannot legalize for aie2
    // ("unable to legalize instruction"). The mask is only zero if no lane compared equal,
    // which needs every lane to be NaN; the scalar scan returns 0 for that input, so match it.
    unsigned m = eq.to_uint32() & 0xFFFFu;
    int32_t idx = 0;
    if (m != 0u) {
        if ((m & 0x00FFu) == 0u) { idx += 8; m >>= 8; }
        if ((m & 0x000Fu) == 0u) { idx += 4; m >>= 4; }
        if ((m & 0x0003u) == 0u) { idx += 2; m >>= 2; }
        if ((m & 0x0001u) == 0u) { idx += 1; }
    }
    out[0] = static_cast<OUTPUT_DTYPE>(idx);
#else
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
#endif

    event1();
}

} // extern "C"
