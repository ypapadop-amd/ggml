// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Adds the cross-entropy loss of @p rows rows to @p loss_out[0].
 *
 * Per row: loss = -sum(labels * log_softmax(logits)), where
 *          log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
 *
 * Three passes per row, for numerical stability:
 * 1. Find max(logits) to prevent overflow in exp().
 * 2. Compute sum_exp = sum(exp(logits - max)).
 * 3. Compute loss = -sum(labels * ((logits - max) - log(sum_exp))).
 *
 * The rows' losses are added to @p loss_out[0] one at a time, in row order.
 *
 * @param[in]     logits   rows x N logits (unnormalized scores), rows contiguous.
 * @param[in]     labels   rows x N labels (typically one-hot or probabilities).
 * @param[in,out] loss_out Single-element array: the running total, to which the rows' losses are
 *                         added.
 * @param[in]     N        Elements per row (arbitrary length).
 * @param[in]     rows     Number of rows.
 */
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t N,
                                int32_t rows) {
    event0();

    // 16-lane vectors, with the last N % 16 elements of a row in vectors too: no scalar float
    // compare or multiply, each a runtime call on the scalar unit. Lanes past the row are excluded
    // from the max and the exp sum, and contribute label 0 times a finite value to the loss. Rows
    // go in groups of up to 16, so that one vector log serves the whole group: scalar_log's float
    // arithmetic is a run of runtime calls, and took half the time when it ran once per row.
    constexpr int32_t V = 16;
    using vec = aie::vector<float, V>;
    const int32_t n_whole = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V - 1});
    const int32_t n_tail = N - n_whole;
    const auto in_tail = aie::mask<V>::from_uint32((uint32_t{1} << n_tail) - 1);
    const vec neg_inf = aie::broadcast<float, V>(-std::numeric_limits<float>::infinity());
    const vec zeros = aie::zeros<float, V>();

    // The last N % 16 elements of the row at p, row r of the tile, lanes past them zero. One
    // vector load when the buffer holds a whole vector from there on (the lanes past the row are
    // the next rows'), else a bounce buffer, so nothing past the buffer is read: its neighbour may
    // be the other half of a ping-pong fifo.
    const auto load_tail = [&](const float * p, int32_t r) {
        if ((rows - r) * N - n_whole >= V) {
            return aie::select(zeros, aie::load_unaligned_v<V>(p + n_whole), in_tail);
        }
        return load_v_partial<V>(p + n_whole, n_tail);
    };

    using vec2 = aie::vector<float, 2 * V>;
    const auto in_tail2 = aie::mask<2 * V>::from_uint32(((uint32_t{1} << n_tail) - 1) |
                                                        (((uint32_t{1} << n_tail) - 1) << V));

    // max(logits) of the row at x_row, whose last N % 16 elements are x_tail.
    const auto max_of_row = [&](const float * x_row, const vec & x_tail) {
        vec vmax = neg_inf;
        for (int32_t i = 0; i < n_whole; i += V) {
            vmax = aie::max(vmax, aie::load_unaligned_v<V>(x_row + i));
        }
        vmax = aie::max(vmax, aie::select(neg_inf, x_tail, in_tail));
        return aie::reduce_max(vmax);
    };

    float total = loss_out[0];
    for (int32_t g = 0; g < rows; g += V) {
        const int32_t g_rows = rows - g < V ? rows - g : V;
        // Zero-initialized: for the last, partial group (g_rows < V) only the first g_rows
        // lanes are ever written before a later aie::load_v(row_*) reads all V. The lanes past
        // g_rows are masked out of every subsequent use, but reading them unwritten is still
        // undefined behavior; zeroing costs nothing here (the compiler elides it once the loop
        // below overwrites every lane for a full group) and removes the UB.
        alignas(aie::vector_decl_align) float row_max[V] = {};
        alignas(aie::vector_decl_align) float row_sum[V] = {};
        alignas(aie::vector_decl_align) float row_lse[V] = {};

        // Passes 1 and 2, two rows at a time: max(logits), then sum(exp(logits - max)). An exp is
        // one long dependency chain per vector; running the two rows' vectors through one 32-lane
        // call overlaps their chains (four rows in 64 lanes overflowed aie2's program memory). An
        // odd last row is paired with itself.
        for (int32_t j = 0; j < g_rows; j += 2) {
            const int32_t j1 = j + 1 < g_rows ? j + 1 : j;
            const float * xa = logits + (g + j) * N;
            const float * xb = logits + (g + j1) * N;
            const vec ta = n_tail > 0 ? load_tail(xa, g + j) : zeros;
            const vec tb = n_tail > 0 ? load_tail(xb, g + j1) : zeros;
            const float ma = max_of_row(xa, ta);
            const float mb = max_of_row(xb, tb);
            const vec2 vm = aie::concat(aie::broadcast<float, V>(ma), aie::broadcast<float, V>(mb));

            vec2 vsum = aie::zeros<float, 2 * V>();
            for (int32_t i = 0; i < n_whole; i += V) {
                vec2 x = aie::sub(
                    aie::concat(aie::load_unaligned_v<V>(xa + i), aie::load_unaligned_v<V>(xb + i)),
                    vm);
                vsum = aie::add(vsum, vec_exp_nonpositive<2 * V>(x));
            }
            if (n_tail > 0) {
                // Lanes past a row may be positive here, outside vec_exp_nonpositive's contract;
                // their results are dropped.
                vec2 x = aie::sub(aie::concat(ta, tb), vm);
                vsum = aie::add(vsum, aie::select(aie::zeros<float, 2 * V>(),
                                                  vec_exp_nonpositive<2 * V>(x), in_tail2));
            }
            row_max[j] = ma;
            row_sum[j] = aie::reduce_add(vsum.template extract<V>(0));
            row_max[j1] = mb;
            row_sum[j1] = aie::reduce_add(vsum.template extract<V>(1));
        }

        // log(sum_exp) of the group's rows at once. A sum is >= 1 (the max's own term is
        // exp(0)), so in vec_log's domain; lanes past the group's rows are given 1.
        const auto in_group = aie::mask<V>::from_uint32((uint32_t{1} << g_rows) - 1);
        aie::store_v(row_lse, vec_log<V>(aie::select(aie::broadcast<float, V>(1.0f),
                                                     aie::load_v<V>(row_sum), in_group)));

        // Pass 3 per row: loss = -sum(labels * ((logits - max) - log_sum_exp)).
        const float * x_row = logits + g * N;
        const float * l_row = labels + g * N;
        for (int32_t j = 0; j < g_rows; ++j, x_row += N, l_row += N) {
            const vec vrow_max = aie::broadcast<float, V>(row_max[j]);
            const vec vlse = aie::broadcast<float, V>(row_lse[j]);
            const auto row_term = [&](vec x, vec l) {
                return aie::mul(l, aie::sub(aie::sub(x, vrow_max), vlse))
                    .template to_vector<float>();
            };
            vec vloss = zeros;
            for (int32_t i = 0; i < n_whole; i += V) {
                vloss = aie::add(vloss, row_term(aie::load_unaligned_v<V>(x_row + i),
                                                 aie::load_unaligned_v<V>(l_row + i)));
            }
            if (n_tail > 0) {
                vloss = aie::add(vloss, row_term(load_tail(x_row, g + j), load_tail(l_row, g + j)));
            }
            total = total + -aie::reduce_add(vloss);
        }
    }
    loss_out[0] = total;

    event1();
}

} // extern "C"
