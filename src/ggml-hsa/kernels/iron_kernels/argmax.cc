// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file argmax.cc
 * @brief Argmax operation for AIE kernels.
 */

#include <bit>
#include <limits>
#include <type_traits>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

namespace {

constexpr int32_t V = 16; // int32/float lanes per 512-bit vector, on aie2 and aie2p
using keys_t = aie::vector<int32_t, V>;

/**
 * @brief ggml_vec_argmax_f32 on one row, one element at a time, on integer keys.
 *
 * The AIE scalar unit has no float compare: `a > b` and `a == b` on float are runtime calls
 * (__gtsf2, __eqsf2), so the reference's loop is evaluated on order keys instead
 * (@ref float_order_key), with the same result:
 *
 * - The key orders non-NaN floats exactly as `<` does, and maps -0 and +0 both to 0, as a tie
 *   must (-0 == +0).
 * - In that order, "max = MAX(max, x); if (max == x) idx = i" takes index i exactly when
 *   !(max > x), i.e. key(x) >= key(max), for non-NaN max and x.
 * - NaN is tracked separately. A NaN x makes the maximum NaN and never takes the index
 *   (NaN == NaN is false); a non-NaN x after a NaN maximum always takes it (NaN > x is false,
 *   so MAX returns x, and x == x).
 *
 * The -infinity seed is key(-inf), non-NaN. Only rows holding a NaN come here (see
 * ggml_op_argmax), so this loop is not tuned.
 */
int32_t argmax_row_scalar(const float * __restrict in, int32_t N) {
    int32_t max_key = float_order_key(-std::numeric_limits<float>::infinity());
    bool max_is_nan = false;
    int32_t argmax_idx = 0;
    for (int32_t i = 0; i < N; ++i) {
        const bool is_nan = float_is_nan(in[i]);
        const int32_t key = float_order_key(in[i]);
        const bool take = !is_nan && (max_is_nan || key >= max_key);
        max_key = take ? key : max_key;
        argmax_idx = take ? i : argmax_idx;
        // A NaN maximum is always displaced by the next non-NaN, so it is NaN iff x was.
        max_is_nan = is_nan;
    }
    return argmax_idx;
}

/**
 * @brief Per-lane state of a row's search in vectors: lane l sees columns l, l + V, l + 2V, ...
 *
 * For a row without NaN the reference returns the last index holding the row's maximum: the
 * running maximum equals the row's maximum from its first occurrence on, every later occurrence
 * compares equal and takes the index, and no element after the last occurrence equals it. So
 * each lane keeps the largest key it has seen and the last column holding it (taking on >=,
 * like the reference), and the row's answer is the largest column among the lanes whose key is
 * the row's largest. A row holding a NaN is flagged instead and handed to the scalar loop: a
 * NaN discards the maximum before it, which the per-lane search cannot see.
 */
struct row_search {
    keys_t max_key = aie::broadcast<int32_t, V>(std::numeric_limits<int32_t>::min());
    keys_t max_col = aie::broadcast<int32_t, V>(-1);
    aie::mask<V> nan = aie::mask<V>::from_uint32(uint32_t{0});

    /// Folds in columns col .. col + V - 1, values v, of which only the lanes in valid exist.
    void step(const aie::vector<float, V> & v, const keys_t & col, const aie::mask<V> & valid) {
        const keys_t key = float_order_keys<V>(v);
        nan = nan | (floats_are_nan<V>(v) & valid);
        // Every valid lane takes its first column: no key is below INT32_MIN.
        const auto take = aie::ge(key, max_key) & valid;
        max_key = aie::select(max_key, key, take);
        max_col = aie::select(max_col, col, take);
    }

    /// The row's argmax, or -1 if the row holds a NaN.
    int32_t result() const {
        const int32_t row_max = aie::reduce_max(max_key);
        const int32_t col = aie::reduce_max(
            aie::select(aie::broadcast<int32_t, V>(-1), max_col, aie::eq(max_key, row_max)));
        return nan.empty() ? col : -1;
    }
};

} // namespace

extern "C" {

/**
 * @brief Finds the index of the maximum value in each row of a tile.
 *
 * Matches @c ggml_vec_argmax_f32 (ggml-cpu/vec.h) exactly. The reference is
 * @c max = MAX(max, x[i]) followed by @c if @c (max @c == @c x[i]) @c idx @c = @c i, with
 * @c MAX(a,b) = @c (a @c > @c b @c ? @c a @c : @c b) and @c max seeded with -infinity. Two
 * consequences, both of which callers have to live with because they are the reference's:
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
 * Rows are searched in vectors (@ref row_search) and those holding a NaN again one element at a
 * time (@ref argmax_row_scalar). Rows are packed back to back and need not start vector-aligned,
 * so loads are unaligned; a load may run past its row into the next, whose lanes are ignored,
 * but never past the tile.
 *
 * @param[in]  in       num_rows rows of N elements each.
 * @param[out] out      Index of each row's max element.
 * @param[in]  N        Number of elements per row.
 * @param[in]  num_rows Number of rows.
 */
void ggml_op_argmax(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t N,
                    int32_t num_rows) {
    event0();

    // See test-argmax-hsa for the rows that pin the reference's tie and NaN behaviour down.
    //
    // The -inf seed is why this kernel is floating-point only: for an integer INPUT_DTYPE
    // numeric_limits<T>::infinity() is 0, so an all-negative row would never update the running
    // maximum and would report index 0. argmax.py rejects non-f32 input, and this catches it at
    // compile time if that check is ever loosened. numeric_limits<T>::lowest() is not a fix --
    // it would break the all- -inf row, which the reference resolves to the last index.
    static_assert(std::numeric_limits<INPUT_DTYPE>::has_infinity,
                  "ggml_op_argmax seeds its running maximum with -infinity to match "
                  "ggml_vec_argmax_f32; INPUT_DTYPE must be a floating-point type");
    static_assert(std::is_same_v<INPUT_DTYPE, float>,
                  "ggml_op_argmax compares float bit patterns as int32");

    alignas(aie::vector_decl_align) static constexpr int32_t lane_index[V] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const keys_t lanes = aie::load_v<V>(lane_index);
    const int32_t tile_elems = N * num_rows;
    const auto all_lanes = aie::lt(lanes, V);

    // Row r's search, its loads reading whole vectors only while V elements remain in the tile.
    const auto search_row = [&](int32_t r) {
        const float * __restrict row = in + r * N;
        row_search s;
        int32_t c = 0;
        for (; c + V <= N; c += V) {
            s.step(aie::load_unaligned_v<V>(row + c), aie::add(lanes, c), all_lanes);
        }
        if (c < N) {
            const int32_t left = tile_elems - r * N - c;
            const auto v =
                left >= V ? aie::load_unaligned_v<V>(row + c) : load_v_partial<V>(row + c, left);
            s.step(v, aie::add(lanes, c), aie::lt(lanes, N - c));
        }
        return s.result();
    };

    if (N <= V) {
        // One vector per row, so the row loop is the innermost loop, for the rows whose vector
        // ends inside the tile (r * N + V <= tile_elems): all but ceil(V / N) - 1 at most. A
        // lane's column is its lane, so @ref row_search reduces to the highest lane holding the
        // row's largest key, which the mask's leading zeros give without a second reduction (the
        // invalid lanes' INT32_MIN is below every float's key, so some valid lane holds it).
        int32_t whole = num_rows;
        while (whole > 0 && (whole - 1) * N + V > tile_elems) {
            --whole;
        }
        const auto valid = aie::lt(lanes, N);
        const auto row_vec = [&](int32_t r) {
            const auto v = aie::load_unaligned_v<V>(in + r * N);
            const keys_t key =
                aie::select(aie::broadcast<int32_t, V>(std::numeric_limits<int32_t>::min()),
                            float_order_keys<V>(v), valid);
            const bool nan = !(floats_are_nan<V>(v) & valid).empty();
            const int32_t row_max = aie::reduce_max(key);
            const uint32_t at_max = aie::eq(key, row_max).to_uint32();
            const int32_t col = 31 - __builtin_clz(at_max);
            out[r] = nan ? -1 : col;
        };
        GGML_AIE_LOOP_MIN_TRIPS(4, r, whole, 1, row_vec(r);)
        for (int32_t r = whole; r < num_rows; ++r) {
            out[r] = search_row(r);
        }
    } else {
        for (int32_t r = 0; r < num_rows; ++r) {
            out[r] = search_row(r);
        }
    }

    // Rows holding a NaN.
    for (int32_t r = 0; r < num_rows; ++r) {
        if (out[r] < 0) {
            out[r] = argmax_row_scalar(in + r * N, N);
        }
    }

    event1();
}

} // extern "C"
