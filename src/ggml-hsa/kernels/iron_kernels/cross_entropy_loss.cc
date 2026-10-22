// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

#if __AIEARCH__ != 20 && defined(GGML_CEL_ROW_LEN) && GGML_CEL_ROW_LEN <= 16

// aie2p, rows of up to 16 elements (MNIST's 10 classes): one row per lane, 64 rows a group, so that
// no step reduces across lanes, and every f32 product on bf16 limbs. aie2p has no f32 multiplier:
// an f32 aie::mul splits both operands into three bf16 limbs and sums nine limb products 32 lanes
// at a time, and f32 min/max are emulated too, while one bf16 multiply-accumulate takes 64 lanes
// into an f32 accumulator, exactly. The limb helpers and exp() are softmax.cc's aie2p ones.

#include "aie_kernel_utils.h"

namespace {

/// Elements per row.
constexpr int32_t N = GGML_CEL_ROW_LEN;

/// Lanes (rows) of a group.
constexpr int32_t W = 64;

using fvec = aie::vector<float, W>;
using bvec = aie::vector<bfloat16, W>;
using ivec = aie::vector<int32_t, W>;
using acc_t = aie::accum<accfloat, W>;

/// The bf16 limbs of an f32 value, h + m + l (|m| <= 2^-9 |h|, |l| <= 2^-9 |m|).
struct limbs {
    bvec h, m, l;
};

__attribute__((always_inline)) inline acc_t to_acc(const fvec & v) {
    acc_t a;
    a.from_vector(v);
    return a;
}

/// Splits @p r into bf16 limbs, each from the exact residual of the ones before (an accumulator
/// subtraction, which is exact where aie2p's vadd.f is not). Only @p NL limbs are taken.
template <int NL = 3>
__attribute__((always_inline)) inline limbs split(acc_t r) {
    const bvec one = aie::broadcast<bfloat16, W>(1.0f);
    limbs s;
    s.h = r.template to_vector<bfloat16>();
    if constexpr (NL > 1) {
        r = aie::msc(r, s.h, one);
        s.m = r.template to_vector<bfloat16>();
    }
    if constexpr (NL > 2) {
        r = aie::msc(r, s.m, one);
        s.l = r.template to_vector<bfloat16>();
    }
    return s;
}

/// p + a * b from the six limb products above 2^-24 of a * b, smallest first.
__attribute__((always_inline)) inline acc_t mac_limbs(acc_t p, const limbs & a, const limbs & b) {
    p = aie::mac(p, a.l, b.h);
    p = aie::mac(p, a.m, b.m);
    p = aie::mac(p, a.h, b.l);
    p = aie::mac(p, a.m, b.h);
    p = aie::mac(p, a.h, b.m);
    return aie::mac(p, a.h, b.h);
}

/// a * b from the six limb products above 2^-24 of it, smallest first.
__attribute__((always_inline)) inline acc_t mul_limbs(const limbs & a, const limbs & b) {
    acc_t p = aie::mul(a.l, b.h);
    p = aie::mac(p, a.m, b.m);
    p = aie::mac(p, a.h, b.l);
    p = aie::mac(p, a.m, b.h);
    p = aie::mac(p, a.h, b.m);
    return aie::mac(p, a.h, b.h);
}

/**
 * @brief exp(x) for the x <= 0 in @p x, 64 lanes; lanes below -86 (and -inf) give 0. softmax.cc's
 * aie2p exp_nonpositive: see there.
 */
__attribute__((always_inline)) inline fvec exp_nonpositive(const acc_t & x) {
    constexpr float kLog2eHi = 1.4453125f;
    constexpr float kLog2eLo = -0.00262451171875f;
    constexpr float kLn2[] = {0.69140625f, 0.001739501953125f, 1.430511474609375e-06f,
                              -1.9063008949160576e-09f};
    constexpr float kMagic = 12582912.0f; // 1.5 * 2^23
    constexpr float kC2 = 0x1.fffd94p-2f;
    constexpr float kC3 = 0x1.555438p-3f;
    constexpr float kC4 = 0x1.57365cp-5f;
    constexpr float kC5 = 0x1.125696p-7f;
    const auto preload = [](float c) { return acc_t(aie::broadcast<float, W>(c)); };

    const limbs xs = split<2>(x);
    const auto underflow = aie::lt(xs.h, aie::broadcast<bfloat16, W>(-86.0f));

    acc_t t = aie::mul(xs.m, static_cast<bfloat16>(kLog2eHi));
    t = aie::mac(t, xs.h, static_cast<bfloat16>(kLog2eLo));
    t = aie::mac(t, xs.h, static_cast<bfloat16>(kLog2eHi));
    t = aie::add(t, kMagic);
    const fvec tm = t.template to_vector<float>();
    const bvec n = aie::add(t, -kMagic).template to_vector<bfloat16>();

    acc_t r = x;
    AIE_LOOP_UNROLL_FULL
    for (float c : kLn2) {
        r = aie::msc(r, n, static_cast<bfloat16>(c));
    }
    const limbs rs = split(r);

    const bfloat16 c5 = static_cast<bfloat16>(kC5);
    acc_t p = aie::mac(preload(kC4), rs.m, c5);
    p = aie::mac(p, rs.h, c5);
    limbs ps = split<2>(p);
    p = aie::mac(preload(kC3), rs.m, ps.h);
    p = aie::mac(p, rs.h, ps.m);
    p = aie::mac(p, rs.h, ps.h);
    ps = split<2>(p);
    p = aie::mac(preload(kC2), rs.m, ps.h);
    p = aie::mac(p, rs.h, ps.m);
    p = aie::mac(p, rs.h, ps.h);
    ps = split(p);
    p = aie::add(mul_limbs(rs, ps), 1.0f);
    ps = split(p);
    p = aie::add(mul_limbs(rs, ps), 1.0f);

    const ivec bits = aie::add(aie::vector_cast<int32_t>(p.template to_vector<float>()),
                               aie::upshift(aie::vector_cast<int32_t>(tm), 23));
    return aie::select(aie::vector_cast<float>(bits), aie::zeros<float, W>(), underflow);
}

/// A group's rows transposed: column c of its rows at [c * W, (c + 1) * W). The logits have a
/// column of -inf past the row for odd lengths (exp() runs two columns at a time).
alignas(aie::vector_decl_align) float logits_columns[(N + 1) / 2 * 2 * W];
alignas(aie::vector_decl_align) float labels_columns[N * W];

} // namespace

extern "C" {

/**
 * @brief Adds the cross-entropy loss of @p rows rows of GGML_CEL_ROW_LEN elements to
 * @p loss_out[0]: see the aie2 version below for the passes. Each group of 64 rows is transposed
 * in, one row per lane, and the rows' losses are summed per lane across the groups, then across
 * the lanes, and subtracted from @p loss_out[0].
 */
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t /* N */,
                                int32_t rows) {
    event0();

    aie::scoped_mode rnd{aie::rounding_mode::conv_even};
    const float * x_cols = logits_columns;
    const float * l_cols = labels_columns;

    if constexpr (N % 2 != 0) {
        aie::store_v(logits_columns + N * W,
                     aie::broadcast<float, W>(-std::numeric_limits<float>::infinity()));
    }

    fvec total = aie::zeros<float, W>();
    for (int32_t r0 = 0; r0 < rows; r0 += W) {
        // In the last group, lanes past its rows hold earlier data: every step is lane-wise, and
        // they are left out of the sum.
        const int32_t g_rows = rows - r0 < W ? rows - r0 : W;

        // Transpose in, a column at a time. The column pointers are restricted only here: the
        // passes read the columns through x_cols and l_cols.
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const float * __restrict xs = logits + r0 * N + c;
            const float * __restrict ls = labels + r0 * N + c;
            float * __restrict xd = logits_columns + c * W;
            float * __restrict ld = labels_columns + c * W;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, g_rows, 1, {
                *xd++ = *xs;
                *ld++ = *ls;
                xs += N;
                ls += N;
            })
        }

        // Pass 1: each row's max, on the bit patterns: the running signed-integer max when it is
        // >= 0 (a non-negative float), else the min (negative floats order backwards as
        // integers).
        ivec kmax = aie::vector_cast<int32_t>(aie::load_v<W>(x_cols));
        ivec kmin = kmax;
        AIE_LOOP_UNROLL_FULL
        for (int32_t c = 1; c < N; ++c) {
            const ivec k = aie::vector_cast<int32_t>(aie::load_v<W>(x_cols + c * W));
            kmax = aie::max(kmax, k);
            kmin = aie::min(kmin, k);
        }
        const fvec vmax = aie::vector_cast<float>(
            aie::select(kmin, kmax, aie::ge(kmax, aie::zeros<int32_t, W>())));
        const acc_t neg_max = to_acc(vec_neg(vmax));

        // Pass 2: sum(exp(logits - max)), two columns a step: their chains are independent. The
        // column past an odd row length is -inf, whose exp() is 0.
        acc_t sum = aie::zeros<accfloat, W>();
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; c += 2) {
            const fvec e0 = exp_nonpositive(aie::add(to_acc(aie::load_v<W>(x_cols + c * W)), neg_max));
            const fvec e1 =
                exp_nonpositive(aie::add(to_acc(aie::load_v<W>(x_cols + (c + 1) * W)), neg_max));
            sum = aie::add(aie::add(sum, e0), e1);
        }

        // log(sum_exp), >= 1 in the group's rows (the max's own term is exp(0)).
        const fvec s = sum.template to_vector<float>();
        const fvec lse = aie::concat(vec_log<W / 2>(s.template extract<W / 2>(0)),
                                     vec_log<W / 2>(s.template extract<W / 2>(1)));

        // Pass 3: sum(labels * ((logits - max) - log_sum_exp)).
        acc_t loss = aie::zeros<accfloat, W>();
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const fvec d = aie::sub(aie::sub(aie::load_v<W>(x_cols + c * W), vmax), lse);
            loss = mac_limbs(loss, split(to_acc(aie::load_v<W>(l_cols + c * W))), split(to_acc(d)));
        }
        const auto in_group = aie::mask<W>::from_uint64(
            g_rows == W ? ~uint64_t{0} : (uint64_t{1} << g_rows) - 1);
        total = aie::add(total, aie::select(aie::zeros<float, W>(),
                                            loss.template to_vector<float>(), in_group));
    }
    loss_out[0] = loss_out[0] - aie::reduce_add(total);

    event1();
}

} // extern "C"

#else // __AIEARCH__ == 20, or rows longer than 16

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
        alignas(aie::vector_decl_align) float row_max[V];
        alignas(aie::vector_decl_align) float row_sum[V];
        alignas(aie::vector_decl_align) float row_lse[V];

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

#endif // __AIEARCH__ != 20 && GGML_CEL_ROW_LEN <= 16
