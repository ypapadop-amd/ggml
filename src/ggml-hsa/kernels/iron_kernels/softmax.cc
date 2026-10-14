// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <stdint.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

/**
 * @brief Computes the ALiBi (Attention with Linear Biases) slope for a head.
 *
 * ALiBi applies position-dependent biases in attention using geometric slopes.
 * The slope for each head is computed based on the head index and total heads.
 *
 * For heads 0 to n_head_log2-1: slope = m0^(head_idx+1)
 * For heads >= n_head_log2:     slope = m1^(2*(head_idx - n_head_log2) + 1)
 *
 * where m0 = 2^(-max_bias/n_head_log2) and m1 = 2^(-max_bias/2/n_head_log2)
 *
 * @param[in] max_bias  Maximum bias value. If <= 0, returns 1.0.
 * @param[in] n_head    Total number of attention heads.
 * @param[in] head_idx  Index of the current head (0-based).
 *
 * @return The computed ALiBi slope for this head.
 */
inline float compute_alibi_slope(float max_bias, int32_t n_head, int32_t head_idx) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    const uint32_t n_head_log2 = 1u << floor_log2(static_cast<uint32_t>(n_head));
    const auto m0 = pow2(-max_bias / n_head_log2);
    const auto m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (static_cast<uint32_t>(head_idx) < n_head_log2) {
        // slope = m0^(head_idx+1) via repeated multiplication
        slope = m0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(head_idx); ++j) {
            slope *= m0;
        }
    } else {
        // slope = m1^(2*(head_idx - n_head_log2) + 1)
        uint32_t exp = 2 * (static_cast<uint32_t>(head_idx) - n_head_log2) + 1;
        slope = m1;
        for (uint32_t j = 1; j < exp; ++j) {
            slope *= m1;
        }
    }

    return slope;
}

#ifdef GGML_OP_SOFT_MAX

namespace {

#if __AIEARCH__ == 20 // aie2p has its own passes below: see there.

/// Softmax of one row of @p N elements: see ggml_op_soft_max.
inline void
soft_max_row(const float * __restrict input, float * __restrict output, int32_t N, float scale) {
    // The scalar passes below call a runtime routine for every float multiply and compare (the
    // scalar unit has neither), so they run in vectors: whole vectors in place, and the last
    // N % V elements through a bounce buffer, so nothing past the row is read or written (the
    // row's neighbour may be the other half of a ping-pong ObjectFifo). Rows need not start
    // 512-bit aligned, so whole vectors use the unaligned accesses.
    constexpr int32_t V = 16;
    using vec = aie::vector<float, V>;

    const int32_t n_whole = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V - 1});
    const int32_t n_tail = N - n_whole;
    const auto in_tail = aie::mask<V>::from_uint32((uint32_t{1} << n_tail) - 1);
    const vec neg_inf = aie::broadcast<float, V>(-std::numeric_limits<float>::infinity());

    // x * 1.0f is x, and the emulated fp32 vector multiply is both slow and not exactly rounded,
    // so a unit scale (plain ggml_soft_max) skips it.
    const bool unit_scale = std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f);
    const auto scaled = [&](vec x) {
        return unit_scale ? x : aie::mul(x, scale).template to_vector<float>();
    };

    // The tail is read once and kept in registers through all three passes.
    const vec x_tail =
        n_tail > 0 ? load_v_partial<V>(input + n_whole, n_tail) : aie::zeros<float, V>();

    // Pass 1: max of the scaled row (see the scalar passes for why it precedes exp()). Rounding
    // scale*x is monotone in x, so max(scale*x) is scale*max(x) for scale >= 0 and scale*min(x)
    // for scale < 0, exactly: the row is reduced unscaled and scaled once.
    vec vmax = neg_inf;
    vec vmin = aie::neg(neg_inf);
    for (int32_t i = 0; i < n_whole; i += V) {
        const vec x = aie::load_unaligned_v<V>(input + i);
        vmax = aie::max(vmax, x);
        vmin = aie::min(vmin, x);
    }
    if (n_tail > 0) {
        vmax = aie::max(vmax, aie::select(neg_inf, x_tail, in_tail));
        vmin = aie::min(vmin, aie::select(aie::neg(neg_inf), x_tail, in_tail));
    }
    const vec vextreme = aie::broadcast<float, V>(
        std::bit_cast<int32_t>(scale) < 0 ? aie::reduce_min(vmin) : aie::reduce_max(vmax));
    const vec vrow_max = scaled(vextreme);

    // Pass 2: exp(scale*x - max) into out, and its sum (tail lanes past the row excluded). Each
    // exp() call takes kExpVectors vectors (see vec_exp_nonpositive); in the last call, lanes past
    // the row are computed and dropped.
    constexpr int32_t kExpVectors = 4;
    vec vsum = aie::zeros<float, V>();
    vec e_tail = aie::zeros<float, V>();
    for (int32_t i = 0; i < N; i += kExpVectors * V) {
        vec x[kExpVectors];
        AIE_LOOP_UNROLL_FULL
        for (int32_t k = 0; k < kExpVectors; ++k) {
            const int32_t j = i + k * V;
            x[k] = aie::sub(scaled(j < n_whole ? aie::load_unaligned_v<V>(input + j) : x_tail),
                            vrow_max);
        }
        const aie::vector<float, kExpVectors * V> e =
            vec_exp_nonpositive<kExpVectors * V>(aie::concat(x[0], x[1], x[2], x[3]));
        AIE_LOOP_UNROLL_FULL
        for (int32_t k = 0; k < kExpVectors; ++k) {
            const int32_t j = i + k * V;
            if (j < n_whole) {
                store_v_any(output + j, e.template extract<V>(k));
                vsum = aie::add(vsum, e.template extract<V>(k));
            } else if (j < N) {
                e_tail = aie::select(aie::zeros<float, V>(), e.template extract<V>(k), in_tail);
                vsum = aie::add(vsum, e_tail);
            }
        }
    }

    // Pass 3: normalize. aie::inv on fp32 ends in a Newton step of its own (see
    // soft_max_lane_rows); it replaces a runtime divide.
    const vec sum_inv = aie::inv(aie::broadcast<float, V>(aie::reduce_add(vsum)));
    // Its stores are a whole vector apart, so at the row's alignment, tested once (see
    // store_v_any); pass 2's loop, which is not pipelined, tests every store.
    const auto normalize = [&](auto store) {
        for (int32_t i = 0; i < n_whole; i += V) {
            store_v_any<decltype(store)::value>(
                output + i, aie::mul(aie::load_unaligned_v<V>(output + i), sum_inv)
                                .template to_vector<float>());
        }
    };
    if (store_v_any_aligned<V>(output)) {
        normalize(std::integral_constant<store_alignment, store_alignment::aligned>{});
    } else {
        normalize(std::integral_constant<store_alignment, store_alignment::unaligned>{});
    }
    if (n_tail > 0) {
        store_v_partial<V>(output + n_whole, aie::mul(e_tail, sum_inv).template to_vector<float>(),
                           n_tail);
    }
}

#if defined(GGML_SOFTMAX_ROW_LEN) && GGML_SOFTMAX_ROW_LEN <= 32
#define GGML_SOFTMAX_LANE_ROWS 1

// Rows this short leave most lanes of a row's vectors idle, and a row is one serial chain of
// dependent vector steps (two reductions, exp(), a divide) that nothing else overlaps. So
// kLaneRows rows run together with one row per lane: column c of the group is one vector, each
// pass is lane-wise and needs no reduction, exp() runs on full vectors, and the group shares one
// reciprocal.
constexpr int32_t kLaneRows = 32;

/// Columns per exp() call: two 32-lane columns as one 64-lane vector, whose halves the scheduler
/// interleaves (see vec_exp_nonpositive).
constexpr int32_t kExpColumns = 2;

/// Columns, rounded up to whole exp() calls.
constexpr int32_t kLaneRowsColumns =
    (GGML_SOFTMAX_ROW_LEN + kExpColumns - 1) / kExpColumns * kExpColumns;

/// The group transposed: column c of its rows at [c * kLaneRows, (c + 1) * kLaneRows). A column
/// past the row (odd lengths) is computed and never read.
alignas(aie::vector_decl_align) float lane_rows_columns[kLaneRowsColumns * kLaneRows];

/// Softmax of each of @p n_rows rows of GGML_SOFTMAX_ROW_LEN elements: see ggml_op_soft_max.
inline void soft_max_lane_rows(const float * __restrict input,
                               float * __restrict output,
                               int32_t n_rows,
                               float scale) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    constexpr int32_t L = kLaneRows;
    using vec = aie::vector<float, L>;
    float * __restrict cols = lane_rows_columns;

    // As in soft_max_row: a unit scale skips the emulated fp32 multiply, and the row max is the
    // scaled maximum (or, for a negative scale, minimum) of the unscaled row.
    const bool unit_scale = std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f);
    const bool negative_scale = std::bit_cast<int32_t>(scale) < 0;
    const auto scaled = [&](vec x) {
        return unit_scale ? x : aie::mul(x, scale).template to_vector<float>();
    };

    for (int32_t r0 = 0; r0 < n_rows; r0 += L) {
        // In the last group, lanes past its rows hold the previous group's columns (or whatever
        // the buffer held): every pass is lane-wise, so they are computed and never stored.
        const int32_t rows = n_rows - r0 < L ? n_rows - r0 : L;

        // Transpose in, a column at a time, as 32-bit words: no float operation touches the data.
        uint32_t * __restrict cols_bits = reinterpret_cast<uint32_t *>(cols);
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict src =
                reinterpret_cast<const uint32_t *>(input + r0 * N) + c;
            uint32_t * __restrict col = cols_bits + c * L;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *col++ = *src;
                src += N;
            })
        }

        // Pass 1: the extreme of each row (lane), over its columns.
        vec extreme = aie::load_v<L>(cols);
        if (negative_scale) {
            for (int32_t c = 1; c < N; ++c) {
                extreme = aie::min(extreme, aie::load_v<L>(cols + c * L));
            }
        } else {
            for (int32_t c = 1; c < N; ++c) {
                extreme = aie::max(extreme, aie::load_v<L>(cols + c * L));
            }
        }
        const vec row_max = scaled(extreme);

        // Pass 2: exp(scale*x - max) in place, two columns at a time, and each row's sum.
        vec sum = aie::zeros<float, L>();
        for (int32_t c = 0; c < N; c += kExpColumns) {
            const vec x0 = aie::sub(scaled(aie::load_v<L>(cols + c * L)), row_max);
            const vec x1 = aie::sub(scaled(aie::load_v<L>(cols + (c + 1) * L)), row_max);
            const aie::vector<float, kExpColumns * L> e =
                vec_exp_nonpositive<kExpColumns * L>(aie::concat(x0, x1));
            aie::store_v(cols + c * L, e);
            sum = aie::add(sum, e.template extract<L>(0));
            if (c + 1 < N) {
                sum = aie::add(sum, e.template extract<L>(1));
            }
        }

        // Pass 3: normalize by 1/sum. aie::inv on fp32 ends in a Newton step of its own: a
        // further one moved the outputs' error vs a double reference by at most 2e-7.
        const vec sum_inv = aie::inv(sum);
        for (int32_t c = 0; c < N; ++c) {
            aie::store_v(
                cols + c * L,
                aie::mul(aie::load_v<L>(cols + c * L), sum_inv).template to_vector<float>());
        }

        // Transpose out, a column at a time.
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict col = cols_bits + c * L;
            uint32_t * __restrict dst = reinterpret_cast<uint32_t *>(output + r0 * N) + c;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *dst = *col++;
                dst += N;
            })
        }
    }
}
#endif

#else // __AIEARCH__ != 20

// aie2p: 64 lanes a step, and every f32 product on bf16 limbs. aie2p has no f32 multiplier: an f32
// aie::mul splits both operands into three bf16 limbs and sums nine limb products 32 lanes at a
// time, and f32 min/max are emulated too, while one bf16 multiply-accumulate takes 64 lanes into
// an f32 accumulator, exactly (a bf16 product is exact in f32). So exp() is a polynomial whose
// products take only the limbs their size calls for, and maxima are taken on integer bit patterns.
// Conversions to bf16 round to nearest even.

/// Lanes of a step.
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
/// subtraction, which is exact where aie2p's vadd.f is not). Only @p N limbs are taken.
template <int N = 3>
__attribute__((always_inline)) inline limbs split(acc_t r) {
    const bvec one = aie::broadcast<bfloat16, W>(1.0f);
    limbs s;
    s.h = r.template to_vector<bfloat16>();
    if constexpr (N > 1) {
        r = aie::msc(r, s.h, one);
        s.m = r.template to_vector<bfloat16>();
    }
    if constexpr (N > 2) {
        r = aie::msc(r, s.m, one);
        s.l = r.template to_vector<bfloat16>();
    }
    return s;
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

/// Broadcasts an f32 constant split into limbs.
__attribute__((always_inline)) inline limbs split_scalar(float f) {
    return split(to_acc(aie::broadcast<float, W>(f)));
}

/**
 * @brief exp(x) for the x <= 0 in @p x, 64 lanes; lanes below -86 (and -inf) give 0.
 *
 * exp(x) = 2^n exp(r): n = round(x log2(e)) from two-limb products, r = x - n ln(2) with ln(2) in
 * four bf16 limbs (n is exact in bf16, so each n * limb is exact and the first subtraction
 * cancels exactly), |r| <= 0.348. exp(r) is a degree-5 polynomial (the Chebyshev fit on
 * [-0.3476, 0.3476], within 2.4e-7) by Horner's rule, each step c + r p from as many limbs of r
 * and p as the step's share of the result needs (two products at the top, six at the bottom).
 * The last two steps add c after the products, so it is rounded once with them. 2^n is added to
 * the exponent field. Within 3.5e-7 of exp(x) in a bit model over 1.1M arguments (the CPU's expf
 * is within ~1e-7). x down to -86 keeps 2^n exp(r) a normal float; below it the lanes are set to
 * 0 (the CPU gives at most 4.5e-38 there).
 *
 * @param[in] x Arguments, in an accumulator: x <= 0, -inf, or a tiny positive rounding of 0.
 */
__attribute__((always_inline)) inline fvec exp_nonpositive(const acc_t & x) {
    // log2(e) and ln(2) as sums of bf16 values.
    constexpr float kLog2eHi = 1.4453125f;
    constexpr float kLog2eLo = -0.00262451171875f;
    constexpr float kLn2[] = {0.69140625f, 0.001739501953125f, 1.430511474609375e-06f,
                              -1.9063008949160576e-09f};
    constexpr float kMagic = 12582912.0f; // 1.5 * 2^23
    // The polynomial's coefficients from degree 0; c0 and c1 round to 1.
    constexpr float kC2 = 0x1.fffd94p-2f;
    constexpr float kC3 = 0x1.555438p-3f;
    constexpr float kC4 = 0x1.57365cp-5f;
    constexpr float kC5 = 0x1.125696p-7f;
    const auto preload = [](float c) { return acc_t(aie::broadcast<float, W>(c)); };

    const limbs xs = split<2>(x);
    const auto underflow = aie::lt(xs.h, aie::broadcast<bfloat16, W>(-86.0f));

    // n = round(x log2(e)): its error, below 2^-15 |x log2(e)|, leaves |r| <= ln(2)/2 + 0.0003.
    // tm = n + 1.5 * 2^23 holds n in its low bits.
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

    // Horner's rule, p_k = c_k + r p_{k+1} from p_5 = c_5.
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

    // p 2^n: n << 23 added to the exponent field (the magic number's bits shift out).
    const ivec bits = aie::add(aie::vector_cast<int32_t>(p.template to_vector<float>()),
                               aie::upshift(aie::vector_cast<int32_t>(tm), 23));
    return aie::select(aie::vector_cast<float>(bits), aie::zeros<float, W>(), underflow);
}

/// exp_nonpositive of two vectors: their chains are independent, and one alone leaves the core
/// waiting on each step's latency.
struct fvec_pair {
    fvec a, b;
};

__attribute__((always_inline)) inline fvec_pair exp_nonpositive(const acc_t & x0,
                                                                const acc_t & x1) {
    return {exp_nonpositive(x0), exp_nonpositive(x1)};
}

/// exp(x0 - m), exp(x1 - m), out of line: for the vectors at a row's ends, so that each row's
/// passes inline exp() once.
__attribute__((noinline)) fvec_pair exp_nonpositive_call(const fvec & x0,
                                                         const fvec & x1,
                                                         float m) {
    return exp_nonpositive(aie::add(to_acc(x0), -m), aie::add(to_acc(x1), -m));
}

/// The largest float of each lane from the running signed-integer max and min of its bit
/// patterns: the max when any is >= 0 (a non-negative float), else the min (negative floats order
/// backwards as integers). Exact for every non-NaN float; -0 and +0 compare as the CPU's max does
/// not, but give the same exp(x - max).
__attribute__((always_inline)) inline ivec lane_max(const ivec & kmax, const ivec & kmin) {
    return aie::select(kmin, kmax, aie::ge(kmax, aie::zeros<int32_t, W>()));
}

/// x * s, rounded once.
__attribute__((always_inline)) inline fvec scaled(const fvec & x, const limbs & s) {
    return mul_limbs(split(to_acc(x)), s).template to_vector<float>();
}

/**
 * @brief Softmax of one row of GGML_SOFTMAX_ROW_LEN elements: see ggml_op_soft_max.
 *
 * The row is a head up to the first 64-byte boundary of @p output (the tile buffers are 64-byte
 * aligned, so @p input's rows have the same offsets), whole 64-lane vectors, aligned, and a tail.
 * Head and tail are copied in and out an element at a time and held in vectors whose other lanes
 * are -inf, so exp() gives 0 there. No vector store is unaligned: with store_v_any's unaligned
 * stores, tiles of four 500-wide rows came back with the bytes a store shares a 32-byte block
 * with stale, where the next row or vector starts (not root-caused).
 *
 * A scale other than 1 (@p s, else null) is applied first, into out, which the passes then read:
 * they see the row the CPU does, scale * x rounded.
 */
__attribute__((always_inline)) inline void
soft_max_row(const float * input, float * output, const limbs * s) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    const int32_t first = static_cast<int32_t>((reinterpret_cast<uintptr_t>(output) >> 2) & 15);
    const int32_t n_head = std::min(N, (16 - first) & 15);
    const int32_t n_whole = (N - n_head) & ~(W - 1);
    const int32_t n_tail = N - n_head - n_whole;
    const fvec pad = aie::broadcast<float, W>(-std::numeric_limits<float>::infinity());
    const auto lanes = [](int32_t n) { return aie::mask<W>::from_uint64((uint64_t{1} << n) - 1); };
    fvec x_head = n_head > 0 ? load_v_partial<W>(input, n_head) : pad;
    fvec x_tail = n_tail > 0 ? load_v_partial<W>(input + n_head + n_whole, n_tail) : pad;
    // The whole vectors: from in, or for a scaled row from out, which is read and then rewritten a
    // vector at a time (so not __restrict).
    const float * x = input + n_head;
    float * out = output + n_head;
    if (s != nullptr) {
        for (int32_t i = 0; i < n_whole; i += W) {
            aie::store_v(out + i, scaled(aie::load_v<W>(x + i), *s));
        }
        x_head = scaled(x_head, *s);
        x_tail = scaled(x_tail, *s);
        x = out;
    }
    x_head = aie::select(pad, x_head, lanes(n_head));
    x_tail = aie::select(pad, x_tail, lanes(n_tail));

    // Pass 1: the row's max, on the bit patterns.
    ivec kmax = aie::max(aie::vector_cast<int32_t>(x_head), aie::vector_cast<int32_t>(x_tail));
    ivec kmin = aie::min(aie::vector_cast<int32_t>(x_head), aie::vector_cast<int32_t>(x_tail));
    for (int32_t i = 0; i < n_whole; i += W) {
        const ivec k = aie::vector_cast<int32_t>(aie::load_v<W>(x + i));
        kmax = aie::max(kmax, k);
        kmin = aie::min(kmin, k);
    }
    const int32_t emax = aie::reduce_max(kmax);
    const float row_max = std::bit_cast<float>(emax >= 0 ? emax : aie::reduce_min(kmin));

    // Pass 2: exp(x - max) into out, and its sum, two whole vectors a step, then the last whole
    // vector of an odd count with the head, and the tail.
    acc_t sum = aie::zeros<accfloat, W>();
    int32_t i = 0;
    for (; i + W < n_whole; i += 2 * W) {
        const fvec_pair e = exp_nonpositive(aie::add(to_acc(aie::load_v<W>(x + i)), -row_max),
                                            aie::add(to_acc(aie::load_v<W>(x + i + W)), -row_max));
        aie::store_v(out + i, e.a);
        aie::store_v(out + i + W, e.b);
        sum = aie::add(aie::add(sum, e.a), e.b);
    }
    const bool odd = i < n_whole;
    fvec e_head = aie::zeros<float, W>();
    if (odd || n_head > 0) {
        const fvec_pair e =
            exp_nonpositive_call(odd ? aie::load_v<W>(x + i) : pad, x_head, row_max);
        if (odd) {
            aie::store_v(out + i, e.a);
        }
        e_head = e.b;
        sum = aie::add(aie::add(sum, e.a), e.b);
    }
    fvec e_tail = aie::zeros<float, W>();
    if (n_tail > 0) {
        e_tail = exp_nonpositive_call(x_tail, pad, row_max).a;
        sum = aie::add(sum, e_tail);
    }

    // Pass 3: normalize, four vectors a step, all loaded before any is stored (the compiler keeps
    // a store before every later load from the row, which serializes the steps' chains).
    // aie::inv: the outputs are within 2.2e-7 of those from three Newton steps in limb products
    // (raw words), which took 1.09-1.14x longer at 256 and 64 wide.
    const limbs inv = split_scalar(aie::inv(aie::reduce_add(sum.template to_vector<float>())));
    const auto normalized = [&](const fvec & e) {
        return mul_limbs(split(to_acc(e)), inv).template to_vector<float>();
    };
    i = 0;
    for (; i + 3 * W < n_whole; i += 4 * W) {
        const fvec e0 = aie::load_v<W>(out + i);
        const fvec e1 = aie::load_v<W>(out + i + W);
        const fvec e2 = aie::load_v<W>(out + i + 2 * W);
        const fvec e3 = aie::load_v<W>(out + i + 3 * W);
        aie::store_v(out + i, normalized(e0));
        aie::store_v(out + i + W, normalized(e1));
        aie::store_v(out + i + 2 * W, normalized(e2));
        aie::store_v(out + i + 3 * W, normalized(e3));
    }
    for (; i < n_whole; i += W) {
        aie::store_v(out + i, normalized(aie::load_v<W>(out + i)));
    }
    if (n_head > 0) {
        store_v_partial<W>(output, normalized(e_head), n_head);
    }
    if (n_tail > 0) {
        store_v_partial<W>(out + n_whole, normalized(e_tail), n_tail);
    }
}

/// Softmax of each of @p n_rows rows of GGML_SOFTMAX_ROW_LEN elements: see ggml_op_soft_max.
inline void soft_max_rows(const float * __restrict input,
                          float * __restrict output,
                          int32_t n_rows,
                          float scale) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    aie::scoped_mode rnd{aie::rounding_mode::conv_even};
    const limbs s = split_scalar(scale);
    const limbs * s_or_unit =
        std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f) ? nullptr : &s;
    for (int32_t r = 0; r < n_rows; ++r) {
        soft_max_row(input + r * N, output + r * N, s_or_unit);
    }
}

#if GGML_SOFTMAX_ROW_LEN <= 32
#define GGML_SOFTMAX_LANE_ROWS 1

// As on aie2 (see there), rows this short run one per lane, here W rows a group.

/// The group transposed: column c of its rows at [c * W, (c + 1) * W), and a column past the row
/// for odd lengths (see soft_max_lane_group's pass 2).
alignas(aie::vector_decl_align) float lane_rows_columns[(GGML_SOFTMAX_ROW_LEN + 1) / 2 * 2 * W];

/// Softmax of each lane's row, the group's columns in @p cols.
__attribute__((always_inline)) inline void soft_max_lane_group(float * __restrict cols,
                                                               const limbs * s) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    if (s != nullptr) {
        for (int32_t c = 0; c < N; ++c) {
            aie::store_v(cols + c * W, scaled(aie::load_v<W>(cols + c * W), *s));
        }
    }

    // Pass 1: each row's (lane's) max.
    ivec kmax = aie::vector_cast<int32_t>(aie::load_v<W>(cols));
    ivec kmin = kmax;
    for (int32_t c = 1; c < N; ++c) {
        const ivec k = aie::vector_cast<int32_t>(aie::load_v<W>(cols + c * W));
        kmax = aie::max(kmax, k);
        kmin = aie::min(kmin, k);
    }
    const acc_t neg_max = to_acc(vec_neg(aie::vector_cast<float>(lane_max(kmax, kmin))));

    // Pass 2: exp(x - max) in place, two columns a step, and each row's sum. A column past the
    // row (odd lengths) is computed and dropped.
    acc_t sum = aie::zeros<accfloat, W>();
    for (int32_t c = 0; c < N; c += 2) {
        const fvec_pair e =
            exp_nonpositive(aie::add(to_acc(aie::load_v<W>(cols + c * W)), neg_max),
                            aie::add(to_acc(aie::load_v<W>(cols + (c + 1) * W)), neg_max));
        aie::store_v(cols + c * W, e.a);
        sum = aie::add(sum, e.a);
        if (c + 1 < N) {
            aie::store_v(cols + (c + 1) * W, e.b);
            sum = aie::add(sum, e.b);
        }
    }

    // Pass 3: normalize, two columns a step, both loaded before either is stored (as in
    // soft_max_row).
    const limbs inv = split(to_acc(aie::inv(sum.template to_vector<float>())));
    const auto normalized = [&](const fvec & e) {
        return mul_limbs(split(to_acc(e)), inv).template to_vector<float>();
    };
    for (int32_t c = 0; c + 1 < N; c += 2) {
        const fvec e0 = aie::load_v<W>(cols + c * W);
        const fvec e1 = aie::load_v<W>(cols + (c + 1) * W);
        aie::store_v(cols + c * W, normalized(e0));
        aie::store_v(cols + (c + 1) * W, normalized(e1));
    }
    if constexpr (N % 2 != 0) {
        aie::store_v(cols + (N - 1) * W, normalized(aie::load_v<W>(cols + (N - 1) * W)));
    }
}

/// Softmax of each of @p n_rows rows of GGML_SOFTMAX_ROW_LEN elements: see ggml_op_soft_max.
inline void soft_max_lane_rows(const float * __restrict input,
                               float * __restrict output,
                               int32_t n_rows,
                               float scale) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    aie::scoped_mode rnd{aie::rounding_mode::conv_even};
    const limbs s = split_scalar(scale);
    const limbs * s_or_unit =
        std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f) ? nullptr : &s;
    float * __restrict cols = lane_rows_columns;
    uint32_t * __restrict cols_bits = reinterpret_cast<uint32_t *>(cols);

    for (int32_t r0 = 0; r0 < n_rows; r0 += W) {
        // In the last group, lanes past its rows hold earlier data: every pass is lane-wise, so
        // they are computed and never stored.
        const int32_t rows = n_rows - r0 < W ? n_rows - r0 : W;

        // Transpose in, a column at a time, as 32-bit words.
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict src =
                reinterpret_cast<const uint32_t *>(input + r0 * N) + c;
            uint32_t * __restrict col = cols_bits + c * W;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *col++ = *src;
                src += N;
            })
        }

        soft_max_lane_group(cols, s_or_unit);

        // Transpose out, a column at a time.
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict col = cols_bits + c * W;
            uint32_t * __restrict dst = reinterpret_cast<uint32_t *>(output + r0 * N) + c;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *dst = *col++;
                dst += N;
            })
        }
    }
}
#endif

#endif // __AIEARCH__ == 20

} // namespace

#endif // GGML_OP_SOFT_MAX

extern "C" {

#ifdef GGML_OP_SOFT_MAX

/**
 * @brief Computes softmax without mask or sink tensors over a tile of rows.
 *
 * Implements the numerically stable softmax: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * for each of the tile's @p n_rows rows of @p N elements.
 *
 * Algorithm, per row:
 * 1. Find global maximum of scaled inputs for numerical stability.
 * 2. Compute exp(scale*x - max) for each element and accumulate sum.
 * 3. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in       Input tile of n_rows * N elements.
 * @param[out] out      Output tile of n_rows * N elements.
 * @param[in]  N        Number of elements in a row.
 * @param[in]  n_rows   Number of rows in the tile.
 * @param[in]  scale    Scale factor applied to input before softmax.
 * @param[in]  max_bias Unused in this variant (kept for API consistency).
 */
void ggml_op_soft_max(const INPUT_DTYPE * __restrict in,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t n_rows,
                      float scale,
                      float max_bias) {
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const float * input = reinterpret_cast<const float *>(in);
    float * output = reinterpret_cast<float *>(out);

#ifdef GGML_SOFTMAX_LANE_ROWS
    soft_max_lane_rows(input, output, n_rows, scale);
#elif __AIEARCH__ == 20
    for (int32_t r = 0; r < n_rows; ++r) {
        soft_max_row(input + r * N, output + r * N, N, scale);
    }
#else
    soft_max_rows(input, output, n_rows, scale);
#endif

    event1();
}

#endif // GGML_OP_SOFT_MAX

#ifdef GGML_OP_SOFT_MAX_WITH_MASK

/**
 * @brief Computes softmax with a mask tensor and ALiBi position biases.
 *
 * Implements: softmax(scale*x + slope*mask) where slope is computed via ALiBi.
 *
 * Algorithm:
 * 1. Compute ALiBi slope for current head based on tile_idx and rows_per_head.
 * 2. Find global maximum of (scale*input + slope*mask) for numerical stability.
 * 3. Compute exp(scale*x + slope*mask - max) and accumulate sum.
 * 4. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements (e.g., causal attention mask).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row.
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Maximum ALiBi bias value.
 * @param[in]  n_head        Total number of attention heads.
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 */
void ggml_op_soft_max_with_mask(const INPUT_DTYPE * __restrict in,
                                const MASK_DTYPE * __restrict mask,
                                OUTPUT_DTYPE * __restrict out,
                                int32_t N,
                                float scale,
                                float max_bias,
                                int32_t n_head,
                                int32_t tile_idx,
                                int32_t rows_per_head) {
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // tile_idx enumerates rows in head-major order (rows_per_head consecutive rows share a head),
    // so this recovers which head the current row belongs to without a separate index tensor.
    const auto head_idx = tile_idx / rows_per_head;
    const auto slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Pass 1: max of the biased row (see ggml_op_soft_max for why this precedes exp()).
    float global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Pass 2: exp(scale*x + slope*mask - max), stored into out for pass 3 to rescale in place.
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Pass 3: normalize.
    float sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; ++i) {
        output[i] *= sum_inv;
    }

    event1();
}

#endif // GGML_OP_SOFT_MAX_WITH_MASK

#ifdef GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

/**
 * @brief Computes softmax with mask and sink (attention sink) tensors.
 *
 * This variant includes a "sink" value per attention head that participates
 * in the softmax normalization but is not stored in the output. Used for
 * streaming attention where early tokens act as attention sinks.
 *
 * Algorithm:
 * 1. Get sink value and ALiBi slope for current head based on tile_idx.
 * 2. Find global maximum including both (scale*input + slope*mask) and sink.
 * 3. Compute exp(scale*x + slope*mask - max) and accumulate sum.
 * 4. Add exp(sink - max) to sum for proper normalization.
 * 5. Normalize output by dividing each exp value by total sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements.
 * @param[in]  sinks         Per-head sink values array (indexed by head_idx).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row.
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Maximum ALiBi bias value.
 * @param[in]  n_head        Total number of attention heads.
 */
void ggml_op_soft_max_with_mask_and_sinks(const INPUT_DTYPE * __restrict in,
                                          const MASK_DTYPE * __restrict mask,
                                          const SINK_DTYPE * __restrict sinks,
                                          OUTPUT_DTYPE * __restrict out,
                                          int32_t N,
                                          int32_t tile_idx,
                                          int32_t rows_per_head,
                                          float scale,
                                          float max_bias,
                                          int32_t n_head) {

    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<SINK_DTYPE, float>::value, "SINK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // Get sink value for this head
    const auto head_idx = tile_idx / rows_per_head;
    const auto sink_val = sinks[head_idx];

    // Compute ALiBi slope
    const auto slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Step 1: Find max for numerical stability (including sink)
    float global_max = sink_val;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Add sink contribution to sum
    const auto sink_exp = scalar_exp(sink_val - global_max);
    sum_total += sink_exp;

    // Step 3: Normalize
    const auto sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}

#endif // GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

} // extern "C"
