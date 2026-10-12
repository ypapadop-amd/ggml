// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

// im2col.py passes the image and window extents, strides, padding and dilation, and the output
// rows per call, as -DGGML_IM2COL_*, so every loop below has compile-time bounds.
namespace {

constexpr int32_t c_iw = GGML_IM2COL_IW, c_ih = GGML_IM2COL_IH, c_ic = GGML_IM2COL_IC;
constexpr int32_t c_kw = GGML_IM2COL_KW, c_kh = GGML_IM2COL_KH, c_ow = GGML_IM2COL_OW;
constexpr int32_t c_s0 = GGML_IM2COL_S0, c_s1 = GGML_IM2COL_S1, c_p0 = GGML_IM2COL_P0,
                  c_p1 = GGML_IM2COL_P1, c_d0 = GGML_IM2COL_D0, c_d1 = GGML_IM2COL_D1;
constexpr int32_t c_rows = GGML_IM2COL_ROWS;
// Output column slots per output pixel, one per (channel, window row, window column).
constexpr int32_t cs = c_ic * c_kh * c_kw;

#if defined(GGML_IM2COL_PACKED)
// Slot-major layout: the call writes its rows as [ROWS][CS][OW], each slot's OW values one
// contiguous run (a shifted window of one input row, zeros where it falls in the padding), and
// the output fifo's DMA walks the object as [ROWS][OW][CS], the im2col order. The runs are
// packed back to back into whole vectors, so every store is one aligned vector.
constexpr int32_t V = 16;
constexpr int32_t ow_chunks = (c_ow + V - 1) / V;
constexpr int32_t group_elems = c_kw * c_ow; // the KW runs of one (row, channel, window row)
constexpr int32_t groups = c_rows * c_ic * c_kh;
// Groups after which a run again starts on a vector boundary: the kernel unrolls one such
// period, so where each chunk lands in a vector, and every shift and mask, is a constant.
constexpr int32_t period = V / std::gcd(group_elems, V);
static_assert(c_s0 == 1, "the packed path copies contiguous input runs");
static_assert(groups % period == 0, "the object must be a whole number of vectors");

constexpr int32_t clamp_lane(int32_t x) { return x < 0 ? 0 : (x > V ? V : x); }

/// Bits of lanes [b, e) of a V-lane mask, for 0 <= b, e <= V.
constexpr uint32_t lane_bits(int32_t b, int32_t e) { return e > b ? (1u << e) - (1u << b) : 0u; }

/// Per group g (row r, channel iic, window row ikh, in that nesting): its input row less
/// oh * s1, and its channel plane's offset.
struct group_table {
    int32_t row[groups];
    int32_t plane[groups];
};
constexpr group_table make_group_table() {
    group_table t{};
    for (int32_t g = 0; g < groups; ++g) {
        const int32_t r = g / (c_ic * c_kh), iic = g / c_kh % c_ic, ikh = g % c_kh;
        t.row[g] = r * c_s1 + ikh * c_d1 - c_p1;
        t.plane[g] = iic * c_ih * c_iw;
    }
    return t;
}
alignas(32) constexpr group_table c_groups = make_group_table();

inline void
im2col_packed(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t oh) {
    using vec = aie::vector<OUTPUT_DTYPE, V>;
    const vec zeros = aie::zeros<OUTPUT_DTYPE, V>();
    vec acc = zeros; // the lanes of the next vector to store gathered so far
    OUTPUT_DTYPE * __restrict o = out;
    const int32_t row0 = oh * c_s1;

    // Chunk j of window column ikw's run in group q of a period.
    const auto chunk = [&]<int q, int ikw, int j>(bool y_in, const INPUT_DTYPE * __restrict srow) {
        constexpr int32_t off = ikw * c_d0 - c_p0;
        // Output columns [lo, hi) read input columns ox + off inside [0, iw).
        constexpr int32_t lo = off < 0 ? -off : 0;
        constexpr int32_t hi = c_iw - off < c_ow ? c_iw - off : c_ow;
        constexpr uint32_t bits = lane_bits(clamp_lane(lo - j * V), clamp_lane(hi - j * V));
        constexpr int32_t pos = q * group_elems + ikw * c_ow + j * V; // offset in the period
        constexpr int32_t f = pos % V; // lane of the vector the chunk starts at
        constexpr int32_t n = c_ow - j * V < V ? c_ow - j * V : V;
        // Input columns j * V + off + [0, V); nothing left of the row is read: a window
        // starting there loads from the row's start and shifts up.
        constexpr int32_t a = j * V + off;
        vec v = zeros;
        if constexpr (bits != 0) {
            v = aie::load_unaligned_v<V>(srow + (a < 0 ? 0 : a));
            if constexpr (a < 0) {
                v = aie::shuffle_up(v, -a);
            }
            v = aie::select(zeros, v, aie::mask<V>::from_uint32(y_in ? bits : 0u));
        }
        if constexpr (f == 0) {
            acc = v;
        } else {
            acc =
                aie::select(acc, aie::shuffle_up(v, f), aie::mask<V>::from_uint32(lane_bits(f, V)));
        }
        if constexpr (f + n >= V) {
            aie::store_v(o, acc);
            o += V;
            if constexpr (f + n > V) {
                acc = aie::shuffle_down(v, V - f);
            }
        }
    };

    // The chunks of window column ikw's run.
    const auto window = [&]<int q, int ikw, int... j>(bool y_in,
                                                      const INPUT_DTYPE * __restrict srow,
                                                      std::integer_sequence<int, j...>) {
        (chunk.template operator()<q, ikw, j>(y_in, srow), ...);
    };

    // Group g, the q-th of its period.
    const auto group = [&]<int q>(int32_t g, std::integral_constant<int, q>) {
        const int32_t iih = row0 + c_groups.row[g];
        const bool y_in = iih >= 0 && iih < c_ih;
        const INPUT_DTYPE * __restrict srow = in + c_groups.plane[g] + (y_in ? iih * c_iw : 0);
        [&]<int... ikw>(std::integer_sequence<int, ikw...>) {
            (window.template operator()<q, ikw>(y_in, srow,
                                                std::make_integer_sequence<int, ow_chunks>{}),
             ...);
        }(std::make_integer_sequence<int, c_kw>{});
    };

    // Not unrolled further: whole unrolled objects keep too many loads in flight and spill.
    AIE_LOOP_NO_UNROLL
    for (int32_t g = 0; g < groups; g += period) {
        [&]<int... q>(std::integer_sequence<int, q...>) {
            (group(g + q, std::integral_constant<int, q>{}), ...);
        }(std::make_integer_sequence<int, period>{});
    }
}
#endif

} // namespace

extern "C" {

/**
 * @brief Gathers GGML_IM2COL_ROWS consecutive output rows of a 2D im2col transform for a
 * single image.
 *
 * Mirrors ggml_compute_forward_im2col_f32 for one batch element and output rows
 * oh .. oh + GGML_IM2COL_ROWS - 1. Output columns pack IC*KH*KW taps channel-major; taps outside
 * the padded input are written as zero rather than skipped, so every column
 * is fully populated. INPUT_DTYPE/OUTPUT_DTYPE may differ: each element is
 * cast on the way out. With GGML_IM2COL_PACKED the rows are written slot-major (see
 * im2col_packed) and the DMA restores the order.
 *
 * The extents are compile-time (-DGGML_IM2COL_*); the run-time ones are kept for the
 * kernel's signature only.
 *
 * @param[in]  in   Input image: IC planes of IH * IW elements (row-major).
 * @param[out] out  Output rows: GGML_IM2COL_ROWS * OW * (IC * KH * KW) elements.
 * @param[in]  oh   First output row index along height.
 */
void ggml_op_im2col(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t oh,
                    int32_t /*iw*/,
                    int32_t /*ih*/,
                    int32_t /*ic*/,
                    int32_t /*kw*/,
                    int32_t /*kh*/,
                    int32_t /*ow*/,
                    int32_t /*s0*/,
                    int32_t /*s1*/,
                    int32_t /*p0*/,
                    int32_t /*p1*/,
                    int32_t /*d0*/,
                    int32_t /*d1*/) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "OUTPUT_DTYPE must be a floating-point type");

    event0();

#if defined(GGML_IM2COL_PACKED)
    static_assert(std::is_same_v<INPUT_DTYPE, OUTPUT_DTYPE>, "the packed path copies, not casts");
    im2col_packed(in, out, oh);
#else
    // Each (channel, window row, window column) is one output column slot, filled along ox
    // with the padding columns split off at compile time: zeros, then a pipelined copy, then
    // zeros. A window row in the padding is all zeros.
    const auto zero = static_cast<OUTPUT_DTYPE>(0.0f);

    // First and one-past-last ox whose input column ox * s0 + off lies in [0, iw).
    constexpr auto ox_lo = [](int32_t off) {
        const int32_t lo = off >= 0 ? 0 : (-off + c_s0 - 1) / c_s0;
        return lo < c_ow ? lo : c_ow;
    };
    constexpr auto ox_hi = [](int32_t off, int32_t lo) {
        const int32_t hi = (c_iw - 1 - off < 0) ? 0 : (c_iw - 1 - off) / c_s0 + 1;
        return hi < lo ? lo : (hi > c_ow ? c_ow : hi);
    };

    for (int32_t r = 0; r < c_rows; ++r) {
        OUTPUT_DTYPE * __restrict const row_out = out + r * (c_ow * cs);
        for (int32_t iic = 0; iic < c_ic; ++iic) {
            const INPUT_DTYPE * __restrict src_plane = in + iic * c_ih * c_iw;
            for (int32_t ikh = 0; ikh < c_kh; ++ikh) {
                const int32_t iih = (oh + r) * c_s1 + ikh * c_d1 - c_p1;
                const bool y_in = iih >= 0 && iih < c_ih;
                const INPUT_DTYPE * __restrict srow = src_plane + (y_in ? iih * c_iw : 0);
                for (int32_t ikw = 0; ikw < c_kw; ++ikw) {
                    const int32_t off = ikw * c_d0 - c_p0;
                    // Bounds selected on y_in, not a separate zero-only branch: with that
                    // branch Peano no longer unrolled the copy loops, 6 cycles per element
                    // against ~3 (MNIST conv1 im2col 2.8 -> 3.9 ms).
                    const int32_t lo = y_in ? ox_lo(off) : c_ow;
                    const int32_t hi = y_in ? ox_hi(off, lo) : c_ow;
                    // Walking cursors, not dst[ox * cs] / srow[ox * s0 + off]: indexed, Peano
                    // recomputed each address (aie2 copy 3 cycles per element, aie2p a 64-bit
                    // multiply in the loop setup).
                    OUTPUT_DTYPE * __restrict d = row_out + (iic * c_kh + ikh) * c_kw + ikw;
                    for (int32_t ox = 0; ox < lo; ++ox, d += cs) {
                        *d = zero;
                    }
                    const INPUT_DTYPE * __restrict sp = srow + (lo * c_s0 + off);
                    for (int32_t ox = lo; ox < hi; ++ox, d += cs, sp += c_s0) {
                        *d = static_cast<OUTPUT_DTYPE>(*sp);
                    }
                    for (int32_t ox = hi; ox < c_ow; ++ox, d += cs) {
                        *d = zero;
                    }
                }
            }
        }
    }
#endif

    event1();
}

} // extern "C"
