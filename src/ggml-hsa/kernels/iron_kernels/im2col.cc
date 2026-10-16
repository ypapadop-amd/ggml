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

#if defined(GGML_IM2COL_PAIRS)
// Pixel-pair word layout, for a 16-bit output (the DMA moves 32-bit words, so it cannot transpose
// 16-bit elements): the output of pixels 2q and 2q + 1 is 2 * CS values, CS words; word j holds
// values 2j and 2j + 1, from slots (2j) % CS and (2j + 1) % CS of pixel 2q + (2j) / CS and
// 2q + (2j + 1) / CS. The call writes its rows as [ROWS][CS][OW / 2] words, each (row, j)'s OW / 2
// words one run, packed back to back into aligned W-word vectors, and the output fifo's DMA walks
// the object as [ROWS][OW / 2][CS], the im2col order. A word's low half is input column 2q + c of
// one row and its high half column 2q + c' of another (or the same), so W words are two V-lane
// windows, the odd lanes of the second taken over the first, converted together.
constexpr int32_t V = 16;           // f32 lanes per chunk: W words of two values
constexpr int32_t W = V / 2;        // words per chunk and per stored vector
constexpr int32_t q_len = c_ow / 2; // words per run
constexpr int32_t q_chunks = (q_len + W - 1) / W;
// Runs after which a run again starts on a vector boundary: the kernel unrolls one such period,
// so where each chunk lands in a vector, and every shuffle and store point, is a constant. Which
// slots a run reads is looked up per run (an unrolled whole row of runs outgrew program memory).
constexpr int32_t period = W / std::gcd(q_len, W);
constexpr int32_t runs = c_rows * cs;
static_assert(c_s0 == 1, "the pairs path reads input runs at a column stride of 2 pixels");
static_assert(c_ow % 2 == 0, "the pairs path pairs pixels");
static_assert(runs % period == 0, "the object must be a whole number of vectors");

/// One half (low: value 2j, high: 2j + 1) of word index j: its slot's row less the output row's
/// first input row, and that row's offset from the latter's start; per chunk, where its window
/// loads, from there in an interior call (pairs_interior) and else from its own row's start (never
/// left of it), how far it then shifts up, and with the other half, the lanes whose input column
/// lies in the row (lanes past the run need not be cleared: the next run's words go over them).
struct pair_half {
    int32_t offset;
    int32_t start[q_chunks];
    int16_t row;
    int16_t load[q_chunks], shift[q_chunks];
};
struct pair_run {
    pair_half lo, hi;
    uint16_t bits[q_chunks];
};
struct pair_table {
    pair_run j[cs];
};
constexpr pair_table make_pair_table() {
    pair_table t{};
    for (int32_t j = 0; j < cs; ++j) {
        for (int32_t h = 0; h < 2; ++h) {
            const int32_t e = 2 * j + h, slot = e % cs, px = e / cs;
            const int32_t iic = slot / (c_kh * c_kw), ikh = slot / c_kw % c_kh, ikw = slot % c_kw;
            const int32_t col = px + ikw * c_d0 - c_p0; // input column of word 0
            pair_half & half = h == 0 ? t.j[j].lo : t.j[j].hi;
            half.row = static_cast<int16_t>(ikh * c_d1 - c_p1);
            half.offset = iic * c_ih * c_iw + half.row * c_iw;
            for (int32_t k = 0; k < q_chunks; ++k) {
                // The high half loads one column early, so column col + 2i lands at lane 2i + 1.
                const int32_t start = col + V * k - h;
                half.start[k] = half.offset + start;
                half.load[k] = static_cast<int16_t>(start < 0 ? 0 : start);
                half.shift[k] = static_cast<int16_t>(start < 0 ? (-start < V ? -start : V) : 0);
                for (int32_t i = 0; i < W; ++i) {
                    const int32_t c = col + 2 * (W * k + i);
                    if (c >= 0 && c < c_iw) {
                        t.j[j].bits[k] |= static_cast<uint16_t>(1u << (2 * i + h));
                    }
                }
            }
        }
    }
    return t;
}
alignas(32) constexpr pair_table c_pairs = make_pair_table();

/// Whether every window row of the call's rows lies in the image, below its first row: then every
/// load (starting at most P0 + 1 columns left of its row) stays in the image unshifted.
constexpr bool pairs_interior(int32_t oh) {
    return c_iw > c_p0 + 1 && oh * c_s1 - c_p1 >= 1 &&
           (oh + c_rows - 1) * c_s1 + (c_kh - 1) * c_d1 - c_p1 < c_ih;
}

template <bool interior>
inline void
im2col_pairs(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t oh) {
    using fvec = aie::vector<INPUT_DTYPE, V>;
    using wvec = aie::vector<uint32_t, W>;
    const fvec zeros = aie::zeros<INPUT_DTYPE, V>();
    wvec acc = aie::zeros<uint32_t, W>(); // the words of the next vector to store gathered so far
    uint32_t * __restrict o = reinterpret_cast<uint32_t *>(out);

    // A half's row start (the image's, when the row lies in the padding: its lanes are cleared,
    // and the loads stay in the image) and, when it does not, its lanes to keep.
    const auto source = [&](int32_t row, int32_t row_start, const pair_half & half, uint32_t lanes,
                            uint32_t & keep) {
        const bool y_in = static_cast<uint32_t>(row + half.row) < static_cast<uint32_t>(c_ih);
        keep |= y_in ? lanes : 0u;
        return in + (y_in ? row_start + half.offset : 0);
    };

    // Chunk k of the u-th run of a period, from its two halves' windows and the lanes to keep.
    const auto chunk = [&]<int u, int k>(const fvec & lo, const fvec & hi, uint32_t keep) {
        constexpr int32_t pos = u * q_len + k * W; // word offset in the period
        constexpr int32_t f = pos % W;             // lane of the vector the chunk starts at
        constexpr int32_t n = q_len - k * W < W ? q_len - k * W : W;
        fvec v = aie::select(lo, hi, aie::mask<V>::from_uint32(0xaaaau));
        v = aie::select(zeros, v, aie::mask<V>::from_uint32(keep));
        // vconv, as the in-order path's static_cast
        const wvec words = aie::vector_cast<uint32_t>(
            aie::accum<accfloat, V>(v).template to_vector<OUTPUT_DTYPE>());
        if constexpr (f == 0) {
            acc = words;
        } else {
            acc = aie::select(acc, aie::shuffle_up(words, f),
                              aie::mask<W>::from_uint32(((1u << W) - 1) & ~((1u << f) - 1)));
        }
        if constexpr (f + n >= W) {
            aie::store_v(o, acc);
            o += W;
            if constexpr (f + n > W) {
                acc = aie::shuffle_down(words, W - f);
            }
        }
    };

    // The u-th run of a period: word index j of the output row whose first input row (less
    // padding) is `row`, at `row_start` in the image.
    const auto run = [&]<int u, int... k>(int32_t row, int32_t row_start, int32_t j,
                                          std::integer_sequence<int, k...>) {
        const pair_run & t = c_pairs.j[j];
        if constexpr (interior) {
            const INPUT_DTYPE * __restrict r = in + row_start;
            (chunk.template operator()<u, k>(aie::load_unaligned_v<V>(r + t.lo.start[k]),
                                             aie::load_unaligned_v<V>(r + t.hi.start[k]),
                                             t.bits[k]),
             ...);
        } else {
            uint32_t keep = 0;
            const INPUT_DTYPE * __restrict s_lo = source(row, row_start, t.lo, 0x5555u, keep);
            const INPUT_DTYPE * __restrict s_hi = source(row, row_start, t.hi, 0xaaaau, keep);
            (chunk.template operator()<u, k>(
                 aie::shuffle_up(aie::load_unaligned_v<V>(s_lo + t.lo.load[k]),
                                 static_cast<unsigned>(t.lo.shift[k])),
                 aie::shuffle_up(aie::load_unaligned_v<V>(s_hi + t.hi.load[k]),
                                 static_cast<unsigned>(t.hi.shift[k])),
                 t.bits[k] & keep),
             ...);
        }
    };

    int32_t row = oh * c_s1;        // the output row's first input row (less padding)
    int32_t row_start = row * c_iw; // its offset in the image
    int32_t j = 0;
    AIE_LOOP_NO_UNROLL
    for (int32_t w = 0; w < runs; w += period) {
        [&]<int... u>(std::integer_sequence<int, u...>) {
            (
                [&] {
                    run.template operator()<u>(row, row_start, j,
                                               std::make_integer_sequence<int, q_chunks>{});
                    const bool wrap = j == cs - 1;
                    j = wrap ? 0 : j + 1;
                    row += wrap ? c_s1 : 0;
                    row_start += wrap ? c_s1 * c_iw : 0;
                }(),
                ...);
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
 * im2col_packed), with GGML_IM2COL_PAIRS as runs of pixel-pair words (see im2col_pairs), and the
 * DMA restores the order.
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
#elif defined(GGML_IM2COL_PAIRS)
    static_assert(sizeof(OUTPUT_DTYPE) == 2, "the pairs path packs two 16-bit values a word");
    if (pairs_interior(oh)) {
        im2col_pairs<true>(in, out, oh);
    } else {
        im2col_pairs<false>(in, out, oh);
    }
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
