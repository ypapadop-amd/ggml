// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

// Pooling op selector (matches enum ggml_op_pool in include/ggml.h).
constexpr int32_t GGML_OP_POOL_MAX = 0;

namespace {

// The plane's extents and op_params, from pool_2d.py (also passed at run time, unused).
constexpr int32_t iw = GGML_POOL_IW, ih = GGML_POOL_IH, ow = GGML_POOL_OW, oh = GGML_POOL_OH;
constexpr int32_t k0 = GGML_POOL_K0, k1 = GGML_POOL_K1, s0 = GGML_POOL_S0, s1 = GGML_POOL_S1,
                  p0 = GGML_POOL_P0, p1 = GGML_POOL_P1;
constexpr bool is_max = GGML_POOL_OP == GGML_OP_POOL_MAX;
constexpr int32_t plane = iw * ih;
constexpr int32_t out_plane = ow * oh;

// Max pooling runs on integer keys: the aie2 scalar unit has no float compare (`v > res` is a
// runtime call, __gtsf2), and neither arch has a native f32 vector max. float_order_key
// (ggml-aie.hpp) orders non-NaN floats exactly as `<` does, so the max key is the key of the max.
// A NaN gets the lowest key and never wins, as `(v > res) ? v : res` never takes one. -0 and +0
// share key 0, so a window whose maximum is a zero returns +0.
constexpr int32_t kLowestKey = -0x7f7fffff; // key(-FLT_MAX), the reference's starting value

// Output rows run in chunks of V columns. Tap group g of a chunk is one load of L = s0 * V
// input columns starting at its window column g * s0: with s0 = 1 the load is the taps kx = g of
// all V outputs, with s0 = 2 its even lanes are kx = 2g and its odd lanes kx = 2g + 1.
constexpr int32_t V = 16; // f32 lanes per 512-bit vector, on aie2 and aie2p
constexpr bool vector_path = s0 == 1 || s0 == 2;
constexpr int32_t L = vector_path ? s0 * V : V; // (any width when unused)
constexpr int32_t groups = (k0 + s0 - 1) / s0;
constexpr int32_t chunks = (ow + V - 1) / V;
// Whether some output's window crosses the row's left or right end: those lanes are masked.
constexpr bool col_mask = p0 > 0 || (ow - 1) * s0 + k0 - p0 > iw;

struct row_range {
    int32_t lo, hi;
};

/// Output rows whose window rows all lie in the plane, whose loads stay at or after the plane's
/// start and end at most @p limit elements past it, and, in the tile's last plane (@p last),
/// whose stores stay inside the plane: [lo, hi).
constexpr row_range interior_rows(int32_t limit, bool last) {
    int32_t lo = 0;
    // A row's first load starts at column -p0: before the plane's start in its first row.
    while (lo < oh && lo * s1 - p1 < (p0 > 0 ? 1 : 0)) {
        ++lo;
    }
    int32_t hi = oh;
    const int32_t load_end = (chunks - 1) * V * s0 - p0 + (groups - 1) * s0 + L;
    while (hi > lo &&
           ((hi - 1) * s1 - p1 + k1 > ih || ((hi - 1) * s1 - p1 + k1 - 1) * iw + load_end > limit ||
            (last && hi == oh && chunks * V > ow))) {
        --hi;
    }
    return {lo, hi};
}

template <typename T>
inline aie::vector<float, L> to_float(const aie::vector<T, L> & v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else {
        aie::accum<accfloat, L> acc;
        acc.from_vector(v);
        return acc.template to_vector<float>();
    }
}

/// L elements at tile[at] through a bounce buffer: those inside [0, len) of the tile, the rest
/// zero.
template <typename T>
inline aie::vector<T, L> load_bounded(const T * tile, int32_t at, int32_t len) {
    alignas(aie::vector_decl_align) T buf[L] = {};
    for (int32_t j = 0; j < L; ++j) {
        if (at + j >= 0 && at + j < len) {
            buf[j] = tile[at + j];
        }
    }
    return aie::load_v<L>(buf);
}

/// L elements at tile[at], as float. Checked loads read only elements inside [0, len) of the
/// tile (the rest are zero); unchecked ones must lie inside it.
template <bool Checked, typename T>
inline aie::vector<float, L> load(const T * __restrict tile, int32_t at, int32_t len) {
    if (Checked && (at < 0 || at + L > len)) {
        return to_float<T>(load_bounded(tile, at, len));
    }
    return to_float<T>(aie::load_unaligned_v<L>(tile + at));
}

/// Lanes of a group's load whose column lies in the row: x is the load's first column.
inline aie::mask<L> in_row(int32_t x) {
    alignas(aie::vector_decl_align) static constexpr int32_t lane_index[32] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const auto col = aie::add(aie::load_v<L>(lane_index), x);
    return aie::ge(col, int32_t{0}) & aie::lt(col, iw);
}

/// The max over a group's taps, from its keys: with s0 = 1 the load is one tap; with s0 = 2 its
/// even lanes are one and, if @p odd, its odd lanes the next.
inline aie::vector<int32_t, V> taps_max(const aie::vector<int32_t, V> & k, bool) { return k; }
inline aie::vector<int32_t, V> taps_max(const aie::vector<int32_t, 2 * V> & k, bool odd) {
    const auto even = aie::filter_even(k);
    return odd ? aie::max(even, aie::filter_odd(k)) : even;
}

/// @p sum plus a group's taps, in column order (see @ref taps_max).
inline aie::vector<float, V>
add_taps(const aie::vector<float, V> & sum, const aie::vector<float, V> & v, bool) {
    return aie::add(sum, v);
}
inline aie::vector<float, V>
add_taps(const aie::vector<float, V> & sum, const aie::vector<float, 2 * V> & v, bool odd) {
    const auto s = aie::add(sum, aie::filter_even(v));
    return odd ? aie::add(s, aie::filter_odd(v)) : s;
}

/// Output row oy of the plane at tile[base], chunk c, as float lanes. Checked rows skip window
/// rows outside the plane and load through @ref load's bounds check.
template <bool Checked, typename T>
inline aie::vector<float, V>
pool_chunk(const T * __restrict tile, int32_t len, int32_t base, int32_t oy, int32_t c) {
    const int32_t x0 = c * V * s0 - p0;
    if constexpr (is_max) {
        // Max is order-free: each group's keys take their max over the window rows first, and
        // its phases are paired up once.
        aie::vector<int32_t, L> acc[groups];
        for (int32_t g = 0; g < groups; ++g) {
            acc[g] = aie::broadcast<int32_t, L>(kLowestKey);
        }
        for (int32_t ky = 0; ky < k1; ++ky) {
            const int32_t y = oy * s1 - p1 + ky;
            if (Checked && (y < 0 || y >= ih)) {
                continue;
            }
            for (int32_t g = 0; g < groups; ++g) {
                const int32_t x = x0 + g * s0;
                const auto v = load<Checked>(tile, base + y * iw + x, len);
                auto skip = floats_are_nan<L>(v);
                if constexpr (col_mask) {
                    skip = skip | ~in_row(x);
                }
                acc[g] = aie::max(acc[g], aie::select(float_order_keys<L>(v),
                                                      aie::broadcast<int32_t, L>(INT32_MIN), skip));
            }
        }
        aie::vector<int32_t, V> best = taps_max(acc[0], k0 > 1);
        for (int32_t g = 1; g < groups; ++g) {
            best = aie::max(best, taps_max(acc[g], g * s0 + 1 < k0));
        }
        return floats_from_order_keys<V>(best);
    } else {
        // Average: the taps are summed in the reference's order (window row, then column), from
        // 0, and an out-of-bounds tap adds 0.
        aie::vector<float, V> sum = aie::zeros<float, V>();
        for (int32_t ky = 0; ky < k1; ++ky) {
            const int32_t y = oy * s1 - p1 + ky;
            if (Checked && (y < 0 || y >= ih)) {
                continue;
            }
            for (int32_t g = 0; g < groups; ++g) {
                const int32_t x = x0 + g * s0;
                auto v = load<Checked>(tile, base + y * iw + x, len);
                if constexpr (col_mask) {
                    v = aie::select(aie::zeros<float, L>(), v, in_row(x));
                }
                sum = add_taps(sum, v, g * s0 + 1 < k0);
            }
        }
        if constexpr (k0 * k1 == 1) {
            return sum;
        } else {
            return aie::mul(sum, 1.0f / (k0 * k1)).template to_vector<float>();
        }
    }
}

/// Output rows [lo, hi) of the plane at tile[base] (out[out_base]). Chunks may write up to V - 1
/// lanes past a row: the next row (or chunk) overwrites them, and the tile's last chunk, which
/// has no next, stores only its lanes.
template <bool Checked, typename T>
inline void pool_rows(const T * __restrict tile,
                      int32_t len,
                      float * __restrict out,
                      int32_t out_len,
                      int32_t base,
                      int32_t out_base,
                      int32_t lo,
                      int32_t hi) {
    for (int32_t oy = lo; oy < hi; ++oy) {
        for (int32_t c = 0; c < chunks; ++c) {
            const auto r = pool_chunk<Checked>(tile, len, base, oy, c);
            const int32_t at = out_base + oy * ow + c * V;
            if (Checked && at + V > out_len) {
                store_v_partial<V>(out + at, r, out_len - at);
            } else {
                store_v_any(out + at, r);
            }
        }
    }
}

/// Plane @p p of the tile: rows [lo, hi) unchecked (compile-time bounds), the rest checked.
template <int32_t lo, int32_t hi, typename T>
inline void pool_plane(
    const T * __restrict tile, int32_t len, float * __restrict out, int32_t out_len, int32_t p) {
    const int32_t base = p * plane;
    const int32_t out_base = p * out_plane;
    pool_rows<true>(tile, len, out, out_len, base, out_base, 0, lo);
    pool_rows<false>(tile, len, out, out_len, base, out_base, lo, hi);
    pool_rows<true>(tile, len, out, out_len, base, out_base, hi, oh);
}

/// The scalar fallback for strides above 2.
template <typename T>
void pool_plane_scalar(const T * __restrict in, float * __restrict out) {
    for (int32_t oy = 0; oy < oh; ++oy) {
        for (int32_t ox = 0; ox < ow; ++ox) {
            int32_t best = kLowestKey;
            float sum = 0.0f;
            for (int32_t ky = 0; ky < k1; ++ky) {
                const int32_t y = oy * s1 - p1 + ky;
                if (y < 0 || y >= ih) {
                    continue;
                }
                for (int32_t kx = 0; kx < k0; ++kx) {
                    const int32_t x = ox * s0 - p0 + kx;
                    if (x < 0 || x >= iw) {
                        continue;
                    }
                    const float v = static_cast<float>(in[y * iw + x]);
                    if constexpr (is_max) {
                        const int32_t k = float_is_nan(v) ? INT32_MIN : float_order_key(v);
                        best = k > best ? k : best;
                    } else {
                        sum += v;
                    }
                }
            }
            if constexpr (is_max) {
                out[oy * ow + ox] = float_from_order_key(best);
            } else {
                out[oy * ow + ox] = sum * (1.0f / (k0 * k1));
            }
        }
    }
}

} // namespace

extern "C" {

/**
 * @brief Reduces each k1 x k0 window of @p num_planes input channel-planes to an output element.
 *
 * Mirrors ggml_compute_forward_pool_2d. Padding is handled by skipping out-of-bounds taps rather
 * than gathering a padded buffer: for MAX this is equivalent to -inf padding, and for AVG the
 * divisor is still the full k0*k1 window area (not the count of in-bounds taps), as in the GGML
 * CPU reference. AVG multiplies by 1 / (k0*k1) where the reference divides, which can differ in
 * the last bit when k0*k1 is not a power of two.
 *
 * The extents and op_params are compiled in (-DGGML_POOL_*, pool_2d.py); the run-time copies
 * are kept for the call's signature.
 *
 * @param[in]  in         num_planes input channel-planes of iw * ih elements each (row-major,
 *                        width fastest), back to back.
 * @param[out] out        num_planes output channel-planes of ow * oh elements each.
 * @param[in]  num_planes Number of planes in the tile.
 */
void ggml_op_pool_2d(const INPUT_DTYPE * __restrict in,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t /*iw*/,
                     int32_t /*ih*/,
                     int32_t /*ow*/,
                     int32_t /*oh*/,
                     int32_t /*k0*/,
                     int32_t /*k1*/,
                     int32_t /*s0*/,
                     int32_t /*s1*/,
                     int32_t /*p0*/,
                     int32_t /*p1*/,
                     int32_t /*op*/,
                     int32_t num_planes) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    if constexpr (vector_path) {
        // Rows whose loads and stores stay inside the tile take the unchecked path: those of every
        // plane but the last may read and write into the next plane, the last plane's must stay
        // inside it.
        constexpr row_range inner = interior_rows(2 * plane, false);
        constexpr row_range last = interior_rows(plane, true);
        const int32_t len = num_planes * plane;
        const int32_t out_len = num_planes * out_plane;
        for (int32_t p = 0; p + 1 < num_planes; ++p) {
            pool_plane<inner.lo, inner.hi>(in, len, out, out_len, p);
        }
        pool_plane<last.lo, last.hi>(in, len, out, out_len, num_planes - 1);
    } else {
        for (int32_t p = 0; p < num_planes; ++p) {
            pool_plane_scalar(in + p * plane, out + p * out_plane);
        }
    }

    event1();
}

} // extern "C"
