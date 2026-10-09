// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

// Pooling op selector (matches enum ggml_op_pool in include/ggml.h).
constexpr int32_t GGML_OP_POOL_MAX = 0;

#if __AIEARCH__ == 20 && defined(GGML_POOL_OP) // aie2 only so far; pool_2d.py passes the shape.
namespace {

// Max pooling on integer keys: the aie2 scalar unit has no float compare, so the float loop
// below paid a runtime call (__gtsf2) for every tap. key() is float_order_key (ggml-aie.hpp),
// which orders non-NaN floats exactly as `<` does, so the max key is the key of the max. A NaN gets
// the lowest key and never wins, as `(v > res) ? v : res` never takes one. -0 and +0 share key 0,
// so a window whose maximum is a zero returns +0.
constexpr int32_t kLowestKey = -0x7f7fffff; // key(-FLT_MAX), the float loop's starting value

inline int32_t key(float f) { return float_is_nan(f) ? INT32_MIN : float_order_key(f); }

template <int32_t V>
inline aie::vector<int32_t, V> keys(aie::vector<float, V> v) {
    return aie::select(float_order_keys<V>(v), aie::broadcast<int32_t, V>(INT32_MIN),
                       floats_are_nan<V>(v));
}

template <typename T_in>
void pool_2d_max(const T_in * __restrict in, float * __restrict out) {
    constexpr int32_t iw = GGML_POOL_IW, ih = GGML_POOL_IH, ow = GGML_POOL_OW, oh = GGML_POOL_OH;
    constexpr int32_t k0 = GGML_POOL_K0, k1 = GGML_POOL_K1, s0 = GGML_POOL_S0, s1 = GGML_POOL_S1,
                      p0 = GGML_POOL_P0, p1 = GGML_POOL_P1;

    if constexpr (std::is_same_v<T_in, float> && k0 == 2 && s0 == 2 && p0 == 0) {
        // Output chunks of V columns read 2V input columns: the row's vertical max is taken
        // in keys, two V-lane vectors at a time, and filter_even/filter_odd pair up adjacent
        // columns. Lanes past the row (into the next row) are computed and never stored; a
        // load that would cross the end of the plane goes through a bounce buffer instead.
        constexpr int32_t V = 16;
        constexpr int32_t n_chunks = (ow + V - 1) / V;
        constexpr int32_t plane = iw * ih;
        const auto load = [&](int32_t at) {
            if (at + V <= plane) {
                return aie::load_unaligned_v<V>(in + at);
            }
            alignas(aie::vector_decl_align) float buf[V] = {};
            for (int32_t j = 0; j < plane - at; ++j) {
                buf[j] = in[at + j];
            }
            return aie::load_v<V>(buf);
        };
        for (int32_t oy = 0; oy < oh; ++oy) {
            for (int32_t c = 0; c < n_chunks; ++c) {
                aie::vector<int32_t, V> lo = aie::broadcast<int32_t, V>(kLowestKey);
                aie::vector<int32_t, V> hi = lo;
                for (int32_t ky = 0; ky < k1; ++ky) {
                    const int32_t y = oy * s1 - p1 + ky;
                    if (y < 0 || y >= ih) {
                        continue;
                    }
                    const int32_t at = y * iw + 2 * V * c;
                    lo = aie::max(lo, keys<V>(load(at)));
                    hi = aie::max(hi, keys<V>(load(at + V)));
                }
                const auto pairs = aie::concat(lo, hi);
                const auto best = aie::max(aie::filter_even(pairs), aie::filter_odd(pairs));
                alignas(aie::vector_decl_align) float chunk[V];
                aie::store_v(chunk, floats_from_order_keys<V>(best));
                const int32_t ox = c * V;
                const int32_t lanes = (ow - ox < V) ? ow - ox : V;
                for (int32_t l = 0; l < lanes; ++l) {
                    out[oy * ow + ox + l] = chunk[l];
                }
            }
        }
    } else {
        for (int32_t oy = 0; oy < oh; ++oy) {
            for (int32_t ox = 0; ox < ow; ++ox) {
                int32_t best = kLowestKey;
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
                        const int32_t k = key(static_cast<float>(in[y * iw + x]));
                        best = k > best ? k : best;
                    }
                }
                out[oy * ow + ox] = float_from_order_key(best);
            }
        }
    }
}

} // namespace
#endif

extern "C" {

/**
 * @brief Reduces each k1 x k0 window of one input channel-plane to an output element.
 *
 * Mirrors ggml_compute_forward_pool_2d for a single channel-plane. Padding is
 * handled by skipping out-of-bounds taps rather than gathering a padded
 * buffer: for MAX this is equivalent to -inf padding, and for AVG the divisor
 * is still the full k0*k1 window area (not the count of in-bounds taps),
 * matching the GGML CPU reference bit-for-bit.
 *
 * @param[in]  in   Input channel-plane of iw * ih elements (row-major, width fastest).
 * @param[out] out  Output channel-plane of ow * oh elements.
 * @param[in]  iw   Input width.
 * @param[in]  ih   Input height.
 * @param[in]  ow   Output width.
 * @param[in]  oh   Output height.
 * @param[in]  k0   Kernel width.
 * @param[in]  k1   Kernel height.
 * @param[in]  s0   Stride along width.
 * @param[in]  s1   Stride along height.
 * @param[in]  p0   Padding along width.
 * @param[in]  p1   Padding along height.
 * @param[in]  op   Pooling op: GGML_OP_POOL_MAX (0) for max, otherwise average (pool_2d.py
 *                  passes 1 and rejects anything else).
 */
void ggml_op_pool_2d(const INPUT_DTYPE * __restrict in,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t iw,
                     int32_t ih,
                     int32_t ow,
                     int32_t oh,
                     int32_t k0,
                     int32_t k1,
                     int32_t s0,
                     int32_t s1,
                     int32_t p0,
                     int32_t p1,
                     int32_t op) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

#if __AIEARCH__ == 20 && defined(GGML_POOL_OP)
    if constexpr (GGML_POOL_OP == GGML_OP_POOL_MAX) {
        pool_2d_max<INPUT_DTYPE>(in, out);
        event1();
        return;
    }
#endif

    const int32_t offset0 = -p0;
    const int32_t offset1 = -p1;

    const bool is_max = (op == GGML_OP_POOL_MAX);

    for (int32_t oy = 0; oy < oh; ++oy) {
        for (int32_t ox = 0; ox < ow; ++ox) {
            const int32_t ix = offset0 + ox * s0;
            const int32_t iy = offset1 + oy * s1;

            float res;
            if (is_max) {
                res = std::numeric_limits<float>::lowest();
                for (int32_t ky = 0; ky < k1; ++ky) {
                    const int32_t y = iy + ky;
                    if (y < 0 || y >= ih) {
                        continue;
                    }
                    const auto * srow = in + static_cast<int32_t>(y) * iw;
                    for (int32_t kx = 0; kx < k0; ++kx) {
                        const int32_t x = ix + kx;
                        if (x < 0 || x >= iw) {
                            continue;
                        }
                        const auto v = static_cast<float>(srow[x]);
                        res = (v > res) ? v : res;
                    }
                }
            } else {
                res = 0.0f;
                for (int32_t ky = 0; ky < k1; ++ky) {
                    const int32_t y = iy + ky;
                    if (y < 0 || y >= ih) {
                        continue;
                    }
                    const auto * srow = in + static_cast<int32_t>(y) * iw;
                    for (int32_t kx = 0; kx < k0; ++kx) {
                        const int32_t x = ix + kx;
                        if (x < 0 || x >= iw) {
                            continue;
                        }
                        res += static_cast<float>(srow[x]);
                    }
                }
                res *= 1.0f / static_cast<float>(k0 * k1);
            }

            out[oy * ow + ox] = res;
        }
    }

    event1();
}

} // extern "C"
