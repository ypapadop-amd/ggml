// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file clamp.cc
 * @brief Clamp operation for AIE kernels.
 */

#include <bit>
#include <cstdint>
#include <type_traits>

#include "ggml-aie.hpp"
#include "transform.h"

extern "C" {

/**
 * @brief Clamps each element to a specified range: out[i] = max(min(in[i], max_val), min_val).
 *
 * The bounds apply in the CPU backend's order, MAX(MIN(x, max), min), so a range with
 * min_val > max_val gives min_val everywhere, as it does there.
 *
 * f32 compares on @ref float_order_key instead of as floats: the aie2 vector unit has no f32
 * min/max (aie::min/max subtract in the accumulator and test the sign after a bf16 conversion) and
 * the scalar unit has no float compare at all (a runtime call). The key orders non-NaN floats as
 * `<` does, -0 equal to +0, and the result is selected from the input or bound bits, so every
 * non-NaN result is the CPU's bit for bit. A NaN with the sign bit clear clamps to max_val, as on
 * the CPU; one with it set clamps to min_val, where the CPU gives max_val.
 *
 * @param[in]  in      Input array of N elements.
 * @param[out] out     Output array of N elements.
 * @param[in]  N       Number of elements to process.
 * @param[in]  min_val Minimum allowed value (inclusive).
 * @param[in]  max_val Maximum allowed value (inclusive).
 */
void ggml_op_clamp(const INPUT_DTYPE * __restrict in,
                   OUTPUT_DTYPE * __restrict out,
                   int32_t N,
                   float min_val,
                   float max_val) {
    // Aligned accesses only when clamp.py guaranteed the tile is a whole number of vectors.
    constexpr bool aligned = tile_vector_aligned_v<OUTPUT_DTYPE>;
    if constexpr (std::is_same_v<INPUT_DTYPE, float> && std::is_same_v<OUTPUT_DTYPE, float>) {
        // On aie2p the floats are walked as their int32 bits: float_order_keys masks a float
        // vector's bits with 0x7fffffff, which LLVM folds into a vector fabs that aie2p runs one
        // lane at a time. The key of negative bits, -(bits & 0x7fffffff), is INT32_MIN - bits.
#if __AIEARCH__ == 20
        const std::int32_t klo = float_order_key(min_val);
        const std::int32_t khi = float_order_key(max_val);
        transform_vector_n<aligned>(
            out, N,
            [=](auto v) {
                const auto k = float_order_keys<static_cast<int>(decltype(v)::size())>(v);
                const auto m = aie::select(max_val, v, aie::lt(k, khi));
                return aie::select(min_val, m, aie::gt(aie::min(k, khi), klo));
            },
            [=](float v) {
                const std::int32_t k = float_order_key(v);
                const float m = k < khi ? v : max_val;
                return (k < khi ? k : khi) > klo ? m : min_val;
            },
            in);
#else  // __AIEARCH__ != 20
       // MAX(MIN(x,max),min) is x > min ? (x >= max ? max : x) : min, and whenever min < max
       // (x >= max then implies x > min). Each test becomes one compare of x's bits against a
       // threshold, signed or unsigned by the bound's sign: signed int32 order matches float
       // order on non-negative floats, and unsigned order reverses float order on negative ones
       // and puts every non-negative float below them.
       //
       // - x >= max: max > 0, bits >= max's bits signed (-0 and every negative float are negative
       //   ints); max a zero, bits <= 0x80000000 unsigned (every non-negative float and -0); max
       //   < 0, bits <= max's bits unsigned.
       // - x > min: min a zero, bits > 0 signed; min > 0, bits > min's bits signed; min < 0, bits
       //   < min's bits unsigned. When min is not below max the result is min everywhere: the
       //   threshold is set so the test is never true (bits > INT32_MAX signed, never).
       //
       // The threshold can differ from the bound's own bits (the zero and "not below" cases
       // above), so the output keeps the bound's bits separately: selecting the threshold itself
       // would store -0 for a +0 bound, or garbage for a min-not-below-max range.
       //
       // Every non-NaN result is the CPU's bit for bit (a zero bound included). A NaN with the
       // sign bit clear clamps to max_val, as on the CPU; one with it set to min_val, where the
       // CPU gives max_val.
        const std::int32_t blo = std::bit_cast<std::int32_t>(min_val);
        const std::int32_t bhi = std::bit_cast<std::int32_t>(max_val);
        const bool hi_signed = bhi > 0;
        const std::int32_t hi_thresh = bhi == 0 ? INT32_MIN : bhi;
        const bool below = float_order_key(min_val) < float_order_key(max_val);
        const bool lo_signed = !below || blo >= 0 || blo == INT32_MIN;
        const std::int32_t lo_thresh = !below ? INT32_MAX : (blo == INT32_MIN ? 0 : blo);
        const auto uhi = static_cast<std::uint32_t>(hi_thresh);
        const auto ulo = static_cast<std::uint32_t>(lo_thresh);

        const auto run = [&](auto hi_signed_c, auto lo_signed_c) {
            constexpr bool HiSigned = decltype(hi_signed_c)::value;
            constexpr bool LoSigned = decltype(lo_signed_c)::value;
            transform_vector_n<aligned>(
                reinterpret_cast<std::int32_t *>(out), N,
                [=](auto v) {
                    const auto u = v.template cast_to<std::uint32_t>();
                    const auto at_hi = [&] {
                        if constexpr (HiSigned) {
                            return aie::ge(v, hi_thresh);
                        } else {
                            return aie::le(u, uhi);
                        }
                    }();
                    const auto above_lo = [&] {
                        if constexpr (LoSigned) {
                            return aie::gt(v, lo_thresh);
                        } else {
                            return aie::lt(u, ulo);
                        }
                    }();
                    return aie::select(blo, aie::select(v, bhi, at_hi), above_lo);
                },
                [=](std::int32_t b) {
                    const auto u = static_cast<std::uint32_t>(b);
                    const bool at_hi = HiSigned ? b >= hi_thresh : u <= uhi;
                    const bool above_lo = LoSigned ? b > lo_thresh : u < ulo;
                    return above_lo ? (at_hi ? bhi : b) : blo;
                },
                reinterpret_cast<const std::int32_t *>(in));
        };
        if (hi_signed) {
            lo_signed ? run(std::true_type{}, std::true_type{})
                      : run(std::true_type{}, std::false_type{});
        } else {
            lo_signed ? run(std::false_type{}, std::true_type{})
                      : run(std::false_type{}, std::false_type{});
        }
#endif // __AIEARCH__ == 20
    } else {
        const auto lo = static_cast<INPUT_DTYPE>(min_val);
        const auto hi = static_cast<INPUT_DTYPE>(max_val);
        transform_vector_n<aligned>(
            out, N, [=](auto v) { return aie::max(aie::min(v, hi), lo); },
            [=](INPUT_DTYPE v) {
                const INPUT_DTYPE m = v < hi ? v : hi;
                return static_cast<OUTPUT_DTYPE>(m > lo ? m : lo);
            },
            in);
    }
}

} // extern "C"
