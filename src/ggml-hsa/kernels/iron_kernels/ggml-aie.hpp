// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

/**
 * @file ggml-aie.hpp
 * @brief Common type definitions and utilities for AIE kernels.
 *
 * This header provides type aliases and type traits used across AIE kernels.
 */

#include <bit>
#include <cstdint>
#include <type_traits>

#include "aie_api/aie.hpp"

using i8 = std::int8_t;   ///< Signed 8-bit integer type alias.
using i16 = std::int16_t; ///< Signed 16-bit integer type alias.
using i32 = std::int32_t; ///< Signed 32-bit integer type alias.
using bf16 = bfloat16;    ///< Brain floating-point 16-bit type alias.
using f32 = float;        ///< 32-bit floating-point type alias.

/**
 * @brief Type trait to check if a type is a floating-point type.
 *
 * This extends std::is_floating_point to also recognize bfloat16 as a
 * floating-point type, which is commonly used in AIE computations.
 *
 * @tparam T The type to check.
 *
 * Usage:
 * @code
 * static_assert(is_floating_point<float>::value);    // true
 * static_assert(is_floating_point<bfloat16>::value); // true
 * static_assert(!is_floating_point<int>::value);     // int is not floating-point
 * @endcode
 */
template <typename T>
struct is_floating_point
    : public std::integral_constant<bool,
                                    std::is_floating_point_v<T> || std::is_same_v<T, bfloat16>> {};

/**
 * @brief Helper variable template for is_floating_point.
 *
 * @tparam T The type to check.
 *
 * @return true if T is a floating-point type (including bfloat16), false otherwise.
 */
template <typename T>
constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * @brief Unsigned integer type with the same width as @p T.
 *
 * Lets a value be reinterpreted as raw bits so it can be manipulated with integer operations --
 * clearing or flipping a float's sign bit, for instance. Specialized for every element type alias
 * above; an unlisted type is a compile error rather than a silent mismatch.
 *
 * @tparam T The type whose width is matched.
 */
template <typename T>
struct same_width_uint;

template <>
struct same_width_uint<i8> {
    using type = std::uint8_t;
};

template <>
struct same_width_uint<i16> {
    using type = std::uint16_t;
};

template <>
struct same_width_uint<i32> {
    using type = std::uint32_t;
};

template <>
struct same_width_uint<bf16> {
    using type = std::uint16_t;
};

template <>
struct same_width_uint<f32> {
    using type = std::uint32_t;
};

/**
 * @brief Helper alias for same_width_uint.
 *
 * @tparam T The type whose width is matched.
 */
template <typename T>
using same_width_uint_t = typename same_width_uint<T>::type;

/**
 * @brief Converts one f32 element to bf16 bits (round-to-nearest-even, NaN -> quiet).
 *
 * Replicates the host @c ggml_compute_fp32_to_bf16 integer arithmetic bit-for-bit, so vectorized
 * and scalar paths (and the host reference) all agree. Returns the raw 16-bit pattern; the caller
 * stores it into a bf16. Used for the scalar tail of any f32 -> bf16 kernel.
 *
 * @param[in] v The f32 value to convert.
 * @return The bf16 bit pattern as a uint16_t.
 */
inline std::uint16_t convert_f32_to_bf16_scalar(f32 v) {
    union {
        f32 f;
        std::uint32_t u;
    } bits;
    bits.f = v;
    if ((bits.u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<std::uint16_t>((bits.u >> 16) | 64u);
    }
    return static_cast<std::uint16_t>((bits.u + (0x7fffu + ((bits.u >> 16) & 1u))) >> 16);
}

/**
 * @brief Widens one f16 bit pattern to f32.
 *
 * IEEE half -> single is exact for every input (single has both more exponent range and more
 * mantissa bits), so this is pure integer bit manipulation with no rounding: rebias the exponent,
 * normalize subnormals, and pass inf/NaN payloads through. Written against the raw 16-bit pattern
 * because IRON has no f16 element type -- f16 tensors are streamed as i16 and reinterpreted here.
 *
 * Deliberately integer-only rather than a transcription of the host @c ggml_compute_fp16_to_fp32,
 * which reaches its result via a float multiply and therefore inherits the hardware's NaN-quieting
 * behaviour. The two agree bit-for-bit on every finite input, and on NaN inputs they differ only in
 * the mantissa MSB, which the following @c convert_f32_to_bf16_scalar sets unconditionally (its
 * `| 64` quieting). See test-convert-hsa, which checks all 65536 patterns.
 *
 * @param[in] h The f16 bit pattern.
 * @return The widened f32 value.
 */
inline f32 convert_f16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;

    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            // Subnormal: shift the mantissa up until its implicit leading bit is in place, and
            // lower the exponent by the same amount. f16 bias 15 -> f32 bias 127 gives 113 for the
            // smallest normal exponent here.
            std::uint32_t shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3FFu;
            bits = ((113u - shift) << 23) | (mantissa << 13);
        }
        // mantissa == 0 leaves bits == 0, i.e. a signed zero
    } else if (exponent == 0x1Fu) {
        bits = 0x7F800000u | (mantissa << 13); // inf / NaN, payload preserved
    } else {
        bits = ((exponent + 112u) << 23) | (mantissa << 13); // rebias: -15 + 127
    }

    union {
        std::uint32_t u;
        f32 f;
    } out;
    out.u = sign | bits;
    return out.f;
}

/**
 * @brief Narrows one f32 element to f16 bits (round-to-nearest-even, NaN -> quiet).
 *
 * Integer-only replica of the host @c ggml_compute_fp32_to_fp16: overflow (and infinity) saturates
 * to inf, an out-of-range small magnitude falls through the f16 subnormal range to a signed zero,
 * and a NaN keeps its sign but is quieted to 0x7e00 with its payload dropped -- exactly what the
 * host reference does. Returns the raw 16-bit pattern; the caller stores it into the i16 the f16
 * tensor is streamed as (IRON has no f16 element type).
 *
 * Deliberately integer-only rather than a transcription of the host reference, which reaches its
 * result through float multiplies and an add whose rounding would be the AIE's rather than x86's.
 * The two were checked to agree on all 2^32 f32 bit patterns; see also test-convert-hsa, which
 * checks all 65536 bf16 patterns through this on the device.
 *
 * @param[in] v The f32 value to convert.
 * @return The f16 bit pattern as a uint16_t.
 */
inline std::uint16_t convert_f32_to_f16_scalar(f32 v) {
    union {
        f32 f;
        std::uint32_t u;
    } bits;
    bits.f = v;

    const std::uint32_t sign = (bits.u >> 16) & 0x8000u;
    const std::uint32_t mag = bits.u & 0x7fffffffu;

    if (mag > 0x7f800000u) {
        return static_cast<std::uint16_t>(sign | 0x7e00u);
    }

    const std::int32_t exponent = static_cast<std::int32_t>(mag >> 23) - 127;
    const std::uint32_t mantissa = mag & 0x7fffffu;

    if (exponent >= 16) {
        // above the largest f16 magnitude, and infinity itself
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        // normal f16: keep 10 mantissa bits, round the dropped 13 to nearest-even. A carry out of
        // the mantissa walks into the exponent field on its own, and at the top of the range that
        // is exactly the 0x7bff -> 0x7c00 (inf) rounding the host produces.
        std::uint32_t h = (static_cast<std::uint32_t>(exponent + 15) << 10) | (mantissa >> 13);
        const std::uint32_t rem = mantissa & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) {
            ++h;
        }
        return static_cast<std::uint16_t>(sign | h);
    }
    if (mag < 0x00800000u) {
        // f32 subnormal, i.e. below 2^-126: far under the smallest f16 subnormal, so a signed zero
        return static_cast<std::uint16_t>(sign);
    }

    // f16 subnormal: the result is |v| / 2^-24 rounded to nearest-even, so shift the 24-bit
    // significand down by shift and round on the discarded bits. shift >= 25 cannot round up to
    // the smallest subnormal, and would be an out-of-range shift below, so it leaves a signed zero.
    const std::uint32_t shift = static_cast<std::uint32_t>(-exponent - 1);
    if (shift >= 25) {
        return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t significand = 0x800000u | mantissa;
    std::uint32_t h = significand >> shift;
    const std::uint32_t rem = significand & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u) != 0u)) {
        ++h;
    }
    return static_cast<std::uint16_t>(sign | h);
}

/**
 * @brief Lanes that @ref convert_bf16_to_f16_vector converts correctly.
 *
 * The vector bf16 -> f16 path covers the two cases that need no rounding and no range handling: a
 * result that is a normal f16 (biased bf16 exponent in [113, 142], i.e. an unbiased exponent in
 * [-14, 15]), and a signed zero. Everything else -- an f16 subnormal, an overflow to infinity, an
 * infinity, or a NaN -- is left to the scalar path, which is the reference for those. Separating
 * the predicate from the conversion lets the caller check a whole block at once and only fall back
 * where it must.
 *
 * @tparam V Vector width (number of 16-bit lanes).
 * @param[in] b The raw bf16 bit patterns.
 * @return Mask of the lanes the vector path handles.
 */
template <int V>
inline aie::mask<V> convert_bf16_to_f16_vector_covers(const aie::vector<std::uint16_t, V> & b) {
    // A biased exponent in [113, 142] is a magnitude in [0x3880, 0x477f]: one unsigned compare
    // after an offset that wraps everything below the range to the top.
    const aie::vector<std::uint16_t, V> mag = aie::bit_and(std::uint16_t{0x7fff}, b);
    return aie::le(aie::sub(mag, std::uint16_t{0x3880}), std::uint16_t{0x477f - 0x3880}) |
           aie::eq(mag, std::uint16_t{0});
}

/**
 * @brief Converts a vector of @p V bf16 lanes to f16 bits, for the lanes the predicate covers.
 *
 * bf16 keeps 7 stored mantissa bits and f16 keeps 10, so a result that lands in the normal f16
 * range loses nothing: the conversion is a pure re-pack of the sign, a rebiased exponent and a
 * left-shifted mantissa, with no rounding to get wrong. Rebiasing the exponent field (127 -> 15) is
 * subtracting 112 << 7 from the magnitude, and moving the fields up by 3 bits is three doublings,
 * so no vector shift is needed (on aie2 each one is a round trip through an accumulator).
 *
 * Lanes outside @ref convert_bf16_to_f16_vector_covers get a meaningless value here; the caller
 * must not use the result unless the mask is full.
 *
 * @tparam V Vector width (number of 16-bit lanes).
 * @param[in] b The raw bf16 bit patterns.
 * @return The f16 bit patterns.
 */
template <int V>
inline aie::vector<std::uint16_t, V>
convert_bf16_to_f16_vector(const aie::vector<std::uint16_t, V> & b) {
    const aie::vector<std::uint16_t, V> sign = aie::bit_and(std::uint16_t{0x8000}, b);
    const aie::vector<std::uint16_t, V> mag = aie::bit_and(std::uint16_t{0x7fff}, b);

    const aie::vector<std::uint16_t, V> x1 = aie::sub(mag, std::uint16_t{112 << 7});
    const aie::vector<std::uint16_t, V> x2 = aie::add(x1, x1);
    const aie::vector<std::uint16_t, V> x4 = aie::add(x2, x2);
    const aie::vector<std::uint16_t, V> packed = aie::add(x4, x4);

    // select(v, a, m) yields a where m holds: a zero magnitude keeps only its sign
    return aie::bit_or(sign, aie::select(packed, std::uint16_t{0}, aie::eq(mag, std::uint16_t{0})));
}

/**
 * @brief Lanes that @ref convert_f16_bits_to_bf16_vector converts correctly.
 *
 * A normal f16 (biased exponent in [1, 30], a magnitude in [0x0400, 0x7bff]) and a signed zero.
 * An f16 subnormal, infinity or NaN is left to the scalar path, which is the reference for those.
 *
 * @tparam V Vector width (number of 16-bit lanes).
 * @param[in] h The raw f16 bit patterns.
 * @return Mask of the lanes the vector path handles.
 */
template <int V>
inline aie::mask<V>
convert_f16_bits_to_bf16_vector_covers(const aie::vector<std::uint16_t, V> & h) {
    const aie::vector<std::uint16_t, V> mag = aie::bit_and(std::uint16_t{0x7fff}, h);
    return aie::le(aie::sub(mag, std::uint16_t{0x0400}), std::uint16_t{0x7bff - 0x0400}) |
           aie::eq(mag, std::uint16_t{0});
}

/**
 * @brief Converts a vector of @p V f16 bit patterns to bf16 bits, for the lanes the predicate
 * covers.
 *
 * Lane-wise replica of @c convert_f32_to_bf16_scalar(convert_f16_bits_to_f32(h)) for a normal f16
 * or a zero: widening to f32 is exact, so the f32 is the sign, the exponent rebiased by 112 and the
 * 10 mantissa bits at the top of the f32 mantissa, and narrowing to bf16 drops the low 3 of them,
 * rounding to nearest even. The magnitude's exponent and mantissa fields are contiguous, so that
 * is the magnitude divided by 8 and rounded to nearest even -- one shift out of an accumulator in
 * the conv_even rounding mode, set here and restored on return -- plus the rebias, 112 << 7. A
 * carry out of the mantissa increments the exponent, as it does in the scalar arithmetic.
 *
 * Lanes outside @ref convert_f16_bits_to_bf16_vector_covers get a meaningless value here; the
 * caller must not use the result unless the mask is full.
 *
 * @tparam V Vector width (number of 16-bit lanes).
 * @param[in] h The raw f16 bit patterns.
 * @return The bf16 bit patterns.
 */
template <int V>
inline aie::vector<std::uint16_t, V>
convert_f16_bits_to_bf16_vector(const aie::vector<std::uint16_t, V> & h) {
    const aie::vector<std::uint16_t, V> sign = aie::bit_and(std::uint16_t{0x8000}, h);
    const aie::vector<std::uint16_t, V> mag = aie::bit_and(std::uint16_t{0x7fff}, h);

    aie::vector<std::uint16_t, V> rounded;
    {
        aie::scoped_mode rnd{aie::rounding_mode::conv_even};
        rounded = aie::accum<acc32, V>(mag).template to_vector<std::uint16_t>(3);
    }
    const aie::vector<std::uint16_t, V> packed = aie::add(rounded, std::uint16_t{112 << 7});

    // select(v, a, m) yields a where m holds: a zero magnitude keeps only its sign
    return aie::bit_or(sign, aie::select(packed, std::uint16_t{0}, aie::eq(mag, std::uint16_t{0})));
}

/**
 * @brief Loads the first @p n (< V) elements at @p p into a vector whose other lanes are zero.
 *
 * Reads exactly @p n elements, so nothing past the end of a row or buffer is touched: the neighbour
 * may be the other half of a ping-pong ObjectFifo. Goes through an aligned bounce buffer.
 */
template <int V, typename T>
inline aie::vector<T, V> load_v_partial(const T * p, std::int32_t n) {
    alignas(aie::vector_decl_align) T buf[V] = {};
    for (std::int32_t j = 0; j < n; ++j) {
        buf[j] = p[j];
    }
    return aie::load_v<V>(buf);
}

/// Stores the first @p n (< V) lanes of @p v at @p p, writing nothing past them.
template <int V, typename T>
inline void store_v_partial(T * p, const aie::vector<T, V> & v, std::int32_t n) {
    alignas(aie::vector_decl_align) T buf[V];
    aie::store_v(buf, v);
    for (std::int32_t j = 0; j < n; ++j) {
        p[j] = buf[j];
    }
}

/**
 * @brief Stores the first @p N lanes of @p v at @p p, writing nothing past them.
 *
 * Aligned stores of @p v or of its halves, quarters, ... no wider than @p G bytes and no narrower
 * than 128 bits, then scalar stores for what is left: no bounce buffer (unlike @ref
 * store_v_partial) and no unaligned vector store (broken for 16-bit types). @p p must be aligned to
 * min(@p G, the size of @p v).
 */
template <int N, int G = 64, unsigned V, typename T>
inline void store_v_prefix(T * p, const aie::vector<T, V> & v) {
    constexpr int bytes = static_cast<int>(V * sizeof(T));
    static_assert(N >= 0 && N <= static_cast<int>(V), "store_v_prefix stores 0 to V lanes");
    if constexpr (N == static_cast<int>(V) && bytes <= G) {
        aie::store_v(p, v);
    } else if constexpr (N > 0) {
        constexpr unsigned H = V / 2;
        if constexpr (bytes <= 16) {
            for (int j = 0; j < N; ++j) {
                p[j] = v[j];
            }
        } else if constexpr (N >= static_cast<int>(H)) {
            store_v_prefix<H, G>(p, v.template extract<H>(0));
            store_v_prefix<N - static_cast<int>(H), G>(p + H, v.template extract<H>(1));
        } else {
            store_v_prefix<N, G>(p, v.template extract<H>(0));
        }
    }
}

/**
 * @brief A float's order key: its signed magnitude, as int32.
 *
 * Orders non-NaN floats exactly as `<` does, with -0 and +0 equal (both 0), so kernels can compare
 * floats on the integer unit: the aie2 scalar unit has no float compare, and `a < b` on float is a
 * runtime call there. A NaN's key lies beyond +inf's or -inf's; callers that can see one must
 * handle it themselves (@ref float_is_nan).
 */
inline std::int32_t float_order_key(float f) {
    const std::int32_t bits = std::bit_cast<std::int32_t>(f);
    const std::int32_t magnitude = bits & 0x7fffffff;
    return bits < 0 ? -magnitude : magnitude;
}

/// Whether @p f is a NaN, on the integer unit.
inline bool float_is_nan(float f) {
    return (std::bit_cast<std::int32_t>(f) & 0x7fffffff) > 0x7f800000;
}

/// The float whose @ref float_order_key is @p k (+0 for key 0).
inline float float_from_order_key(std::int32_t k) {
    return std::bit_cast<float>(k < 0 ? (-k | INT32_MIN) : k);
}

/// @ref float_order_key lane-wise.
template <int V>
inline aie::vector<std::int32_t, V> float_order_keys(const aie::vector<float, V> & v) {
    const auto bits = v.template cast_to<std::int32_t>();
    const auto magnitude = aie::bit_and(0x7fffffff, bits);
    return aie::select(magnitude, aie::neg(magnitude),
                       aie::lt(bits, aie::zeros<std::int32_t, V>()));
}

/// @ref float_is_nan lane-wise.
template <int V>
inline aie::mask<V> floats_are_nan(const aie::vector<float, V> & v) {
    return aie::gt(aie::bit_and(0x7fffffff, v.template cast_to<std::int32_t>()), 0x7f800000);
}

/// @ref float_from_order_key lane-wise.
template <int V>
inline aie::vector<float, V> floats_from_order_keys(const aie::vector<std::int32_t, V> & k) {
    const auto bits = aie::select(k, aie::bit_or(INT32_MIN, aie::neg(k)),
                                  aie::lt(k, aie::zeros<std::int32_t, V>()));
    return bits.template cast_to<float>();
}

/**
 * @brief Converts a vector of @p V f32 lanes to bf16 (round-to-nearest-even, NaN -> quiet).
 *
 * Lane-wise replica of @c convert_f32_to_bf16_scalar: applies the exact RNE integer arithmetic of
 * @c ggml_compute_fp32_to_bf16 rather than relying on hardware rounding/NaN handling, so the result
 * is bit-identical to the scalar path and the host reference. The bf16 bits land in the low 16 of
 * each u32 lane; @c filter_even grabs those low halves.
 *
 * @tparam V Vector width (number of f32 lanes).
 * @param[in] fv The f32 vector to convert.
 * @return The converted bf16 vector.
 */
template <int V>
inline aie::vector<bf16, V> convert_f32_to_bf16_vector(const aie::vector<f32, V> & fv) {
    const aie::vector<std::uint32_t, V> u = aie::vector_cast<std::uint32_t>(fv);
    const aie::vector<std::uint32_t, V> hi16 = aie::logical_downshift(u, 16);

    // NaN: (u >> 16) | 64
    const aie::vector<std::uint32_t, V> nan_val = aie::bit_or(64u, hi16);

    // RNE: (u + (0x7fff + ((u >> 16) & 1))) >> 16
    const aie::vector<std::uint32_t, V> lsb = aie::bit_and(1u, hi16);
    const aie::vector<std::uint32_t, V> rounded = aie::add(u, aie::add(lsb, 0x7fffu));
    const aie::vector<std::uint32_t, V> rne_val = aie::logical_downshift(rounded, 16);

    // nan_mask ? nan_val : rne_val   (select(v1, v2, m) == m ? v2 : v1)
    const auto nan_mask = aie::gt(aie::bit_and(0x7fffffffu, u), 0x7f800000u);
    const aie::vector<std::uint32_t, V> res32 = aie::select(rne_val, nan_val, nan_mask);

    const aie::vector<std::uint16_t, V> res16 =
        aie::filter_even(aie::vector_cast<std::uint16_t>(res32));
    return aie::vector_cast<bf16>(res16);
}

/**
 * @brief Converts a vector of f32 to bf16 like @ref convert_f32_to_bf16_vector, bit for bit, in
 * fewer vector operations.
 *
 * It never shifts the 32-bit lanes right (on aie2 that is a round trip through the accumulator,
 * which also saves and restores the rounding mode): the result is the high 16-bit half of each
 * rounded lane, taken with @c filter_odd; the NaN case sets the quiet bit in the high half
 * (@c 0x400000); and the lane's bit 16, the rounding's tie-breaker, is moved to bit 0 by shifting
 * the 16-bit view down one lane. aie2: 10 vector ALU operations per vector, II 14 -> 10 in
 * convert_pad's row loop.
 *
 * @tparam V Vector width (number of f32 lanes).
 * @param[in] fv The f32 vector to convert.
 * @return The converted bf16 vector.
 */
template <int V>
inline aie::vector<bf16, V> convert_f32_to_bf16_vector_hi16(const aie::vector<f32, V> & fv) {
    const aie::vector<std::uint32_t, V> u = aie::vector_cast<std::uint32_t>(fv);

    // (u >> 16) & 1, from the 16-bit view shifted down one lane (lane 2i then holds lane i's high
    // half; the odd lanes are masked off)
    const aie::vector<std::uint32_t, V> lsb = aie::bit_and(
        1u,
        aie::vector_cast<std::uint32_t>(aie::shuffle_down(aie::vector_cast<std::uint16_t>(u), 1)));

    // RNE: u + (0x7fff + lsb); NaN: u | (64 << 16). The result is each lane's high half.
    const aie::vector<std::uint32_t, V> rounded = aie::add(u, aie::add(lsb, 0x7fffu));
    const aie::vector<std::uint32_t, V> nan_val = aie::bit_or(0x400000u, u);
    const auto nan_mask = aie::gt(aie::bit_and(0x7fffffffu, u), 0x7f800000u);
    const aie::vector<std::uint32_t, V> res32 = aie::select(rounded, nan_val, nan_mask);

    return aie::vector_cast<bf16>(aie::filter_odd(aie::vector_cast<std::uint16_t>(res32)));
}

/**
 * @brief 1 / sqrt(a) per lane, for positive normal @p a, within 8e-7 relative.
 *
 * The bit-trick estimate aie2's aie::invsqrt uses for f32 (http://rrrola.wz.cz/inv_sqrt.html, up to
 * 6.5e-4 relative), written with the portable aie API so it builds for every target (the aie2p
 * f32 aie::invsqrt is a scalar call per lane), then one Newton step, y * (1.5 - 0.5 * a * y * y),
 * which squares that error. Checked in f32 numpy over 1e-30..1e30: at most 7.6e-7 relative.
 */
template <unsigned N>
inline aie::vector<float, N> invsqrt_newton(const aie::vector<float, N> & a) {
    constexpr std::int32_t c1 = 0x5F1FFFF9;
    constexpr float c2 = 0.703952253f;
    constexpr float c3 = 2.38924456f;
    const auto mul = [](const aie::vector<float, N> & x, const auto & y) {
        return aie::mul(x, y).template to_vector<float>();
    };
    const aie::vector<float, N> y0 = aie::vector_cast<float>(aie::sub(
        aie::broadcast<std::int32_t, N>(c1), aie::downshift(aie::vector_cast<std::int32_t>(a), 1)));
    const aie::vector<float, N> y =
        mul(y0, aie::sub(aie::broadcast<float, N>(c2 * c3), mul(mul(a, c2), mul(y0, y0))));
    return mul(y, aie::sub(aie::broadcast<float, N>(1.5f), mul(mul(a, 0.5f), mul(y, y))));
}
