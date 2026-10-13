/*
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#ifndef AIE_KERNEL_MATH
#define AIE_KERNEL_MATH

#include <stdint.h>

#include <aie_api/aie.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

/**
 * @brief Computes exp(x) using range reduction.
 *
 * Uses the identity exp(x) = 2^(x * log2(e)) = 2^n * 2^f
 * where n = floor(x * log2(e)) and f is the fractional part.
 *
 * @param[in] x The input value.
 * @return The computed value of exp(x).
 */
inline float scalar_exp(float x) {
    // Clamp to avoid overflow/underflow
    if (x < -88.0f)
        x = -88.0f;
    if (x > 88.0f)
        x = 88.0f;

    constexpr float log2e = 1.4426950408889634f;
    float t = x * log2e;

    // n = floor(t)
    int32_t n = static_cast<int32_t>(t);
    if (t < static_cast<float>(n))
        n--;

    float f = t - static_cast<float>(n);

    // Compute 2^f using polynomial approximation for exp(f * ln2)
    // where f is in [0, 1), so f*ln2 is in [0, 0.693)
    constexpr float ln2 = 0.6931471805599453f;
    float y = f * ln2;

    // Taylor series for exp(y) with more terms for better precision
    // exp(y) = 1 + y + y^2/2! + y^3/3! + y^4/4! + y^5/5! + y^6/6! + y^7/7!
    float exp_f =
        1.0f + y * (1.0f + y * (0.5f + y * (0.16666666666666666f +
                                            y * (0.041666666666666664f +
                                                 y * (0.008333333333333333f +
                                                      y * (0.001388888888888889f +
                                                           y * 0.0001984126984126984f))))));

    // Compute 2^n using IEEE 754 bit manipulation
    if (n < -126)
        n = -126;
    if (n > 127)
        n = 127;
    int32_t bits = (127 + n) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));

    return exp_f * scale;
}

/**
 * @brief Checks if a floating-point value is infinite.
 *
 * @param[in] x The input value.
 * @return @c true if x is infinite, @c false otherwise.
 */
inline bool is_infinite(float x) { return x != 0.0f && x + x == x; }

/**
 * @brief Computes the natural logarithm using IEEE 754 range reduction.
 *
 * Implements ln(x) = ln(m * 2^e) = ln(m) + e * ln(2), where m is the mantissa
 * normalized to [1, 2). The ln(m) is computed using a 2*atanh series:
 * ln(m) = 2 * atanh((m-1)/(m+1)) with a polynomial approximation.
 *
 * @param[in] x The input value.
 *
 * @return The natural logarithm of x.
 *         Returns -inf for x == 0, and quiet NaN for x < 0, matching logf().
 */
inline float scalar_log(float x) {
    // Handle NaN and infinities explicitly to match logf() behavior
    if (std::isnan(x)) {
        return x;
    }
    if (is_infinite(x)) {
        if (x > 0.0f) {
            return x; // +inf
        } else {
            return std::numeric_limits<float>::quiet_NaN(); // logf(-inf) -> NaN
        }
    }

    if (x == 0.0f) {
        return -std::numeric_limits<float>::infinity();
    }
    if (x < 0.0f) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    int32_t bits;
    std::memcpy(&bits, &x, sizeof(float)); // Avoid strict aliasing issues

    const int32_t e_int = ((bits >> 23) & 0xFF) - 127;
    float e = static_cast<float>(e_int);

    const int32_t m_bits = (bits & 0x007FFFFF) | 0x3F800000;
    float m;
    std::memcpy(&m, &m_bits, sizeof(float));

    const float z = (m - 1.0f) / (m + 1.0f);
    const float z2 = z * z;

    float poly = 0.0909090909f;       // 1/11
    poly = poly * z2 + 0.1111111111f; // 1/9
    poly = poly * z2 + 0.1428571429f; // 1/7
    poly = poly * z2 + 0.2000000000f; // 1/5
    poly = poly * z2 + 0.3333333333f; // 1/3
    poly = poly * z2 + 1.0f;

    const auto ln_m = 2.0f * (z * poly);
    constexpr float ln2 = 0.6931471805599453f;
    return ln_m + (e * ln2);
}

/**
 * @brief Computes the natural logarithm of every lane, for positive normal finite lanes.
 *
 * Vector form of @ref scalar_log without its special cases, so it calls no runtime routine:
 * ln(x) = ln(m) + e * ln(2), with x = m * 2^e and m folded into [sqrt(1/2), sqrt(2)), and ln(m)
 * by the Cephes logf polynomial in f = m - 1 (no divide); ln(2) is split in two parts so that the
 * large part of e * ln(2) is exact.
 *
 * Zero, negative, subnormal, infinite and NaN lanes give unspecified values; callers that can see
 * them must handle them (@ref scalar_log does).
 *
 * @tparam V Vector width.
 * @param[in] x The input vector.
 * @return The element-wise natural logarithm.
 */
template <unsigned V>
inline aie::vector<float, V> vec_log(const aie::vector<float, V> & x) {
    using fvec = aie::vector<float, V>;
    using ivec = aie::vector<int32_t, V>;
    const auto mul = [](const fvec & a, const auto & b) {
        return aie::mul(a, b).template to_vector<float>();
    };

    const ivec bits = aie::vector_cast<int32_t>(x);
    const ivec mantissa = aie::bit_and(int32_t{0x007FFFFF}, bits);
    // m >= sqrt(2) (mantissa bits >= 0x3504F3) is halved into [sqrt(1/2), 1) and its exponent
    // raised by one. Adds build the bit patterns rather than bit_or: aie2p cannot legalize the
    // vector OR.
    const auto halve = aie::ge(mantissa, aie::broadcast<int32_t, V>(0x003504F3));
    const fvec m = aie::vector_cast<float>(
        aie::add(mantissa, aie::select(aie::broadcast<int32_t, V>(0x3F800000),
                                       aie::broadcast<int32_t, V>(0x3F000000), halve)));
    const ivec e_int =
        aie::add(aie::sub(aie::downshift(bits, 23), aie::broadcast<int32_t, V>(127)),
                 aie::select(aie::zeros<int32_t, V>(), aie::broadcast<int32_t, V>(1), halve));
    // int -> float of the small exponent by the 1.5 * 2^23 magic number (exact for |e| < 2^22).
    constexpr int32_t magic_bits = 0x4B400000; // 1.5 * 2^23
    const fvec e =
        aie::sub(aie::vector_cast<float>(aie::add(e_int, aie::broadcast<int32_t, V>(magic_bits))),
                 aie::broadcast<float, V>(12582912.0f));

    const fvec f = aie::sub(m, aie::broadcast<float, V>(1.0f));
    const fvec z = mul(f, f);

    constexpr float log_coeffs[] = {7.0376836292e-2f,  -1.1514610310e-1f, 1.1676998740e-1f,
                                    -1.2420140846e-1f, 1.4249322787e-1f,  -1.6668057665e-1f,
                                    2.0000714765e-1f,  -2.4999993993e-1f, 3.3333331174e-1f};
    constexpr int32_t NUM_LOG_COEFFS = sizeof(log_coeffs) / sizeof(log_coeffs[0]);
    fvec poly = aie::broadcast<float, V>(log_coeffs[0]);
    AIE_LOOP_UNROLL_FULL
    for (int32_t i = 1; i < NUM_LOG_COEFFS; ++i) {
        poly = aie::add(mul(poly, f), aie::broadcast<float, V>(log_coeffs[i]));
    }

    constexpr float ln2_lo = -2.12194440e-4f; // ln(2) = ln2_hi + ln2_lo
    constexpr float ln2_hi = 0.693359375f;    // exact in 9 bits, so e * ln2_hi is exact
    fvec y = mul(mul(poly, f), z);
    y = aie::add(y, mul(e, ln2_lo));
    y = aie::sub(y, mul(z, 0.5f));
    return aie::add(aie::add(f, y), mul(e, ln2_hi));
}

/**
 * @brief Computes floor(log2(x)) for positive integers.
 *
 * Finds the position of the most significant bit set in x.
 *
 * @param[in] x The input value (must be > 0 for meaningful result).
 *
 * @return The floor of log base 2 of x. Returns 0 for x <= 1.
 */
inline uint32_t floor_log2(uint32_t x) {
    uint32_t result = 0;
    while (x > 1) {
        x >>= 1;
        result++;
    }
    return result;
}

/**
 * @brief Computes 2^x using range reduction for improved precision.
 *
 * Horner's method suffers from precision loss for large input values.
 * This function applies range reduction by splitting x into integer (i) and
 * fractional (f) parts where i = floor(x) and f is in [0, 1).
 * Formula: 2^x = 2^i * 2^f
 *
 * The fractional part 2^f is computed using a degree-10 Taylor series
 * approximation of exp(f * ln(2)) via Horner's method. The integer part 2^i
 * is computed using IEEE 754 bit manipulation.
 *
 * @param[in] x The exponent value.
 *
 * @return The computed value of 2^x.
 */
inline float pow2(float x) {
    int32_t i = static_cast<int32_t>(x);
    if (x < static_cast<float>(i))
        i--;
    float f = x - static_cast<float>(i);

    // 2^f = exp(f * ln2) using polynomial
    constexpr float ln2 = 0.6931471805599453f;
    float y = f * ln2;
    float exp_f =
        1.0f + y * (1.0f + y * (0.5f + y * (0.16666666666666666f +
                                            y * (0.041666666666666664f +
                                                 y * (0.008333333333333333f +
                                                      y * (0.001388888888888889f +
                                                           y * 0.0001984126984126984f))))));

    if (i < -126)
        i = -126;
    if (i > 127)
        i = 127;
    int32_t bits = (127 + i) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(scale));

    return exp_f * scale;
}

/**
 * @brief Computes the vectorized exponential function exp(x) for AIE.
 *
 * This function implements a range-reduced exponential using the identity
 * exp(x) = 2^n * exp(r), where n = floor(x * log2(e)) and r = x - n * ln(2).
 *
 * The implementation uses:
 * - Cody-Waite splitting of ln(2) into high and low parts to minimize
 *   rounding error during range reduction.
 * - A degree-13 polynomial approximation of exp(r) on [0, ln2) with
 *   approximately 6e-13 peak error.
 * - IEEE 754 bit manipulation to compute 2^n efficiently.
 *
 * @tparam VecSize The SIMD vector width.
 *
 * @param[in,out] x Input vector of float values. The input is clamped to
 *                  [-88, 88] to avoid overflow/underflow in float32. The
 *                  vector may be modified during computation.
 *
 * @return A vector of float values containing exp(x) for each element.
 *         Results are clamped to a minimum of 1e-38 to avoid exact zero
 *         from underflow.
 */
template <int32_t VecSize>
inline aie::vector<float, VecSize> vec_exp(aie::vector<float, VecSize> & x) {
    constexpr float log2e = 1.4426950408889634f; // log2(e)
    // Cody-Waite split of ln(2) for accurate range reduction
    constexpr float ln2_hi = 6.93145751953125e-1f;      // upper bits, exact in float
    constexpr float ln2_lo = 1.4286068203094172321e-6f; // residual

    // Clamp to representable range of exp() in float32
    x = aie::max(x, aie::broadcast<float, VecSize>(-88.0f));
    x = aie::min(x, aie::broadcast<float, VecSize>(88.0f));

    // Compute t = x * log2(e)
    aie::accum<accfloat, VecSize> t_acc = aie::mul(x, log2e);
    aie::vector<float, VecSize> t = t_acc.template to_vector<float>();

    // n = floor(t) via magic-number rounding
    constexpr float magic = 12582912.0f; // 1.5 * 2^23
    aie::vector<float, VecSize> v_magic = aie::broadcast<float, VecSize>(magic);
    aie::vector<float, VecSize> n_f = aie::sub(aie::add(t, v_magic), v_magic);

    // Adjust round-to-nearest -> floor: if n_f > t, we rounded up
    auto overshot = aie::lt(t, n_f);
    n_f = aie::sub(n_f, aie::select(aie::broadcast<float, VecSize>(0.0f),
                                    aie::broadcast<float, VecSize>(1.0f), overshot));

    // r = x - n * ln(2) with Cody-Waite precision
    aie::accum<accfloat, VecSize> hi_acc = aie::mul(n_f, ln2_hi);
    aie::vector<float, VecSize> r = aie::sub(x, hi_acc.template to_vector<float>());
    aie::accum<accfloat, VecSize> lo_acc = aie::mul(n_f, ln2_lo);
    r = aie::sub(r, lo_acc.template to_vector<float>());

    // Evaluate exp(r) using Horner's method (degree 13)
    //
    // exp(r) ~ 1 + r + r^2/2! + r^3/3! + ... + r^13/13!
    // Coefficients in high-to-low order for Horner evaluation:
    constexpr float exp_coeffs[] = {
        0.0000000001605904f, // 1/13!
        0.0000000020876757f, // 1/12!
        0.0000000250521084f, // 1/11!
        0.0000002755731922f, // 1/10!
        0.0000027557319224f, // 1/9!
        0.0000248015873016f, // 1/8!
        0.0001984126984127f, // 1/7!
        0.0013888888888889f, // 1/6!
        0.0083333333333333f, // 1/5!
        0.0416666666666667f, // 1/4!
        0.1666666666666667f, // 1/3!
        0.5f,                // 1/2!
        1.0f,                // 1/1!
        1.0f                 // 1/0!
    };
    constexpr int32_t NUM_EXP_COEFFS = sizeof(exp_coeffs) / sizeof(exp_coeffs[0]);

    aie::vector<float, VecSize> poly = aie::broadcast<float, VecSize>(exp_coeffs[0]);
    aie::accum<accfloat, VecSize> tmp;

    AIE_LOOP_UNROLL_FULL
    for (int32_t i = 1; i < NUM_EXP_COEFFS; ++i) {
        tmp = aie::mul(poly, r);
        poly = aie::add(tmp.template to_vector<float>(),
                        aie::broadcast<float, VecSize>(exp_coeffs[i]));
    }

    // Compute 2^n via IEEE 754 bit construction
    n_f = aie::max(n_f, aie::broadcast<float, VecSize>(-126.0f));
    n_f = aie::min(n_f, aie::broadcast<float, VecSize>(127.0f));

    // aie::to_fixed<int32_t>(n_f, 23) = floor(n_f * 2^23) = n << 23
    auto n_shifted = aie::to_fixed<int32_t>(n_f, 23);
    // (n + 127) << 23 = (n << 23) + (127 << 23)
    auto scale_bits = aie::add(n_shifted, aie::broadcast<int32_t, VecSize>(0x3F800000));
    aie::vector<float, VecSize> scale = scale_bits.template cast_to<float>();

    // Reconstruct exp(x) = exp(r) * 2^n
    aie::accum<accfloat, VecSize> result_acc = aie::mul(poly, scale);
    aie::vector<float, VecSize> result = result_acc.template to_vector<float>();

    // Clamp to positive minimum (avoid exact zero from underflow)
    result = aie::max(result, aie::broadcast<float, VecSize>(1e-38f));

    return result;
}

/// @ref vec_exp_nonpositive on up to 32 lanes, always inlined so that its caller can interleave
/// independent calls.
template <int32_t VecSize>
__attribute__((always_inline)) inline aie::vector<float, VecSize>
vec_exp_nonpositive_lanes(aie::vector<float, VecSize> x) {
    using vec = aie::vector<float, VecSize>;
    constexpr float log2e = 1.4426950408889634f;
    // Cody-Waite split of ln(2): n * ln2_hi is exact for the n here.
    constexpr float ln2_hi = 6.93145751953125e-1f;
    constexpr float ln2_lo = 1.4286068203094172321e-6f;
    constexpr float magic = 12582912.0f; // 1.5 * 2^23

    x = aie::max(x, aie::broadcast<float, VecSize>(-86.0f));

    // n = floor(t), t = x * log2(e) + 1/2: round with the magic number, then step down where that
    // rounded up.
    aie::accum<accfloat, VecSize> t_acc;
    t_acc.from_vector(aie::broadcast<float, VecSize>(0.5f));
    const vec t = aie::mac(t_acc, x, log2e).template to_vector<float>();
    const vec v_magic = aie::broadcast<float, VecSize>(magic);
    vec n_f = aie::sub(aie::add(t, v_magic), v_magic);
    n_f = aie::sub(n_f, aie::select(aie::zeros<float, VecSize>(),
                                    aie::broadcast<float, VecSize>(1.0f), aie::lt(t, n_f)));

    vec r = aie::sub(x, aie::mul(n_f, ln2_hi).template to_vector<float>());
    r = aie::sub(r, aie::mul(n_f, ln2_lo).template to_vector<float>());

    // exp(r) by Horner's rule, 1/7! down to 1/0!.
    constexpr float coeffs[] = {1.0f / 5040.0f, 1.0f / 720.0f, 1.0f / 120.0f, 1.0f / 24.0f,
                                1.0f / 6.0f,    0.5f,          1.0f,          1.0f};
    vec poly = aie::broadcast<float, VecSize>(coeffs[0]);
    AIE_LOOP_UNROLL_FULL
    for (int32_t i = 1; i < static_cast<int32_t>(sizeof(coeffs) / sizeof(coeffs[0])); ++i) {
        aie::accum<accfloat, VecSize> acc;
        acc.from_vector(aie::broadcast<float, VecSize>(coeffs[i]));
        poly = aie::mac(acc, poly, r).template to_vector<float>();
    }

    // poly * 2^n: n << 23 added to the exponent field (n_f is a whole number, so exact).
    return aie::add(poly.template cast_to<int32_t>(), aie::to_fixed<int32_t>(n_f, 23))
        .template cast_to<float>();
}

/**
 * @brief Computes exp(x) for x <= 0, the arguments of softmax and log-sum-exp once the row
 * maximum is subtracted, in fewer steps than @ref vec_exp.
 *
 * exp(x) = 2^n * exp(r) with n = floor(x * log2(e) + 1/2), so r = x - n * ln(2) lies in
 * [-ln2/2, ln2/2], where a degree-7 Taylor polynomial is within 6e-9 of exp(r) (against degree 13
 * on [0, ln2) in @ref vec_exp). 2^n is applied by adding n to the result's exponent field. Every
 * fp32 multiply here is emulated and dominates the cost: 10 of them, against 17 in @ref vec_exp.
 *
 * Each step waits on the previous one, so one call leaves most of the core idle: 64 lanes run as
 * two independent 32-lane halves, which the scheduler interleaves (887 bundles against 674 for 32
 * lanes on aie2; 32 is the widest fp32 accumulator).
 *
 * @tparam VecSize The SIMD vector width: up to 32, or 64.
 * @param[in] x Arguments, <= 0. Below -86 they are taken as -86: exp(-86) ~ 4.5e-38 is still a
 *              normal float, so the exponent field cannot underflow. A positive argument is
 *              outside the contract.
 * @return exp(x) in each lane.
 */
template <int32_t VecSize>
inline aie::vector<float, VecSize> vec_exp_nonpositive(const aie::vector<float, VecSize> & x) {
    if constexpr (VecSize > 32) {
        static_assert(VecSize == 64, "vec_exp_nonpositive takes up to 32 lanes, or 64");
        return aie::concat(vec_exp_nonpositive_lanes<32>(x.template extract<32>(0)),
                           vec_exp_nonpositive_lanes<32>(x.template extract<32>(1)));
    } else {
        return vec_exp_nonpositive_lanes<VecSize>(x);
    }
}

#if __AIE_ARCH__ != 20

/**
 * @brief Applies the GELU activation (tanh approximation) to 32 lanes; aie2p form.
 *
 * Same sigmoid form as the aie2 version below, x / (1 + exp(-2y)), rearranged so that every step
 * is a multiply, an add or a bit operation:
 *
 * - Only e = exp(-2|y|) = 2^-w <= 1 is computed, w = 2 log2(e) |y| clamped at 125:
 *   gelu(x) = x / (1 + e) for x >= 0 and x e / (1 + e) for x < 0. 2^-w = 2^-n 2^-r, with n the
 *   nearest integer by the 1.5 * 2^23 magic number (whose low bits then hold n, so 2^-n is a
 *   subtraction from the exponent field), and 2^-r on [-1/2, 1/2] by a degree-5 minimax
 *   polynomial (relative error 1.5e-7). The clamp compares w's bits as integers, so it needs no
 *   emulated f32 min/max and takes infinite and NaN magnitudes too.
 * - 1 / (1 + e), on [1, 2], is the linear minimax seed 24/17 - 8/17 d (relative error 1/17)
 *   refined by three Newton steps, instead of aie::inv, which aie2p runs as one scalar inverse per
 *   lane.
 * - The emulated f32 products are aie2p's "fast" six bf16 partial products, not aie::mul's nine,
 *   and fewer where a later step corrects or damps the error.
 *
 * Within 1.5e-6 relative of the exact tanh form, against ggml's CPU f32 GELU's fp16 table at
 * 3.8e-3. Always inlined: Peano otherwise calls it once per vector.
 *
 * @param[in] x The input vector.
 * @return The element-wise GELU.
 */
__attribute__((always_inline)) inline aie::vector<float, 32>
vec_gelu_32(const aie::vector<float, 32> & x) {
    using fvec = aie::vector<float, 32>;
    using ivec = aie::vector<int32_t, 32>;
    using acc = aie::accum<accfloat, 32>;
    // Emulated f32 products, at two accuracies: "fast", six of the nine bf16 partial products
    // of aie::mul (within ~1 ulp), and "low", three (relative error ~2^-16), where a later step
    // corrects or damps the error.
    const auto mul = [](const fvec & a, const fvec & b) -> fvec {
        return acc(::mul_elem_32_accuracy_fast(a, b)).template to_vector<float>();
    };
    const auto mul_low = [](const fvec & a, const fvec & b) -> fvec {
        return acc(::mul_elem_32_accuracy_low(a, b)).template to_vector<float>();
    };
    // c + a * b at low accuracy.
    const auto mac_low = [](float c, const fvec & a, const fvec & b) -> fvec {
        return acc(::mac_elem_32_accuracy_low(a, b, aie::broadcast<float, 32>(c), 0, 0, 0))
            .template to_vector<float>();
    };
    // c - a * b at low accuracy.
    const auto msc_low = [](float c, const fvec & a, const fvec & b) -> fvec {
        return acc(::msc_elem_32_accuracy_low(a, b, aie::broadcast<float, 32>(c), 0, 0, 0))
            .template to_vector<float>();
    };
    const auto bc = [](float c) { return aie::broadcast<float, 32>(c); };
    const auto to_bf16 = [](const fvec & v) { return acc(v).template to_vector<bfloat16>(); };

    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCoefA = 0.044715f;
    constexpr float kLog2e = 1.4426950408889634f;
    // 2y log2(e) = x (k1 + k2 x^2).
    constexpr float kK1 = 2.0f * kLog2e * kSqrt2OverPi;
    constexpr float kK2 = kK1 * kCoefA;
    // w = 2 |y| log2(e) is clamped at 125: 2^-125 and 2^-n 2^-r are still normal floats.
    constexpr int32_t kWMaxBits = __builtin_bit_cast(int32_t, 125.0f);
    constexpr float kMagic = 12582912.0f; // 1.5 * 2^23
    constexpr uint32_t kTenBits = __builtin_bit_cast(uint32_t, 10.0f);
    constexpr uint32_t kSignBit = 0x80000000u;
    // 2^-r on [-1/2, 1/2], highest degree first: the minimax 2^r's with odd terms negated.
    constexpr float kExp2Coeffs[] = {-1.3276466634e-3f, 9.6755418926e-3f,  -5.5507134646e-2f,
                                     2.4022120237e-1f,  -6.9314694405e-1f, 1.0000001192e+0f};

    const ivec x_bits = aie::vector_cast<int32_t>(x);
    const ivec abs_bits =
        aie::bit_and(int32_t{0x7FFFFFFF},
                     aie::vector_cast<int32_t>(mul(x, aie::add(mul(mul(x, x), bc(kK2)), kK1))));
    const fvec w = aie::vector_cast<float>(
        aie::select(abs_bits, aie::broadcast<int32_t, 32>(kWMaxBits),
                    aie::gt(abs_bits, aie::broadcast<int32_t, 32>(kWMaxBits))));

    // w = n + r: w + magic rounds to the nearest integer n, held in its low bits.
    const fvec wm = aie::add(w, kMagic);
    const fvec r = aie::sub(w, aie::sub(wm, kMagic));
    // Horner's rule; |r| <= 1/2 damps the low-accuracy error of the inner steps below 1e-7.
    fvec p = bc(kExp2Coeffs[0]);
    AIE_LOOP_UNROLL_FULL
    for (int32_t i = 1; i < 4; ++i) {
        p = mac_low(kExp2Coeffs[i], p, r);
    }
    p = aie::add(mul(p, r), kExp2Coeffs[4]);
    p = aie::add(mul(p, r), kExp2Coeffs[5]);
    // e = 2^-r * 2^-n: n << 23 taken off the exponent field. The magic number's bits shift out.
    const fvec e = aie::vector_cast<float>(
        aie::sub(aie::vector_cast<int32_t>(p), aie::upshift(aie::vector_cast<int32_t>(wm), 23)));

    // q = 1 / d, d = 1 + e in [1, 2]. Newton's step q (2 - d q) squares the relative error, so
    // the seed and the first step run in bf16 and the second at low accuracy (1/17, ~7e-3,
    // ~7e-5), and only the last needs the fast product.
    const fvec d = aie::add(e, 1.0f);
    const auto d16 = to_bf16(d);
    fvec q = aie::mac(acc(bc(24.0f / 17.0f)), d16, static_cast<bfloat16>(-8.0f / 17.0f))
                 .template to_vector<float>();
    const auto q16 = to_bf16(q);
    q = aie::mul(q16, to_bf16(aie::msc(acc(bc(2.0f)), d16, q16).template to_vector<float>()))
            .template to_vector<float>();
    q = mul_low(q, msc_low(2.0f, d, q));
    q = mul(q, aie::sub(bc(2.0f), mul(d, q)));

    // y < 0 (the sign of x): x e / (1 + e), else x / (1 + e).
    const fvec num = aie::select(x, mul(x, e), aie::lt(x_bits, aie::zeros<int32_t, 32>()));
    // As ggml's CPU GELU: x for x >= 10, 0 for x <= -10, decided on the bits of x (unsigned,
    // the negatives order above the positives). This also keeps infinities, whose emulated
    // product is NaN, and x^3 overflowing out of the result.
    const auto ux = aie::vector_cast<uint32_t>(x);
    fvec y = aie::select(mul(num, q), x,
                         aie::ge(ux, aie::broadcast<uint32_t, 32>(kTenBits)) &
                             aie::lt(ux, aie::broadcast<uint32_t, 32>(kSignBit)));
    return aie::select(y, aie::zeros<float, 32>(),
                       aie::ge(ux, aie::broadcast<uint32_t, 32>(kSignBit | kTenBits)));
}

/**
 * @brief Applies the GELU activation (tanh approximation) to every lane; see @ref vec_gelu_32.
 *
 * Narrower vectors run at 32 lanes: the emulated f32 multiply costs the same at 16 lanes as at 32.
 *
 * @tparam VecSize Vector width, up to 32.
 * @param[in] x The input vector.
 * @return The element-wise GELU.
 */
template <unsigned VecSize>
__attribute__((always_inline)) inline aie::vector<float, VecSize>
vec_gelu(const aie::vector<float, VecSize> & x) {
    static_assert(VecSize <= 32, "vec_gelu takes up to 32 lanes");
    if constexpr (VecSize < 32) {
        return vec_gelu_32(x.template grow<32>()).template extract<VecSize>(0);
    } else {
        return vec_gelu_32(x);
    }
}

#else // __AIE_ARCH__ == 20

/**
 * @brief Applies the GELU activation (tanh approximation) to every lane.
 *
 * Vector counterpart of the scalar path in unary_ops.cc, computing
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3))).
 *
 * Evaluated in the algebraically equivalent sigmoid form
 *   0.5 * x * (1 + tanh(y)) == x * sigmoid(2y) == x / (1 + exp(-2y)),
 * which needs one exp and one reciprocal per lane instead of the scalar form's exp, divide and
 * two sign branches. The scalar form takes |y| and re-applies the sign for stability against
 * exp() overflow; that is unnecessary here because @ref vec_exp clamps its argument to
 * [-88, 88], so a large negative y saturates the denominator and the result goes to zero rather
 * than to NaN, and a large positive y leaves the denominator at 1 so the result approaches x.
 *
 * @tparam VecSize Vector width.
 * @param[in] x The input vector.
 * @return The element-wise GELU.
 */
template <unsigned VecSize>
inline aie::vector<float, VecSize> vec_gelu(const aie::vector<float, VecSize> & x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCoefA = 0.044715f;

    // y = sqrt(2/pi) * (x + 0.044715 * x^3)
    const aie::vector<float, VecSize> x2 = aie::mul(x, x).template to_vector<float>();
    const aie::vector<float, VecSize> x3 = aie::mul(x2, x).template to_vector<float>();
    const aie::vector<float, VecSize> inner =
        aie::add(x, aie::mul(x3, kCoefA).template to_vector<float>());

    // t = -2y, then exp(t). vec_exp clamps t at 88, but 1 + e^88 is too large for aie::inv: on aie2
    // it returned NaN for every x <= -11, where GELU is -0. At t = 80 the reciprocal is still a
    // normal float, so the result is a tiny negative number instead.
    aie::vector<float, VecSize> t =
        aie::mul(inner, -2.0f * kSqrt2OverPi).template to_vector<float>();
    t = aie::min(t, 80.0f);
    const aie::vector<float, VecSize> e = vec_exp<static_cast<int32_t>(VecSize)>(t);

    return aie::mul(x, aie::inv(aie::add(e, 1.0f))).template to_vector<float>();
}

#endif // __AIE_ARCH__ != 20

/**
 * @brief Returns |v| for one element.
 *
 * For floats this clears the sign bit, which is exactly @c std::fabs -- the reference ggml computes
 * on the host -- for every input, including -0.0 (which a @c v < 0 ? -v : v test leaves negative)
 * and NaN. Integer types keep the ordinary comparison form.
 *
 * @tparam T Element type.
 * @param[in] v The value to take the magnitude of.
 * @return The magnitude of @p v.
 */
template <typename T>
inline T scalar_abs(T v) {
    if constexpr (is_floating_point_v<T>) {
        using U = same_width_uint_t<T>;
        constexpr U magnitude_mask = static_cast<U>(~(U{1} << (sizeof(U) * 8 - 1)));
        U bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        bits &= magnitude_mask;
        T result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    } else {
        return v < T(0) ? -v : v;
    }
}

/**
 * @brief Returns |v| for every lane, by clearing the sign bit.
 *
 * Exactly @c fabs for every input, so signed zeros and NaNs come back with the sign cleared like
 * the host reference. Deliberately not @c aie::max(v, aie::neg(v)): that picks whichever operand
 * compares larger, which for -0.0 and NaN can keep the input's sign. Also avoids @c aie::abs, which
 * does not compute a floating-point magnitude in this aie_api version (it returned 0.875 for -5.0f
 * on aie2). Pure integer bit manipulation, so it needs no architecture split.
 *
 * @tparam T Float element type.
 * @tparam V Vector width.
 * @param[in] v The vector to take the magnitude of.
 * @return The element-wise magnitude.
 */
template <typename T, unsigned V>
inline aie::vector<T, V> vec_abs(const aie::vector<T, V> & v) {
    using U = same_width_uint_t<T>;
    constexpr U magnitude_mask = static_cast<U>(~(U{1} << (sizeof(U) * 8 - 1)));
    return aie::vector_cast<T>(aie::bit_and(magnitude_mask, aie::vector_cast<U>(v)));
}

// Constants of the inverse square root seed and first refinement step, from
// http://rrrola.wz.cz/inv_sqrt.html, as used by aie_api's aie2 InvSqrt (mlir-aie). The seed plus
// this step is accurate to ~6.5e-4 relative. The sqrt helpers below add a Newton step on the
// inverse and then correct the result, r + y/2 * (x - r^2): within 1 ulp of the correctly rounded
// sqrt for every non-negative float, subnormals included, with or without flush-to-zero (85%
// exact), where the Newton step alone leaves up to 12 ulp.
inline constexpr uint32_t kInvSqrtMagic = 0x5F1FFFF9u;
inline constexpr float kInvSqrtC2 = 0.703952253f;
inline constexpr float kInvSqrtC3 = 2.38924456f;

// The AIE vector unit flushes subnormal inputs *and* intermediates to zero, and its float path
// does not carry -0.0 through. So special values are classified on the bit pattern, and every
// x < 2^-64 is first scaled by 2^150 (result by 2^-75) so no intermediate goes subnormal: below
// that, 0.5 * x or the residual x - r^2 would, losing the Newton step or the final correction.
inline constexpr uint32_t kSqrtSignBit = 0x80000000u;
inline constexpr uint32_t kSqrtMinNormalBits = 0x00800000u;
inline constexpr uint32_t kSqrtScaleBelowBits = 0x1F800000u; // 2^-64
inline constexpr uint32_t kSqrtInfBits = 0x7F800000u;
inline constexpr uint32_t kSqrtQuietNaNBits = 0x7FC00000u;
// A normal is scaled by adding 150 to its exponent field. A subnormal is m * 2^-149 with m < 2^23,
// so x * 2^150 = 2m, built exactly from normal floats as the bit pattern of 2^24 + 2m minus 2^24.
inline constexpr uint32_t kSqrtScaleExponentBits = 150u << 23;
inline constexpr uint32_t kSqrtTwo24Bits = 0x4B800000u;
inline constexpr float kSqrtTwo24 = 0x1p24f;
inline constexpr float kSqrtUnscale = 0x1p-75f;

/**
 * @brief Computes sqrt(x) for 2^-64 <= x <= FLT_MAX from its approximate inverse square root.
 */
inline float scalar_sqrt_in_range(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    bits = kInvSqrtMagic - (bits >> 1);
    float y;
    std::memcpy(&y, &bits, sizeof(y));
    // (x * y) * y rather than x * (y * y): y * y underflows for x near FLT_MAX.
    y = y * (kInvSqrtC2 * (kInvSqrtC3 - (x * y) * y));
    y = y * (1.5f - ((x * y) * y) * 0.5f);
    const float r = x * y;
    return r + (0.5f * y) * (x - r * r);
}

/**
 * @brief Computes sqrt(x) from invsqrt(x), within 1 ulp of the correctly rounded result.
 *
 * Scalar counterpart of @ref vec_sqrt, using the same arithmetic so a tile's scalar tail agrees
 * with its vector body. aie::sqrt is not usable: aie2 has no implementation of it at all.
 *
 * @param[in] x The input value.
 * @return sqrt(x); x for +/-0, NaN for any other negative input, and x for +inf and NaN.
 */
inline float scalar_sqrt(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    if ((bits & ~kSqrtSignBit) == 0) {
        return x;
    }
    if ((bits & kSqrtSignBit) != 0) {
        // The NaN is built from x's bits rather than returned as a constant. Under
        // GGML_HSA_KERNEL_INLINE, aiecc relinks this IR with its newer LLVM, which prints a NaN
        // phi operand as `+qnan`; Peano's opt cannot parse that, and aiecc does not rewrite it.
        // The local is not named `nan`: aiecc's rewrite of bare `nan` literals also hits `%nan`.
        const uint32_t nan_bits = bits | kSqrtQuietNaNBits;
        float result;
        std::memcpy(&result, &nan_bits, sizeof(result));
        return result;
    }
    if (bits >= kSqrtInfBits) {
        return x;
    }
    if (bits < kSqrtScaleBelowBits) {
        float scaled;
        if (bits < kSqrtMinNormalBits) {
            const uint32_t two_m_bits = kSqrtTwo24Bits + bits;
            std::memcpy(&scaled, &two_m_bits, sizeof(scaled));
            scaled -= kSqrtTwo24;
        } else {
            const uint32_t scaled_bits = bits + kSqrtScaleExponentBits;
            std::memcpy(&scaled, &scaled_bits, sizeof(scaled));
        }
        return scalar_sqrt_in_range(scaled) * kSqrtUnscale;
    }
    return scalar_sqrt_in_range(x);
}

/**
 * @brief Computes sqrt(x) for every lane from invsqrt(x); see @ref scalar_sqrt.
 *
 * Vectorized on both aie2 and aie2p, where aie::sqrt is scalar (aie2p) or missing (aie2). Special
 * values match @ref scalar_sqrt.
 *
 * @tparam V Vector width.
 * @param[in] x The input vector.
 * @return The element-wise square root.
 */
template <unsigned V>
inline aie::vector<float, V> vec_sqrt(const aie::vector<float, V> & x) {
    using ivec = aie::vector<int32_t, V>;
    auto mul = [](const aie::vector<float, V> & a, const auto & b) {
        return aie::mul(a, b).template to_vector<float>();
    };
    auto ibroadcast = [](uint32_t v) {
        return aie::broadcast<int32_t, V>(static_cast<int32_t>(v));
    };

    const ivec bits = aie::vector_cast<int32_t>(x);
    const ivec abs_bits = aie::bit_and(static_cast<int32_t>(~kSqrtSignBit), bits);

    // Zero lanes land in both masks; they are replaced at the end. An add builds 2^24 + 2m rather
    // than a bit_or (equal here, as m < 2^23), because aie2p cannot legalize the vector OR.
    const auto scale = aie::lt(abs_bits, ibroadcast(kSqrtScaleBelowBits));
    const auto subnormal = aie::lt(abs_bits, ibroadcast(kSqrtMinNormalBits));
    const aie::vector<float, V> two_m =
        aie::sub(aie::vector_cast<float>(aie::add(ibroadcast(kSqrtTwo24Bits), abs_bits)),
                 aie::broadcast<float, V>(kSqrtTwo24));
    const aie::vector<float, V> tiny_scaled =
        aie::vector_cast<float>(aie::add(abs_bits, ibroadcast(kSqrtScaleExponentBits)));
    // Negative lanes are computed on |x| (keeping the seed's shift logical) and replaced below.
    aie::vector<float, V> xn = aie::select(aie::vector_cast<float>(abs_bits), tiny_scaled, scale);
    xn = aie::select(xn, two_m, subnormal);

    aie::vector<float, V> y = aie::vector_cast<float>(
        aie::sub(ibroadcast(kInvSqrtMagic), aie::downshift(aie::vector_cast<int32_t>(xn), 1)));
    y = mul(y, mul(aie::sub(aie::broadcast<float, V>(kInvSqrtC3), mul(mul(xn, y), y)), kInvSqrtC2));
    y = mul(y, aie::sub(aie::broadcast<float, V>(1.5f), mul(mul(mul(xn, y), y), 0.5f)));

    aie::vector<float, V> r = mul(xn, y);
    r = aie::add(r, mul(mul(y, 0.5f), aie::sub(xn, mul(r, r))));
    r = aie::select(r, mul(r, kSqrtUnscale), scale);

    const ivec zero = aie::zeros<int32_t, V>();
    r = aie::select(r, x, aie::ge(abs_bits, ibroadcast(kSqrtInfBits)));
    r = aie::select(r, aie::broadcast<float, V>(std::numeric_limits<float>::quiet_NaN()),
                    aie::lt(bits, zero));
    r = aie::select(r, x, aie::eq(abs_bits, zero));
    return r;
}

// aie2p has no legalization rule for G_FNEG on vector types, so aie::neg on a float vector fails to
// compile there ("unable to legalize instruction: G_FNEG <16 x s32>" for f32, "<32 x s16>" for
// bf16). Flipping the sign bit with integer operations is exactly IEEE-754 negation and needs no
// such rule. aie2 does legalize it, so it keeps aie::neg -- no behaviour change on an architecture
// that cannot be tested here, and both forms are bit-exact for negation. Drop the split once the
// aie2p backend grows the missing rule.
//
// Still required as of llvm-aie 22.0.0.2026092301. A narrow probe (aie::neg on a vector straight
// out of load_v) does compile for aie2p on that version, which is misleading -- building the real
// GGML_UNARY_OP_NEG kernel through transform_vector_n still fails to legalize. Re-check by
// compiling unary_ops.cc for aie2p, not by a standalone snippet.
#if __AIE_ARCH__ != 20

/**
 * @brief Returns -v for every lane, by flipping the sign bit.
 *
 * @tparam T Float element type.
 * @tparam V Vector width.
 * @param[in] v The vector to negate.
 * @return The negated vector.
 */
template <typename T, unsigned V>
inline aie::vector<T, V> vec_neg(const aie::vector<T, V> & v) {
    using U = same_width_uint_t<T>;
    constexpr U sign_bit = static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    return aie::vector_cast<T>(aie::bit_xor(sign_bit, aie::vector_cast<U>(v)));
}

#else // __AIE_ARCH__ == 20

/**
 * @brief Returns -v for every lane. aie2 legalizes vector @c G_FNEG, so use it directly.
 *
 * @tparam T Float element type.
 * @tparam V Vector width.
 * @param[in] v The vector to negate.
 * @return The negated vector.
 */
template <typename T, unsigned V>
inline aie::vector<T, V> vec_neg(const aie::vector<T, V> & v) {
    return aie::neg(v);
}

#endif // __AIE_ARCH__ != 20

#endif // AIE_KERNEL_MATH
