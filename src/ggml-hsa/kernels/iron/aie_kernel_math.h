/*
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#pragma once

#include <algorithm>
#include <cstring>

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

/**
 * @brief Computes the exponential function using range reduction.
 *
 * Implements exp(x) = 2^(x * log2(e)) = 2^n * 2^f where n is the integer part
 * and f is the fractional part. The 2^f term is computed via a Taylor series
 * after converting back to natural base.
 *
 * @param[in] x Input value (clamped to [-88, 88] to avoid overflow).
 * @return exp(x), with a floor of 1e-38f for very small results.
 */
inline float scalar_exp(float x) {
    // Clamp to avoid overflow/underflow
    x = std::clamp(x, -88.0f, 88.0f);

    // Range reduction: exp(x) = 2^(x * log2(e)) = 2^n * 2^f
    constexpr float log2e = 1.4426950408889634f;
    float t = x * log2e;
    int32_t n = static_cast<int32_t>(t);
    if (t < static_cast<float>(n))
        n--;
    float f = t - static_cast<float>(n);

    // Convert fractional part back to natural base: 2^f = exp(f * ln2)
    constexpr float ln2 = 0.6931471805599453f;
    float r = f * ln2;

    // Taylor series for exp(r) where r is in [0, ln2)
    float poly =
        1.0f +
        r * (1.0f + r * (0.5f + r * (0.166666667f +
                                     r * (0.041666667f + r * (0.008333333f + r * 0.001388889f)))));

    // Compute 2^n via bit manipulation
    n = (n < -126) ? -126 : ((n > 127) ? 127 : n);
    int32_t bits = (127 + n) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(float));

    float result = poly * scale;
    return (result < 1e-38f) ? 1e-38f : result;
}

/**
 * @brief Computes the natural logarithm using IEEE 754 range reduction.
 *
 * Implements ln(x) = ln(m * 2^e) = ln(m) + e * ln(2), where m is the mantissa
 * normalized to [1, 2). The ln(m) is computed using a 2*atanh series:
 * ln(m) = 2 * atanh((m-1)/(m+1)) with a polynomial approximation.
 *
 * @param[in] x The input value (must be positive).
 *
 * @return The natural logarithm of x. Returns -88.0f for x <= 0.
 */
inline float scalar_log(float x) {
    if (x <= 0.0f)
        return -88.0f;

    int32_t bits = reinterpret_cast<int32_t &>(x);
    int32_t e_int = ((bits >> 23) & 0xFF) - 127;
    float e = static_cast<float>(e_int);

    int32_t m_bits = (bits & 0x007FFFFF) | 0x3F800000;
    float m = reinterpret_cast<float &>(m_bits);

    float z = (m - 1.0f) / (m + 1.0f);
    float z2 = z * z;

    float poly = 0.0909090909f;       // 1/11
    poly = poly * z2 + 0.1111111111f; // 1/9
    poly = poly * z2 + 0.1428571429f; // 1/7
    poly = poly * z2 + 0.2000000000f; // 1/5
    poly = poly * z2 + 0.3333333333f; // 1/3
    poly = poly * z2 + 1.0f;

    float ln_m = 2.0f * (z * poly);

    constexpr float ln2 = 0.6931471805599453f;
    return ln_m + (e * ln2);
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
 * @tparam VecSize The SIMD vector width
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
aie::vector<float, VecSize> vec_exp(aie::vector<float, VecSize> & x) {
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

#pragma unroll
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
    // split x into integer and fractional parts
    int32_t i = static_cast<int32_t>(x);
    if (x < static_cast<float>(i)) {
        i--;
    }
    float f = x - static_cast<float>(i);

    constexpr float pow2_coeffs[] = {
        0.0000000070549116f, // ln(2)^10 / 10!
        0.0000001017808600f, // ln(2)^9 / 9!
        0.0000013215486790f, // ln(2)^8 / 8!
        0.0000152525277765f, // ln(2)^7 / 7!
        0.0001540353039338f, // ln(2)^6 / 6!
        0.0013333558146428f, // ln(2)^5 / 5!
        0.0096181291076285f, // ln(2)^4 / 4!
        0.0555041086648216f, // ln(2)^3 / 3!
        0.2402265069591007f, // ln(2)^2 / 2!
        0.6931471805599453f, // ln(2)^1 / 1!
        1.0f                 // ln(2)^0 / 0!
    };
    constexpr int NUM_POW2_COEFFS = sizeof(pow2_coeffs) / sizeof(pow2_coeffs[0]);

    // compute 2^f using Horner's method for Taylor series of exp(f * ln(2))
    float exp_f = pow2_coeffs[0];

#pragma unroll
    for (int j = 1; j < NUM_POW2_COEFFS; ++j) {
        exp_f = exp_f * f + pow2_coeffs[j];
    }

    // this takes a couple of cycles to compute 2^i using IEEE 754 bit manipulation
    // IEEE 754 float: 2^i is represented as exponent = 127 + i, mantissa = 0
    // create the integer representation of 2^i
    int32_t bits = (127 + i) << 23;
    // cast the bits directly to float
    float scale = reinterpret_cast<float &>(bits);

    return exp_f * scale;
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

    uint32_t n_head_log2 = 1u << floor_log2((uint32_t)n_head);

    // compute base values m0 and m1
    float m0 = pow2(-max_bias / n_head_log2);
    float m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (head_idx < n_head_log2) {
        // slope = m0^(head_idx+1) via repeated multiplication
        slope = m0;
        for (int32_t j = 0; j < head_idx; ++j) {
            slope *= m0;
        }
    } else {
        // slope = m1^(2*(head_idx - n_head_log2) + 1)
        uint32_t exp = 2 * (head_idx - n_head_log2) + 1;
        slope = m1;
        for (uint32_t j = 1; j < exp; ++j) {
            slope *= m1;
        }
    }

    return slope;
}
