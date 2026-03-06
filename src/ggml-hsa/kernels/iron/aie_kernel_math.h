/*
    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

#ifndef _AIE_KERNEL_MATH_
#define _AIE_KERNEL_MATH_

// =============================================================================
// Range-reduced vectorized exponential: exp(x) = 2^n * exp(r)
//
// Cody-Waite splitting of ln(2) into high and low parts minimises
// rounding error in step 2. A degree-13 polynomial on [0, ln2) gives
// ~6e-13 peak error.
// =============================================================================
template <int VecSize = KERN_VEC_SIZE>
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
    constexpr int NUM_EXP_COEFFS = sizeof(exp_coeffs) / sizeof(exp_coeffs[0]);

    aie::vector<float, VecSize> poly = aie::broadcast<float, VecSize>(exp_coeffs[0]);
    aie::accum<accfloat, VecSize> tmp;

#pragma unroll
    for (int i = 1; i < NUM_EXP_COEFFS; ++i) {
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

#endif