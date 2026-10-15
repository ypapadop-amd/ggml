// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file scale.cc
 * @brief Scale and bias operation for AIE kernels.
 */

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"
#include "transform.h"

#if __AIEARCH__ != 20
namespace {

// aie2p has no f32 multiplier: an f32 aie::mul splits both operands into three bf16 limbs and adds
// their nine products one at a time, one long chain per 32 lanes. Here the scale is split once per
// call and x per vector, and the products are summed in three groups, smallest first, so the chain
// is shorter. The kernel rounds to nearest even: the accumulator's sums follow the core's rounding
// mode too (under floor, 30% of the products were 1-2 ulp off the nearest).

/// Lanes of a step: the emulated multiply's width.
constexpr unsigned L = 32;

template <unsigned N>
using bvec = aie::vector<bfloat16, N>;
template <unsigned N>
using fvec = aie::vector<float, N>;
template <unsigned N>
using acc_t = aie::accum<accfloat, N>;

/// The bf16 limbs of an f32 vector, h + m + l (|m| < 2^-7 |h|, |l| <= 2^-9 |m|).
template <unsigned N>
struct limbs {
    bvec<N> h, m, l;
};

/// Splits @p x into limbs, each from the exact residual of the ones before (an accumulator
/// subtraction, which is exact where aie2p's vadd.f is not). The leading limb is x's upper half,
/// x truncated: rounded, an x above the largest finite bf16 would give inf.
template <unsigned N>
__attribute__((always_inline)) inline limbs<N> split(const fvec<N> & x) {
    const bvec<N> one = aie::broadcast<bfloat16, N>(1.0f);
    acc_t<N> r;
    r.from_vector(x);
    limbs<N> c;
    c.h = aie::filter_odd(x.template cast_to<bfloat16>());
    r = aie::msc(r, c.h, one);
    c.m = r.template to_vector<bfloat16>();
    r = aie::msc(r, c.m, one);
    c.l = r.template to_vector<bfloat16>();
    return c;
}

/**
 * x * s + b on the first N <= L lanes of the broadcast scale's limbs @p s and bias @p b.
 *
 * Every limb product is exact in the f32 accumulator. The six below 2^-14 of x * s are
 * accumulated with multiply-accumulates; the two near 2^-7 of it, then the leading one, are added
 * in separate additions. Measured over random x, 0.4% of the products round 1 ulp off the nearest
 * this way (aie::mul: 0.3-0.7%), and 8.5% when the last three are multiply-accumulates too; a
 * scale exact in bf16 (0.125, 3) gives the nearest everywhere.
 */
template <unsigned N>
__attribute__((always_inline)) inline fvec<N>
scale_bias(const fvec<N> & x, const limbs<L> & s, const fvec<L> & b) {
    const limbs<N> c = split(x);
    const bvec<N> sh = s.h.template extract<N>(0);
    const bvec<N> sm = s.m.template extract<N>(0);
    const bvec<N> sl = s.l.template extract<N>(0);
    acc_t<N> p = aie::mul(c.l, sl);
    p = aie::mac(p, c.l, sm);
    p = aie::mac(p, c.m, sl);
    p = aie::mac(p, c.h, sl);
    p = aie::mac(p, c.m, sm);
    p = aie::mac(p, c.l, sh);
    p = aie::add(p, aie::add(aie::mul(c.m, sh), aie::mul(c.h, sm)));
    p = aie::add(p, aie::mul(c.h, sh));
    return aie::add(p, b.template extract<N>(0)).template to_vector<float>();
}

/**
 * x * s + b over the first @p n elements (a multiple of 2 L), every pointer aligned for a vector
 * access. Each vector is one long latency-bound chain: two independent ones a trip, and a promised
 * minimum trip count, let the pipeliner overlap them.
 */
__attribute__((always_inline)) inline void scale_bias_aligned(const float * __restrict in,
                                                              float * __restrict out,
                                                              int32_t n,
                                                              const limbs<L> & s,
                                                              const fvec<L> & b) {
    GGML_AIE_LOOP_MIN_TRIPS(2, i, n, 2 * L, {
        const fvec<L> x0 = aie::load_v<L>(in);
        const fvec<L> x1 = aie::load_v<L>(in + L);
        aie::store_v(out, scale_bias(x0, s, b));
        aie::store_v(out + L, scale_bias(x1, s, b));
        in += 2 * L;
        out += 2 * L;
    })
}

} // namespace
#endif

extern "C" {

/**
 * @brief Applies scale and bias to each element: out[i] = in[i] * scale + bias.
 *
 * Vectorized (512-bit registers on aie2; 32 lanes of bf16 limb products on aie2p) with a scalar
 * tail for the remainder. scale/bias are runtime args, not compile-time constants, since
 * op_params can vary per dispatch of the same compiled kernel.
 *
 * @param[in]  in    Input array of N float elements.
 * @param[out] out   Output array of N float elements.
 * @param[in]  N     Number of elements to process.
 * @param[in]  scale Multiplicative scale factor.
 * @param[in]  bias  Additive bias term.
 */
void ggml_op_scale(
    const float * __restrict in, float * __restrict out, int32_t N, float scale, float bias) {
    // scale.py streams the largest tile dividing the element count, which need not be a whole
    // number of vectors; aligned accesses only when the builder says it is.
#if __AIEARCH__ == 20
    transform_vector_n<tile_vector_aligned_v<float>>(
        out, N,
        [scale, bias](auto v) {
            return aie::add(aie::mul(v, scale).template to_vector<float>(), bias);
        },
        [scale, bias](float v) { return v * scale + bias; }, in);
#else
    aie::scoped_mode rnd{aie::rounding_mode::conv_even};
    const limbs<L> s = split(aie::broadcast<float, L>(scale));
    const fvec<L> b = aie::broadcast<float, L>(bias);
    // Whole pairs of vectors in the aligned loop when the pointers allow it; the rest (or all of
    // it) a vector of up to L lanes at a time, then a scalar tail.
    int32_t n = 0;
    constexpr uintptr_t mask = aie::vector_ldst_align_v<float, L> - 1;
    if (tile_vector_aligned_v<float> ||
        ((reinterpret_cast<uintptr_t>(in) | reinterpret_cast<uintptr_t>(out)) & mask) == 0) {
        n = N & ~static_cast<int32_t>(2 * L - 1);
        scale_bias_aligned(in, out, n, s, b);
    }
    transform_vector_walk_n<tile_vector_aligned_v<float>, true, 2 * 512>(
        out + n, N - n, [&s, &b](auto v) { return scale_bias(v, s, b); },
        [scale, bias](float v) { return v * scale + bias; }, in + n);
#endif
}

} // extern "C"
