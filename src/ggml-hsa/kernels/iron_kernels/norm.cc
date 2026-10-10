// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file norm.cc
 * @brief Layer normalization (GGML_OP_NORM) over dim 0 for AIE kernels.
 */

#include <stdint.h>

#include <bit>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

namespace {

constexpr int32_t V = 512 / (sizeof(float) * 8);
using vec = aie::vector<float, V>;

/// Rows whose statistics are finished together, one per lane.
constexpr int32_t G = V;

/// Vectors per trip of the unrolled loops: an f32 multiply is a long serial chain, and a single
/// one per trip leaves the loop latency-bound.
constexpr int32_t U = 4;

template <bool Aligned>
__attribute__((always_inline)) inline vec load(const float * p) {
    if constexpr (Aligned) {
        return aie::load_v<V>(p);
    } else {
        return aie::load_unaligned_v<V>(p);
    }
}

template <bool Aligned>
__attribute__((always_inline)) inline void store(float * p, const vec & v) {
    if constexpr (Aligned) {
        aie::store_v(p, v);
    } else {
        aie::store_unaligned_v(p, v);
    }
}

#if __AIEARCH__ == 20

// f32 products on bf16 limbs. aie2 has no f32 multiplier: an f32 aie::mul splits both operands into
// three bf16 limbs (hi, mid, lo; truncating conversions, each limb from the exact residual of the
// ones before) and sums the nine limb products, on every call. Here a row's scale is split once
// per row, and the bf16 multiply's two products per lane (lanes i and i + V of its operands) pair
// the limb products, so a deviation's square takes two multiplies and its product with the
// scale four. The caller sets floor rounding (truncating conversions).

using pair_t = aie::vector<bfloat16, 2 * V>;
using acc_t = aie::accum<accfloat, V>;

struct limbs {
    aie::vector<bfloat16, V> h, m, l;
};

__attribute__((always_inline)) inline limbs split(const vec & x) {
    const aie::vector<bfloat16, V> one = aie::broadcast<bfloat16, V>(1.0f);
    acc_t r;
    r.from_vector(x);
    limbs s;
    s.h = r.template to_vector<bfloat16>();
    r = aie::msc(r, s.h, one);
    s.m = r.template to_vector<bfloat16>();
    r = aie::msc(r, s.m, one);
    s.l = r.template to_vector<bfloat16>();
    return s;
}

/// A running sum of squares, a + 2 * b, of the limbs h, m, l of each square: h h + m m + 2 (h m +
/// h l). The dropped 2 m l + l l is below 2^-21 of the square (|m| < 2^-7 |h|, |l| < 2^-15 |h|).
struct sum_sq {
    acc_t a = aie::zeros<accfloat, V>();
    acc_t b = aie::zeros<accfloat, V>();

    __attribute__((always_inline)) void add(const vec & c) {
        const limbs s = split(c);
        const pair_t hm = aie::concat(s.h, s.m);
        a = acc_t(::mac_elem_16_2(hm, hm, a));                                       // h h + m m
        b = acc_t(::mac_elem_16_2(aie::concat(s.h, s.h), aie::concat(s.m, s.l), b)); // h m + h l
    }

    __attribute__((always_inline)) void add(const sum_sq & o) {
        a = aie::add(a, o.a);
        b = aie::add(b, o.b);
    }

    __attribute__((always_inline)) vec total() const {
        return aie::add(a.template to_vector<float>(),
                        aie::add(b.template to_vector<float>(), b.template to_vector<float>()));
    }
};

/// A row's scale s, split once, as the pair operands of c * s: every limb product but lo * lo.
struct factor {
    pair_t hm, mh, ll;

    __attribute__((always_inline)) explicit factor(float f) {
        const limbs s = split(aie::broadcast<float, V>(f));
        hm = aie::concat(s.h, s.m);
        mh = aie::concat(s.m, s.h);
        ll = aie::concat(s.l, s.l);
    }

    __attribute__((always_inline)) vec times(const vec & x) const {
        const limbs c = split(x);
        const pair_t c_hm = aie::concat(c.h, c.m);
        acc_t p(::mul_elem_16_2(aie::concat(c.l, c.l), hm)); // l s_h + l s_m
        p = acc_t(::mac_elem_16_2(c_hm, ll, p));             // h s_l + m s_l
        p = acc_t(::mac_elem_16_2(c_hm, mh, p));             // h s_m + m s_h
        p = acc_t(::mac_elem_16_2(c_hm, hm, p));             // h s_h + m s_m
        return p.template to_vector<float>();
    }
};

#else

/// A running sum of squares (the f32 multiply as the target provides it).
struct sum_sq {
    vec a = aie::zeros<float, V>();

    __attribute__((always_inline)) void add(const vec & c) {
        a = aie::add(a, aie::mul(c, c).template to_vector<float>());
    }

    __attribute__((always_inline)) void add(const sum_sq & o) { a = aie::add(a, o.a); }

    __attribute__((always_inline)) vec total() const { return a; }
};

/// A row's scale s, as a multiplier.
struct factor {
    vec s;

    __attribute__((always_inline)) explicit factor(float f) : s(aie::broadcast<float, V>(f)) {}

    __attribute__((always_inline)) vec times(const vec & x) const {
        return aie::mul(x, s).template to_vector<float>();
    }
};

#endif

// Rows end anywhere in a tile, so a vector access at a row's end may reach past the row. A load
// may read into the rows that follow; it must not read past the tile. A store may write over
// them too, since they are written after it, but aie::store_unaligned_v is a read-modify-write of
// every 32-byte block its two 256-bit halves touch, up to 32 bytes past the vector: nothing it
// rewrites may lie past the tile, where the DMA may be filling or draining the neighbouring
// buffer. `room` is the number of floats from a row's tail to the end of the tile.

/// Floats past an unaligned vector store's end that its read-modify-write may rewrite.
constexpr int32_t STORE_SPILL = 32 / sizeof(float);

/**
 * The last N % V elements of a row, as a vector whose lanes past the row hold whatever follows
 * (garbage to the caller): read in place when the vector fits in the tile, else through a bounce
 * buffer.
 */
__attribute__((always_inline)) inline vec load_tail(const float * p, int32_t tail, int32_t room) {
    return room >= V ? aie::load_unaligned_v<V>(p) : load_v_partial<V>(p, tail);
}

/// Sum of a row's vend + tail elements (vend a multiple of V, tail < V).
template <bool Aligned>
__attribute__((always_inline)) inline float row_sum(
    const float * __restrict p, int32_t vend, int32_t tail, aie::mask<V> tail_lanes, int32_t room) {
    vec acc = aie::zeros<float, V>();
    GGML_AIE_LOOP_MIN_TRIPS(4, i, vend, V, {
        acc = aie::add(acc, load<Aligned>(p));
        p += V;
    })
    if (tail > 0) {
        acc = aie::add(acc,
                       aie::select(aie::zeros<float, V>(), load_tail(p, tail, room), tail_lanes));
    }
    return aie::reduce_add(acc);
}

/// Sum of a row's squared deviations from @p m.
template <bool Aligned>
__attribute__((always_inline)) inline float row_sum_sq_dev(const float * __restrict p,
                                                           const vec & m,
                                                           int32_t vend,
                                                           int32_t tail,
                                                           aie::mask<V> tail_lanes,
                                                           int32_t room) {
    sum_sq acc0, acc1;
    const int32_t vendu = vend & ~(U * V - 1);
    for (int32_t i = 0; i < vendu; i += U * V) {
        acc0.add(aie::sub(load<Aligned>(p), m));
        acc1.add(aie::sub(load<Aligned>(p + V), m));
        acc0.add(aie::sub(load<Aligned>(p + 2 * V), m));
        acc1.add(aie::sub(load<Aligned>(p + 3 * V), m));
        p += U * V;
    }
    acc0.add(acc1);
    for (int32_t i = vendu; i < vend; i += V) {
        acc0.add(aie::sub(load<Aligned>(p), m));
        p += V;
    }
    if (tail > 0) {
        acc0.add(
            aie::select(aie::zeros<float, V>(), aie::sub(load_tail(p, tail, room), m), tail_lanes));
    }
    return aie::reduce_add(acc0.total());
}

/**
 * Writes a row's (x - m) * s. Unaligned stores need STORE_SPILL floats of the tile after them:
 * a tail with less room goes through a bounce buffer, and so does the last whole vector of an
 * unaligned row when the tail after it is shorter than that (only on a tile's last row).
 */
template <bool Aligned>
__attribute__((always_inline)) inline void row_normalize(const float * __restrict p,
                                                         float * __restrict q,
                                                         const vec & m,
                                                         const factor & s,
                                                         int32_t vend,
                                                         int32_t tail,
                                                         int32_t room) {
    const bool bounce_last = !Aligned && vend > 0 && room < STORE_SPILL;
    const int32_t vend_direct = bounce_last ? vend - V : vend;
    const int32_t vendu = vend_direct & ~(U * V - 1);
    for (int32_t i = 0; i < vendu; i += U * V) {
        AIE_LOOP_UNROLL_FULL
        for (int32_t u = 0; u < U; ++u) {
            store<Aligned>(q + u * V, s.times(aie::sub(load<Aligned>(p + u * V), m)));
        }
        p += U * V;
        q += U * V;
    }
    for (int32_t i = vendu; i < vend_direct; i += V) {
        store<Aligned>(q, s.times(aie::sub(load<Aligned>(p), m)));
        p += V;
        q += V;
    }
    if (bounce_last) {
        store_v_partial<V>(q, s.times(aie::sub(load<Aligned>(p), m)), V);
        p += V;
        q += V;
    }
    if (tail > 0) {
        const vec y = s.times(aie::sub(load_tail(p, tail, room), m));
        if (room >= V + STORE_SPILL) {
            aie::store_unaligned_v(q, y);
        } else {
            store_v_partial<V>(q, y, tail);
        }
    }
}

/**
 * Normalizes @p rows rows of @p N floats. The rows go in groups of G: one pass sums each row of
 * the group, the group's means are one vector multiply, a second pass sums each row's squared
 * deviations, and the group's reciprocal square roots are one vector computation, so no row takes
 * a scalar float operation (each one a runtime call on the AIE scalar unit). A third pass writes
 * each row. A row's last N % V elements go through a vector too, the lanes past the row kept out
 * of the sums.
 */
template <bool Aligned>
void norm_rows(const float * __restrict in,
               float * __restrict out,
               int32_t N,
               int32_t rows,
               float eps,
               float inv_n) {
    const int32_t vend = (N / V) * V;
    const int32_t tail = N - vend;
    const aie::mask<V> tail_lanes = aie::mask<V>::from_uint32((1u << tail) - 1u);
    // Floats from row r's tail to the end of the tile.
    const auto room = [&](int32_t r)
                          __attribute__((always_inline)) { return tail + (rows - 1 - r) * N; };

    for (int32_t r0 = 0; r0 < rows; r0 += G) {
        const int32_t g = rows - r0 < G ? rows - r0 : G;

        alignas(aie::vector_decl_align) float mean[G] = {};
        alignas(aie::vector_decl_align) float scale[G] = {};

        for (int32_t r = 0; r < g; ++r) {
            mean[r] = row_sum<Aligned>(in + (r0 + r) * N, vend, tail, tail_lanes, room(r0 + r));
        }
        aie::store_v(mean, aie::mul(aie::load_v<V>(mean), inv_n).template to_vector<float>());

        for (int32_t r = 0; r < g; ++r) {
            scale[r] = row_sum_sq_dev<Aligned>(in + (r0 + r) * N, aie::broadcast<float, V>(mean[r]),
                                               vend, tail, tail_lanes, room(r0 + r));
        }
        // 1 / sqrt(variance + eps); lanes past the group's rows compute 1 / sqrt(eps), unused.
        aie::store_v(scale, invsqrt_newton(aie::add(
                                aie::mul(aie::load_v<V>(scale), inv_n).template to_vector<float>(),
                                aie::broadcast<float, V>(eps))));

        for (int32_t r = 0; r < g; ++r) {
            row_normalize<Aligned>(in + (r0 + r) * N, out + (r0 + r) * N,
                                   aie::broadcast<float, V>(mean[r]), factor(scale[r]), vend, tail,
                                   room(r0 + r));
        }
    }
}

} // namespace

extern "C" {

/**
 * @brief Normalizes rows over dim 0.
 *
 * Computes y = (x - mean) / sqrt(variance + eps) per row, where mean and variance are the
 * population statistics over the N row elements, as ggml_compute_forward_norm_f32 does (the
 * variance divides by N, not N-1; here as a multiply by 1/N).
 *
 * @param[in]  in         Tile of @p rows contiguous rows of N floats.
 * @param[out] out        Output tile, same layout (must not alias @p in).
 * @param[in]  N          Row length (nc = ne00).
 * @param[in]  rows       Rows in the tile.
 * @param[in]  eps_bits   Raw IEEE-754 bits of eps; float arguments are passed as int32 to avoid
 *                        the peano-compat IR pass mangling hex float immediates.
 * @param[in]  inv_n_bits Raw IEEE-754 bits of 1/N (rounded to float).
 */
void ggml_op_norm(const float * __restrict in,
                  float * __restrict out,
                  int32_t N,
                  int32_t rows,
                  int32_t eps_bits,
                  int32_t inv_n_bits) {
    event0();

    const float eps = std::bit_cast<float>(eps_bits);
    const float inv_n = std::bit_cast<float>(inv_n_bits);

#if __AIEARCH__ == 20
    const aie::rounding_mode rnd = aie::get_rounding();
    aie::set_rounding(aie::rounding_mode::floor);
#endif

    // Every row starts aligned for a vector access when the tile does and a row is a whole
    // number of alignment units: then all accesses but the tails are aligned (an unaligned store
    // is a read-modify-write).
    constexpr uintptr_t mask = aie::vector_ldst_align_v<float, V> - 1;
    if (((reinterpret_cast<uintptr_t>(in) | reinterpret_cast<uintptr_t>(out) |
          static_cast<uintptr_t>(N) * sizeof(float)) &
         mask) == 0) {
        norm_rows<true>(in, out, N, rows, eps, inv_n);
    } else {
        norm_rows<false>(in, out, N, rows, eps, inv_n);
    }

#if __AIEARCH__ == 20
    aie::set_rounding(rnd);
#endif

    event1();
}

} // extern "C"
