// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file convert_pad.cc
 * @brief Row zero-padding for the MUL_MAT pre-amble, with an optional f32 -> bf16 conversion.
 *
 * Processes one logical row at a time: writes @p d0 valid elements into a @p d0pad-wide output row
 * and zero-fills the [d0, d0pad) tail. Two modes, selected at compile time:
 *   - default (f32 -> bf16): converts each element bit-identically to the host reference
 *     @c ggml_compute_fp32_to_bf16 (round-to-nearest-even);
 *   - CONVERT_PAD_PAD_ONLY (bf16 -> bf16): copies the elements unchanged (the operand is already
 *     bf16, e.g. produced by an in-graph cast), so only the tile padding is added.
 * Streaming both sides one row at a time keeps the shim DMA transfers linear (no strided 2D
 * descriptor), which avoids the hardware wrap-size limits a single large strided scatter would hit.
 * The trailing rows [d1, d1pad) of the destination are left untouched (buffer pre-zeroed).
 */

#include <aie_api/aie.hpp>
#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

#ifndef CONVERT_PAD_PAD_ONLY
/**
 * @brief Converts one f32 element to bf16 (RNE, NaN->quiet), bit-identical to the host.
 *
 * Mirrors @c ggml_compute_fp32_to_bf16 exactly; used for the scalar tail so vectorized and
 * scalar paths produce identical bits.
 */
static inline uint16_t convert_scalar(f32 v) {
    union {
        f32 f;
        uint32_t u;
    } bits;
    bits.f = v;
    if ((bits.u & 0x7fffffffu) > 0x7f800000u) {
        return static_cast<uint16_t>((bits.u >> 16) | 64u);
    }
    return static_cast<uint16_t>((bits.u + (0x7fffu + ((bits.u >> 16) & 1u))) >> 16);
}
#endif

/**
 * @brief Writes one row of @p d0 valid elements into a @p d0pad-wide output row (padding the tail).
 *
 * In the default mode @p in is f32 and each element is converted to bf16 (round-to-nearest-even,
 * matching the host). In CONVERT_PAD_PAD_ONLY mode both sides are bf16 and elements are copied
 * unchanged. The [d0, d0pad) tail is zeroed so the padded GEMM operand reads zero there.
 *
 * @param[in]  in     Input row of @p d0 elements (INPUT_DTYPE).
 * @param[out] out    Output row of @p d0pad elements (OUTPUT_DTYPE).
 * @param[in]  d0     Number of valid elements.
 * @param[in]  d0pad  Padded row width (>= d0).
 */
void ggml_hsa_convert_pad(const INPUT_DTYPE * __restrict in,
                          OUTPUT_DTYPE * __restrict out,
                          int32_t d0,
                          int32_t d0pad) {
    event0();

    // Row shape is fixed per JIT-compiled kernel instance, so convert_pad.py passes it as -D
    // defines. Using the compile-time bounds lets Peano fold the vector trip count and
    // software-pipeline the hot loop (fills the VLIW nop slots the runtime-bound version left).
    // The runtime args still arrive over the ABI; fall back to them if the defines are absent.
#ifdef CONVERT_PAD_D0
    constexpr int32_t d0v = CONVERT_PAD_D0;
    constexpr int32_t d0padv = CONVERT_PAD_D0PAD;
    (void)d0;
    (void)d0pad;
#else
    const int32_t d0v = d0;
    const int32_t d0padv = d0pad;
#endif

#ifdef CONVERT_PAD_PAD_ONLY
    // bf16 -> bf16: pad only, no conversion. V = 512 / 16 = 32 bf16 lanes. Unaligned load (the
    // per-row fifo stride is not vector-aligned) + aligned store (output rows are d0pad-wide, a
    // tile multiple, so vector-aligned; also dodges the broken 16-bit unaligned vector store).
    constexpr int32_t V = 512 / (sizeof(OUTPUT_DTYPE) * 8);
    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    AIE_PREPARE_FOR_PIPELINING
#if defined(CONVERT_PAD_D0) && (CONVERT_PAD_D0) >= 32
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
        const aie::vector<OUTPUT_DTYPE, V> v = aie::load_unaligned_v<V>(in + i);
        aie::store_v(out + i, v);
    }

    for (int32_t i = vend; i < d0v; ++i) {
        out[i] = in[i];
    }
#else
    constexpr int32_t V = 512 / (sizeof(f32) * 8);
    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    // Vectorized f32 -> bf16, replicating ggml_compute_fp32_to_bf16's integer arithmetic
    // lane-wise so the result is bit-identical to the scalar host reference (rather than
    // relying on hardware rounding/NaN handling). Unaligned load/store: rows stream through
    // double-buffered fifos at a non-vector-aligned per-row stride (same as binary_ops bias).
    // No AIE_LOOP_MIN_ITERATION_COUNT: d0 can be < V, giving nblk == 0.
    AIE_PREPARE_FOR_PIPELINING
#ifdef CONVERT_PAD_D0
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
        const aie::vector<f32, V> fv = aie::load_unaligned_v<V>(in + i);
        const aie::vector<uint32_t, V> u = aie::vector_cast<uint32_t>(fv);
        const aie::vector<uint32_t, V> hi16 = aie::logical_downshift(u, 16);

        // NaN: (u >> 16) | 64
        const aie::vector<uint32_t, V> nan_val = aie::bit_or(64u, hi16);

        // RNE: (u + (0x7fff + ((u >> 16) & 1))) >> 16
        const aie::vector<uint32_t, V> lsb = aie::bit_and(1u, hi16);
        const aie::vector<uint32_t, V> rounded = aie::add(u, aie::add(lsb, 0x7fffu));
        const aie::vector<uint32_t, V> rne_val = aie::logical_downshift(rounded, 16);

        // nan_mask ? nan_val : rne_val   (select(v1, v2, m) == m ? v2 : v1)
        const auto nan_mask = aie::gt(aie::bit_and(0x7fffffffu, u), 0x7f800000u);
        const aie::vector<uint32_t, V> res32 = aie::select(rne_val, nan_val, nan_mask);

        // The bf16 bits sit in the low 16 of each u32 lane; grab the even (low) uint16 halves.
        // Output rows are d0pad-wide (tile-multiple, vector-aligned), so an aligned store is safe.
        const aie::vector<uint16_t, V> res16 = aie::filter_even(aie::vector_cast<uint16_t>(res32));
        aie::store_v(out + i, aie::vector_cast<bf16>(res16));
    }

    for (int32_t i = vend; i < d0v; ++i) {
        const uint16_t hi = convert_scalar(in[i]);
        __builtin_memcpy(&out[i], &hi, sizeof(bf16));
    }
#endif

    // Zero-fill the [d0, d0pad) tail. Small and off the hot path; kept scalar because the 16-bit
    // unaligned vector store is broken in this aie_api version and d0 is not vector-aligned.
    const OUTPUT_DTYPE zero = {};
    for (int32_t i = d0v; i < d0padv; ++i) {
        out[i] = zero;
    }

    event1();
}

} // extern "C"
