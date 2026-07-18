// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file depad.cc
 * @brief Row de-padding for the MUL_MAT post-amble, with an optional f32 -> bf16 conversion.
 *
 * Processes one logical row at a time: copies the first @p d0 elements out of a @p d0pad-wide
 * padded input row into a dense @p d0-wide output row. Two modes, selected at compile time:
 *   - default (f32 -> f32): plain copy of the first d0 elements;
 *   - DEPAD_CONVERT_F32_TO_BF16 (f32 -> bf16): converts each element to bf16 bit-identically to the
 *     host reference @c ggml_compute_fp32_to_bf16 (round-to-nearest-even), fusing the per-layer cast
 *     that would otherwise run as a separate CPY after the MUL_MAT.
 * Streaming both sides one row at a time keeps the shim DMA transfers linear (no strided 2D
 * descriptor), avoiding the hardware BD wrap-size limits that a single large strided gather would
 * hit. The [d0, d0pad) padding is never read (only the first d0 elements are copied).
 */

#include <aie_api/aie.hpp>
#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

#ifdef DEPAD_CONVERT_F32_TO_BF16
/**
 * @brief Converts one f32 element to bf16 (RNE, NaN->quiet), bit-identical to the host.
 *
 * Mirrors @c ggml_compute_fp32_to_bf16 exactly; used for the scalar tail so vectorized and scalar
 * paths produce identical bits.
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
 * @brief Copies the first @p d0 elements of a @p d0pad-wide padded f32 row to a dense output row.
 *
 * In the default mode the output is f32 and elements are copied unchanged. In
 * DEPAD_CONVERT_F32_TO_BF16 mode the output is bf16 and each element is converted (RNE, matching the
 * host). The [d0, d0pad) tail is never read.
 *
 * @param[in]  in     Input padded row of @p d0pad f32 elements.
 * @param[out] out    Output dense row of @p d0 elements (OUTPUT_DTYPE).
 * @param[in]  d0     Number of valid elements to copy.
 * @param[in]  d0pad  Padded input row width (>= d0); only [0, d0) is read.
 */
void ggml_hsa_depad(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t d0,
                    int32_t d0pad) {
    event0();
    (void)d0pad;

    constexpr int32_t V = 512 / (sizeof(f32) * 8);

    // Row width is fixed per JIT-compiled kernel instance, so depad.py passes it as a -D define.
    // Using the compile-time bound lets Peano fold the vector trip count and software-pipeline the
    // hot loop (fills the VLIW nop slots the runtime-bound version left). The runtime arg still
    // arrives over the ABI; fall back to it if the define is absent.
#ifdef DEPAD_D0
    constexpr int32_t d0v = DEPAD_D0;
    (void)d0;
#else
    const int32_t d0v = d0;
#endif

    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    // Unaligned load/store: rows stream through double-buffered fifos at a per-row
    // stride that is not vector-aligned (same as binary_ops bias). No
    // AIE_LOOP_MIN_ITERATION_COUNT: d0 can be < V (fc2 d0=10) so nblk may be 0.
    AIE_PREPARE_FOR_PIPELINING
    // V == 512 / (32 bits) == 16 f32 lanes. Only bind the range hint when the row spans at least
    // one full vector (fc2 output rows have d0 < V, giving nblk == 0, for which the trip-count
    // pragma would be an invalid min_iteration_count(0)).
#if defined(DEPAD_D0) && (DEPAD_D0) >= 16
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
#ifdef DEPAD_CONVERT_F32_TO_BF16
        // f32 -> bf16, bit-identical to the host reference (see convert_pad.cc for the derivation).
        // Aligned store: output rows are d0-wide dense bf16; the per-row fifo buffer base is
        // vector-aligned (same as convert_pad's bf16 pad-only), which also dodges the broken 16-bit
        // unaligned vector store in this aie_api version. Input still streams unaligned.
        const aie::vector<f32, V> fv = aie::load_unaligned_v<V>(in + i);
        const aie::vector<uint32_t, V> u = aie::vector_cast<uint32_t>(fv);
        const aie::vector<uint32_t, V> hi16 = aie::logical_downshift(u, 16);

        const aie::vector<uint32_t, V> nan_val = aie::bit_or(64u, hi16);

        const aie::vector<uint32_t, V> lsb = aie::bit_and(1u, hi16);
        const aie::vector<uint32_t, V> rounded = aie::add(u, aie::add(lsb, 0x7fffu));
        const aie::vector<uint32_t, V> rne_val = aie::logical_downshift(rounded, 16);

        const auto nan_mask = aie::gt(aie::bit_and(0x7fffffffu, u), 0x7f800000u);
        const aie::vector<uint32_t, V> res32 = aie::select(rne_val, nan_val, nan_mask);

        const aie::vector<uint16_t, V> res16 = aie::filter_even(aie::vector_cast<uint16_t>(res32));
        aie::store_v(out + i, aie::vector_cast<bf16>(res16));
#else
        const aie::vector<f32, V> v = aie::load_unaligned_v<V>(in + i);
        aie::store_unaligned_v(out + i, v);
#endif
    }

    for (int32_t i = vend; i < d0v; ++i) {
#ifdef DEPAD_CONVERT_F32_TO_BF16
        const uint16_t hi = convert_scalar(in[i]);
        __builtin_memcpy(&out[i], &hi, sizeof(bf16));
#else
        out[i] = in[i];
#endif
    }

    event1();
}

} // extern "C"
