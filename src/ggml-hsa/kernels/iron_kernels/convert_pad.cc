// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file convert_pad.cc
 * @brief f32 -> bf16 conversion with row zero-padding for the MUL_MAT pre-amble.
 *
 * Processes one logical row at a time: converts @p d0 f32 elements to bf16 (bit-identical to the
 * host reference @c ggml_compute_fp32_to_bf16, round-to-nearest-even) into a @p d0pad-wide output
 * row, zero-filling the [d0, d0pad) tail. Streaming both sides one row at a time keeps the shim DMA
 * transfers linear (no strided 2D descriptor), which avoids the hardware wrap-size limits that a
 * single large strided scatter would hit. The trailing rows [d1, d1pad) of the destination are left
 * untouched (the buffer is pre-zeroed before dispatch).
 */

#include <aie_api/aie.hpp>
#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Converts one row of @p d0 f32 elements to bf16 into a @p d0pad-wide row.
 *
 * The valid [0, d0) elements are converted (round-to-nearest-even, matching the host); the
 * [d0, d0pad) tail is zeroed so the padded GEMM operand reads zero there.
 *
 * @param[in]  in     Input row of @p d0 f32 elements.
 * @param[out] out    Output row of @p d0pad bf16 elements.
 * @param[in]  d0     Number of valid elements to convert.
 * @param[in]  d0pad  Padded row width (>= d0).
 */
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

void ggml_hsa_convert_pad(const f32 * __restrict in,
                          bf16 * __restrict out,
                          int32_t d0,
                          int32_t d0pad) {
    event0();

    constexpr int32_t V = 512 / (sizeof(f32) * 8);
    const int32_t vend = (d0 / V) * V;

    // Vectorized f32 -> bf16, replicating ggml_compute_fp32_to_bf16's integer arithmetic
    // lane-wise so the result is bit-identical to the scalar host reference (rather than
    // relying on hardware rounding/NaN handling). Unaligned load/store: rows stream through
    // double-buffered fifos at a non-vector-aligned per-row stride (same as binary_ops bias).
    // No AIE_LOOP_MIN_ITERATION_COUNT: d0 can be < V, giving vend == 0.
    AIE_PREPARE_FOR_PIPELINING
    for (int32_t i = 0; i < vend; i += V) {
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
        const aie::vector<uint16_t, V> res16 =
            aie::filter_even(aie::vector_cast<uint16_t>(res32));
        aie::store_v(out + i, aie::vector_cast<bf16>(res16));
    }

    for (int32_t i = vend; i < d0; ++i) {
        const uint16_t hi = convert_scalar(in[i]);
        __builtin_memcpy(&out[i], &hi, sizeof(bf16));
    }

    // Zero-fill the [d0, d0pad) tail. Small and off the hot path; kept scalar because the 16-bit
    // unaligned vector store is broken in this aie_api version and d0 is not vector-aligned.
    const bf16 zero = {};
    for (int32_t i = d0; i < d0pad; ++i) {
        out[i] = zero;
    }

    event1();
}

} // extern "C"
