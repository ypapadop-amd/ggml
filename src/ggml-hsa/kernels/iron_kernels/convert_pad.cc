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
void ggml_hsa_convert_pad(const f32 * __restrict in,
                          bf16 * __restrict out,
                          int32_t d0,
                          int32_t d0pad) {
    event0();

    for (int32_t i = 0; i < d0; ++i) {
        union {
            f32 f;
            uint32_t u;
        } bits;
        bits.f = in[i];

        uint16_t hi;
        if ((bits.u & 0x7fffffffu) > 0x7f800000u) {
            // NaN: force to quiet
            hi = static_cast<uint16_t>((bits.u >> 16) | 64u);
        } else {
            // round-to-nearest-even bias
            hi = static_cast<uint16_t>((bits.u + (0x7fffu + ((bits.u >> 16) & 1u))) >> 16);
        }

        __builtin_memcpy(&out[i], &hi, sizeof(bf16));
    }

    const bf16 zero = {};
    for (int32_t i = d0; i < d0pad; ++i) {
        out[i] = zero;
    }

    event1();
}

} // extern "C"
