// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file depad.cc
 * @brief Row de-padding for the MUL_MAT post-amble (f32, no dtype change).
 *
 * Processes one logical row at a time: copies the first @p d0 f32 elements out of a @p d0pad-wide
 * padded input row into a dense @p d0-wide output row. Streaming both sides one row at a time keeps
 * the shim DMA transfers linear (no strided 2D descriptor), avoiding the hardware BD wrap-size
 * limits that a single large strided gather would hit.
 */

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Copies the first @p d0 f32 elements of a @p d0pad-wide padded row to a dense row.
 *
 * @param[in]  in     Input padded row of @p d0pad f32 elements.
 * @param[out] out    Output dense row of @p d0 f32 elements.
 * @param[in]  d0     Number of valid elements to copy.
 * @param[in]  d0pad  Padded input row width (>= d0); only [0, d0) is read.
 */
void ggml_hsa_depad(const f32 * __restrict in, f32 * __restrict out, int32_t d0, int32_t d0pad) {
    event0();
    (void)d0pad;

    for (int32_t i = 0; i < d0; ++i) {
        out[i] = in[i];
    }

    event1();
}

} // extern "C"
