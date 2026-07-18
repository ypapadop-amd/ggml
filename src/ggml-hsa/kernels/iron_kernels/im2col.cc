// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <type_traits>

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Gathers one output row of a 2D im2col transform for a single image.
 *
 * Mirrors ggml_compute_forward_im2col_f32 for one batch element and output row
 * (fixed oh). Output columns pack IC*KH*KW taps channel-major; taps outside
 * the padded input are written as zero rather than skipped, so every column
 * is fully populated. INPUT_DTYPE/OUTPUT_DTYPE may differ: each element is
 * cast on the way out.
 *
 * @param[in]  in   Input image: IC planes of IH * IW elements (row-major).
 * @param[out] out  Output row: OW * (IC * KH * KW) elements.
 * @param[in]  oh   Output row index along height.
 * @param[in]  iw   Input width (IW).
 * @param[in]  ih   Input height (IH).
 * @param[in]  ic   Input channels (IC).
 * @param[in]  kw   Kernel width (KW).
 * @param[in]  kh   Kernel height (KH).
 * @param[in]  ow   Output width (OW).
 * @param[in]  s0   Stride along width.
 * @param[in]  s1   Stride along height.
 * @param[in]  p0   Padding along width.
 * @param[in]  p1   Padding along height.
 * @param[in]  d0   Dilation along width.
 * @param[in]  d1   Dilation along height.
 */
void ggml_op_im2col(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t oh,
                    int32_t iw,
                    int32_t ih,
                    int32_t ic,
                    int32_t kw,
                    int32_t kh,
                    int32_t ow,
                    int32_t s0,
                    int32_t s1,
                    int32_t p0,
                    int32_t p1,
                    int32_t d0,
                    int32_t d1) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");

    event0();

    const int32_t col_stride = ic * kh * kw;
    const int32_t plane_size = ih * iw;

    for (int32_t ox = 0; ox < ow; ++ox) {
        OUTPUT_DTYPE * __restrict dst_col = out + ox * col_stride;
        for (int32_t iic = 0; iic < ic; ++iic) {
            const INPUT_DTYPE * __restrict src_plane = in + iic * plane_size;
            for (int32_t ikh = 0; ikh < kh; ++ikh) {
                const int32_t iih = oh * s1 + ikh * d1 - p1;
                const bool y_in = (iih >= 0) && (iih < ih);
                for (int32_t ikw = 0; ikw < kw; ++ikw) {
                    const int32_t iiw = ox * s0 + ikw * d0 - p0;
                    const int32_t idx = iic * (kh * kw) + ikh * kw + ikw;
                    if (y_in && (iiw >= 0) && (iiw < iw)) {
                        dst_col[idx] = static_cast<OUTPUT_DTYPE>(src_plane[iih * iw + iiw]);
                    } else {
                        dst_col[idx] = static_cast<OUTPUT_DTYPE>(0.0f);
                    }
                }
            }
        }
    }

    event1();
}

} // extern "C"
