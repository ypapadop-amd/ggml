// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file conv_2d.cc
 * @brief Direct 2D convolution for AIE kernels.
 *
 * Computes one output plane [OW, OH] of a 2D convolution for a single batch
 * element and a fixed output channel (oc_idx). Streaming planes in
 * (batch, oc_idx) order reproduces GGML's contiguous dst layout
 * [OW, OH, OC, N], where oc has stride OW*OH.
 *
 * Weight layout matches GGML's [KW, KH, IC, OC] column-major storage:
 *   wts[kx + ky*KW + ic*KW*KH + oc*KW*KH*IC]
 *
 * Input layout: IC contiguous planes of IH * IW elements (row-major within
 * each plane), matching the image buffer streamed from src1.
 *   in[ic*IH*IW + iy*IW + ix]
 *
 * Output layout: one plane, row-major:
 *   out[oy*OW + ox]
 */

#include <type_traits>

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Compute one output plane of a 2D convolution for one batch element.
 *
 * @param[in]  in      Input image: IC planes of IH * IW elements.
 * @param[in]  wts     Weight tensor: KW*KH*IC*OC elements, layout [KW,KH,IC,OC].
 * @param[out] out     Output plane: OW * OH elements, layout [OW, OH] (row-major).
 * @param[in]  oc_idx  Output channel index.
 * @param[in]  iw      Input width.
 * @param[in]  ih      Input height.
 * @param[in]  ic      Input channels.
 * @param[in]  kw      Kernel width.
 * @param[in]  kh      Kernel height.
 * @param[in]  ow      Output width.
 * @param[in]  oh      Output height.
 * @param[in]  s0      Stride along width.
 * @param[in]  s1      Stride along height.
 * @param[in]  p0      Padding along width.
 * @param[in]  p1      Padding along height.
 * @param[in]  d0      Dilation along width.
 * @param[in]  d1      Dilation along height.
 */
void ggml_op_conv_2d(const INPUT_DTYPE * __restrict in,
                     const INPUT_DTYPE * __restrict wts,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t oc_idx,
                     int32_t iw,
                     int32_t ih,
                     int32_t ic,
                     int32_t kw,
                     int32_t kh,
                     int32_t ow,
                     int32_t oh,
                     int32_t s0,
                     int32_t s1,
                     int32_t p0,
                     int32_t p1,
                     int32_t d0,
                     int32_t d1) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "INPUT_DTYPE must be a floating-point type");

    event0();

    const int32_t plane_size = ih * iw;
    const int32_t knl_plane = kh * kw;
    const int32_t knl_vol = ic * knl_plane; // KH*KW*IC per output channel

    // Channel reduction as the OUTERMOST loop, accumulating into the output
    // plane. Keeping the per-channel spatial convolution as the inner nest (and
    // the channel loop outside it) avoids a Peano miscompile that dropped the
    // iic>=1 contribution when the channel loop sat between the spatial loops.
    const int32_t n_out = oh * ow;
    for (int32_t i = 0; i < n_out; ++i) {
        out[i] = static_cast<OUTPUT_DTYPE>(0.0f);
    }

    for (int32_t iic = 0; iic < ic; ++iic) {
        const INPUT_DTYPE * __restrict src_plane = in + iic * plane_size;
        // Weight base for this (oc_idx, iic) slice: wts[kx + ky*KW + iic*KW*KH + oc_idx*KW*KH*IC]
        const INPUT_DTYPE * __restrict wt_base = wts + iic * knl_plane + oc_idx * knl_vol;

        for (int32_t oy = 0; oy < oh; ++oy) {
            for (int32_t ox = 0; ox < ow; ++ox) {
                float acc = 0.0f;

                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    const bool y_in = (iih >= 0) && (iih < ih);

                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox * s0 + ikw * d0 - p0;
                        const float wt_val = static_cast<float>(wt_base[ikh * kw + ikw]);

                        if (y_in && (iiw >= 0) && (iiw < iw)) {
                            acc += static_cast<float>(src_plane[iih * iw + iiw]) * wt_val;
                        }
                    }
                }

                out[oy * ow + ox] =
                    static_cast<OUTPUT_DTYPE>(static_cast<float>(out[oy * ow + ox]) + acc);
            }
        }
    }

    event1();
}

} // extern "C"
