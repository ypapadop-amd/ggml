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
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "OUTPUT_DTYPE must be a floating-point type");

    event0();

#if __AIEARCH__ == 20 && defined(GGML_IM2COL_IW) // aie2 only so far; im2col.py passes the shape.
    {
        // The generic loops below keep every extent at run time, so each element pays four
        // levels of tiny unpipelined loops and a bounds check. With the shape known, each
        // (channel, window row, window column) is one output column slot, filled along ox with
        // the padding columns split off at compile time: zeros, then a pipelined copy, then
        // zeros. A window row in the padding is all zeros.
        constexpr int32_t c_iw = GGML_IM2COL_IW, c_ih = GGML_IM2COL_IH, c_ic = GGML_IM2COL_IC;
        constexpr int32_t c_kw = GGML_IM2COL_KW, c_kh = GGML_IM2COL_KH, c_ow = GGML_IM2COL_OW;
        constexpr int32_t c_s0 = GGML_IM2COL_S0, c_s1 = GGML_IM2COL_S1, c_p0 = GGML_IM2COL_P0,
                          c_p1 = GGML_IM2COL_P1, c_d0 = GGML_IM2COL_D0, c_d1 = GGML_IM2COL_D1;
        constexpr int32_t cs = c_ic * c_kh * c_kw;
        const auto zero = static_cast<OUTPUT_DTYPE>(0.0f);

        // First and one-past-last ox whose input column ox * s0 + off lies in [0, iw).
        constexpr auto ox_lo = [](int32_t off) {
            const int32_t lo = off >= 0 ? 0 : (-off + c_s0 - 1) / c_s0;
            return lo < c_ow ? lo : c_ow;
        };
        constexpr auto ox_hi = [](int32_t off, int32_t lo) {
            const int32_t hi = (c_iw - 1 - off < 0) ? 0 : (c_iw - 1 - off) / c_s0 + 1;
            return hi < lo ? lo : (hi > c_ow ? c_ow : hi);
        };

        for (int32_t iic = 0; iic < c_ic; ++iic) {
            const INPUT_DTYPE * __restrict src_plane = in + iic * c_ih * c_iw;
            for (int32_t ikh = 0; ikh < c_kh; ++ikh) {
                const int32_t iih = oh * c_s1 + ikh * c_d1 - c_p1;
                const bool y_in = iih >= 0 && iih < c_ih;
                const INPUT_DTYPE * __restrict srow = src_plane + (y_in ? iih * c_iw : 0);
                for (int32_t ikw = 0; ikw < c_kw; ++ikw) {
                    OUTPUT_DTYPE * __restrict dst = out + (iic * c_kh + ikh) * c_kw + ikw;
                    const int32_t off = ikw * c_d0 - c_p0;
                    // Bounds selected on y_in, not a separate zero-only branch: with that branch
                    // Peano no longer unrolled the copy loops, 6 cycles per element against ~3
                    // (MNIST conv1 im2col 2.8 -> 3.9 ms). The cost: for some shapes (a 5x3
                    // window, say) it computes these loops' trip counts with a 64-bit multiply
                    // (__muldi3), once per loop, not per element.
                    const int32_t lo = y_in ? ox_lo(off) : c_ow;
                    const int32_t hi = y_in ? ox_hi(off, lo) : c_ow;
                    for (int32_t ox = 0; ox < lo; ++ox) {
                        dst[ox * cs] = zero;
                    }
                    for (int32_t ox = lo; ox < hi; ++ox) {
                        dst[ox * cs] = static_cast<OUTPUT_DTYPE>(srow[ox * c_s0 + off]);
                    }
                    for (int32_t ox = hi; ox < c_ow; ++ox) {
                        dst[ox * cs] = zero;
                    }
                }
            }
        }
    }
#else
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
#endif

    event1();
}

} // extern "C"
