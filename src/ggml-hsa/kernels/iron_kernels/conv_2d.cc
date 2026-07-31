// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file conv_2d.cc
 * @brief Direct 2D convolution for AIE kernels.
 */

#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

namespace {

/**
 * @brief Direct 2D convolution over one output plane.
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
 *
 * Vectorization strategy: the hot loop walks the output width (ox), which is
 * contiguous in both the input row and the output plane. Each output row is
 * split into a scalar border (where some kernel taps fall in the padding) and a
 * bounds-free interior. Over the interior, when s0 == 1 the input window for a
 * fixed tap is a contiguous run, so V output columns are computed at once with a
 * broadcast-weight fused multiply-add (aie::mac). The channel reduction stays
 * the OUTERMOST loop, accumulating into the output plane, to avoid a Peano
 * miscompile that dropped the iic>=1 contribution when the channel loop sat
 * between the spatial loops.
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
template <typename T_in, typename T_out>
void conv_2d_impl(const T_in * __restrict in,
                  const T_in * __restrict wts,
                  T_out * __restrict out,
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
    static_assert(is_floating_point_v<T_in>, "T_in must be a floating-point type");
    static_assert(is_floating_point_v<T_out>, "T_out must be a floating-point type");

    // 512-bit-register lane count: 16 for f32, 32 for bf16.
    constexpr int32_t V = 512 / (8 * sizeof(T_in));

    event0();

    const int32_t plane_size = ih * iw;
    const int32_t knl_plane = kh * kw;
    const int32_t knl_vol = ic * knl_plane; // KH*KW*IC per output channel

    const int32_t n_out = oh * ow;
    for (int32_t i = 0; i < n_out; ++i) {
        out[i] = static_cast<T_out>(0.0f);
    }

    // Interior output-column range [ox_lo, ox_hi) where every kernel tap lands
    // inside the input for all ikw in [0, kw): the padded border columns are
    // peeled off so the inner loop needs no per-tap bounds check.
    //   lower (ikw=0):    ox*s0 - p0 >= 0            -> ox >= ceil(p0 / s0)
    //   upper (ikw=kw-1): ox*s0 + (kw-1)*d0 - p0 < iw
    // The divisors below are the (positive) strides; the numerators are
    // non-negative here (p0 >= 0, and the upper bound is clamped to 0), so the
    // unsigned divides fold to plain operations instead of a signed __divsi3.
    const int32_t ox_lo =
        (p0 > 0)
            ? static_cast<int32_t>((static_cast<uint32_t>(p0) + static_cast<uint32_t>(s0) - 1u) /
                                   static_cast<uint32_t>(s0))
            : 0;
    const int32_t hi_num = iw - 1 + p0 - (kw - 1) * d0;
    int32_t ox_hi =
        (hi_num < 0)
            ? 0
            : static_cast<int32_t>(static_cast<uint32_t>(hi_num) / static_cast<uint32_t>(s0)) + 1;
    if (ox_hi > ow) {
        ox_hi = ow;
    }
    if (ox_hi < ox_lo) {
        ox_hi = ox_lo;
    }

    // Vector chunks only when the input window is contiguous along ox (s0 == 1).
    const bool vectorize = (s0 == 1);
    const int32_t interior = ox_hi - ox_lo;
    const int32_t ox_vec_end = vectorize ? (ox_lo + (interior / V) * V) : ox_lo;

    // Channel reduction stays the outermost loop, accumulating into the output
    // plane, to avoid a Peano miscompile that dropped the iic>=1 contribution
    // when the channel loop sat between the spatial loops.
    for (int32_t iic = 0; iic < ic; ++iic) {
        const T_in * __restrict src_plane = in + iic * plane_size;
        // Weight base for this (oc_idx, iic) slice.
        const T_in * __restrict wt_base = wts + iic * knl_plane + oc_idx * knl_vol;

        for (int32_t oy = 0; oy < oh; ++oy) {
            T_out * __restrict out_row = out + oy * ow;

            // Left / right border columns: some taps fall in the padding, so
            // each element keeps its bounds check.
            for (int32_t ox = 0; ox < ox_lo; ++ox) {
                float acc = 0.0f;
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= ih) {
                        continue;
                    }
                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox * s0 + ikw * d0 - p0;
                        if (iiw >= 0 && iiw < iw) {
                            acc += static_cast<float>(src_plane[iih * iw + iiw]) *
                                   static_cast<float>(wt_base[ikh * kw + ikw]);
                        }
                    }
                }
                out_row[ox] = static_cast<T_out>(static_cast<float>(out_row[ox]) + acc);
            }
            for (int32_t ox = ox_vec_end; ox < ow; ++ox) {
                float acc = 0.0f;
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= ih) {
                        continue;
                    }
                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox * s0 + ikw * d0 - p0;
                        if (iiw >= 0 && iiw < iw) {
                            acc += static_cast<float>(src_plane[iih * iw + iiw]) *
                                   static_cast<float>(wt_base[ikh * kw + ikw]);
                        }
                    }
                }
                out_row[ox] = static_cast<T_out>(static_cast<float>(out_row[ox]) + acc);
            }

            // Interior vector chunks (s0 == 1 only): every tap is in-bounds, so
            // V output columns are computed at once with a broadcast-weight FMA.
            // Columns past ox_vec_end (the interior remainder and the true right
            // border) are handled by the bounds-checked right-border loop above.
            for (int32_t ox = ox_lo; ox < ox_vec_end; ox += V) {
                aie::accum<accfloat, V> acc;
                acc.from_vector(aie::load_unaligned_v<V>(out_row + ox));
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= ih) {
                        continue;
                    }
                    const T_in * __restrict srow = src_plane + iih * iw;
                    const T_in * __restrict wrow = wt_base + ikh * kw;
                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox + ikw * d0 - p0; // s0 == 1
                        const aie::vector<T_in, V> wvec = aie::broadcast<T_in, V>(wrow[ikw]);
                        const aie::vector<T_in, V> ivec = aie::load_unaligned_v<V>(srow + iiw);
                        acc = aie::mac(acc, wvec, ivec);
                    }
                }
                aie::store_unaligned_v(out_row + ox, acc.template to_vector<T_out>());
            }
        }
    }

    event1();
}

} // namespace

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
    conv_2d_impl<INPUT_DTYPE, OUTPUT_DTYPE>(in, wts, out, oc_idx, iw, ih, ic, kw, kh, ow, oh, s0,
                                            s1, p0, p1, d0, d1);
}

} // extern "C"
