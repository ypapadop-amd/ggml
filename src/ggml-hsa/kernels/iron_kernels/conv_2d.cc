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

// Fully unrolling the kernel-window loops is the single biggest win on a 3x3
// convolution, but the tap body is replicated across four loops (left border,
// the two interior vector widths, and the tail), so the unrolled code grows as
// 4 * KW * KH and the AIE core only has 16 KB of program memory. At 3x3 it
// fits; at 5x5 and above it does not, and the link fails with ".text will not
// fit in region 'program'" (by 35520 bytes at 11x11). That failure is caught by
// ggml_backend_hsa_device_supports_op and quietly routes the op to the CPU, so
// unrolling unconditionally does not corrupt anything -- it just silently drops
// NPU support for every window larger than 3x3, which previously worked. Gate
// the unroll on the tap count so those shapes keep their (unrolled-free) kernel.
#if (GGML_CONV2D_KW * GGML_CONV2D_KH) <= 9
#    define GGML_CONV2D_UNROLL_TAPS AIE_LOOP_UNROLL_FULL
#else
#    define GGML_CONV2D_UNROLL_TAPS
#endif

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
 *
 * Every extent and op_param (IW, IH, IC, KW, KH, OW, OH, S0, S1, P0, P1, D0,
 * D1) arrives as a -DGGML_CONV2D_* compile definition rather than an argument.
 * conv_2d.py knows them all at build time and emits one specialization per
 * configuration, which is what lets the tap loops take a constant trip count
 * and the address arithmetic fold to immediates. This is safe because the JIT
 * cache key encodes the tensor shapes and the op_params losslessly (see
 * ggml_hsa_encode_op_params), so a specialization is never handed to a tensor
 * it was not built for.
 */
template <typename T_in, typename T_out>
void conv_2d_impl(const T_in * __restrict in,
                  const T_in * __restrict wts,
                  T_out * __restrict out,
                  int32_t oc_idx) {
    static_assert(is_floating_point_v<T_in>, "T_in must be a floating-point type");
    static_assert(is_floating_point_v<T_out>, "T_out must be a floating-point type");

    constexpr int32_t iw = GGML_CONV2D_IW;
    constexpr int32_t ih = GGML_CONV2D_IH;
    constexpr int32_t ic = GGML_CONV2D_IC;
    constexpr int32_t kw = GGML_CONV2D_KW;
    constexpr int32_t kh = GGML_CONV2D_KH;
    constexpr int32_t ow = GGML_CONV2D_OW;
    constexpr int32_t oh = GGML_CONV2D_OH;
    constexpr int32_t s0 = GGML_CONV2D_S0;
    constexpr int32_t s1 = GGML_CONV2D_S1;
    constexpr int32_t p0 = GGML_CONV2D_P0;
    constexpr int32_t p1 = GGML_CONV2D_P1;
    constexpr int32_t d0 = GGML_CONV2D_D0;
    constexpr int32_t d1 = GGML_CONV2D_D1;

    static_assert(ic > 0, "GGML_CONV2D_IC must be positive");
    static_assert(oh > 0 && ow > 0, "GGML_CONV2D_OH/OW must be positive");

    // With a single input channel each output element is written exactly once,
    // so there is nothing to accumulate across: the plane needs no zeroing and
    // each chunk can start from a zero accumulator instead of reloading what
    // the zero-init just wrote.
    constexpr bool accumulate = (ic > 1);

    // 512-bit-register lane count: 16 for f32, 32 for bf16.
    constexpr int32_t V = 512 / (8 * sizeof(T_in));

    event0();

    const int32_t plane_size = ih * iw;
    const int32_t knl_plane = kh * kw;
    const int32_t knl_vol = ic * knl_plane; // KH*KW*IC per output channel

    if (accumulate) {
        const int32_t n_out = oh * ow;
        for (int32_t i = 0; i < n_out; ++i) {
            out[i] = static_cast<T_out>(0.0f);
        }
    }

    // Interior output-column range [ox_lo, ox_hi) where every kernel tap lands
    // inside the input for all ikw in [0, kw): the padded border columns are
    // peeled off so the inner loop needs no per-tap bounds check.
    //   lower (ikw=0):    ox*s0 - p0 >= 0            -> ox >= ceil(p0 / s0)
    //   upper (ikw=kw-1): ox*s0 + (kw-1)*d0 - p0 < iw
    // The divisors below are the (positive) strides; the numerators are
    // non-negative here (p0 >= 0, and the upper bound is clamped to 0), so the
    // unsigned divides fold to plain operations instead of a signed __divsi3.
    int32_t ox_lo =
        (p0 > 0)
            ? static_cast<int32_t>((static_cast<uint32_t>(p0) + static_cast<uint32_t>(s0) - 1u) /
                                   static_cast<uint32_t>(s0))
            : 0;
    // ceil(p0/s0) can exceed ow when the padding is large relative to the
    // output width: iw=1, kw=5, d0=2, p0=5, s0=1 gives ow=3 but ox_lo=5. Left
    // unclamped, the border loop below runs past the end of the row, and on the
    // last row past the end of the output plane buffer entirely.
    if (ox_lo > ow) {
        ox_lo = ow;
    }
    const int32_t hi_num = iw - 1 + p0 - (kw - 1) * d0;
    int32_t ox_hi =
        (hi_num < 0)
            ? 0
            : static_cast<int32_t>(static_cast<uint32_t>(hi_num) / static_cast<uint32_t>(s0)) + 1;
    if (ox_hi < ox_lo) {
        ox_hi = ox_lo;
    }
    if (ox_hi > ow) {
        ox_hi = ow;
    }

    // Vector chunks only when the input window is contiguous along ox (s0 == 1).
    const bool vectorize = (s0 == 1);

    // One interior chunk of W output columns, every tap in bounds, computed with
    // a broadcast-weight FMA. W is a template parameter so the interior can step
    // down through the vector widths below.
    auto vec_chunk = [&]<int32_t W>(int32_t ox, int32_t oy, T_out * __restrict out_row,
                                    const T_in * __restrict src_plane,
                                    const T_in * __restrict wt_base) {
        aie::accum<accfloat, W> acc;
        acc.from_vector(accumulate ? aie::load_unaligned_v<W>(out_row + ox)
                                   : aie::zeros<T_out, W>());
        GGML_CONV2D_UNROLL_TAPS
        for (int32_t ikh = 0; ikh < kh; ++ikh) {
            const int32_t iih = oy * s1 + ikh * d1 - p1;
            if (iih < 0 || iih >= ih) {
                continue;
            }
            const T_in * __restrict srow = src_plane + iih * iw;
            const T_in * __restrict wrow = wt_base + ikh * kw;
            GGML_CONV2D_UNROLL_TAPS
            for (int32_t ikw = 0; ikw < kw; ++ikw) {
                const int32_t iiw = ox + ikw * d0 - p0; // s0 == 1
                const aie::vector<T_in, W> wvec = aie::broadcast<T_in, W>(wrow[ikw]);
                const aie::vector<T_in, W> ivec = aie::load_unaligned_v<W>(srow + iiw);
                acc = aie::mac(acc, wvec, ivec);
            }
        }
        // Store the W lanes through an aligned temporary. out_row + ox is only
        // element-aligned in general (ox_lo is ceil(p0/s0), and out_row advances
        // by ow), and an unaligned 512-bit aie::store_unaligned_v() is a
        // read-modify-write over the whole 64-byte window around the target
        // whose preserved neighbour bytes come back displaced by 32 bytes. That
        // corrupts memory outside the W lanes, which here holds the
        // already-computed scalar border columns.
        alignas(W * sizeof(T_out)) T_out chunk[W];
        aie::store_v(chunk, acc.template to_vector<T_out>());
        for (int32_t l = 0; l < W; ++l) {
            out_row[ox + l] = chunk[l];
        }
    };

    // Channel reduction stays the outermost loop, accumulating into the output
    // plane, to avoid a Peano miscompile that dropped the iic>=1 contribution
    // when the channel loop sat between the spatial loops.
    for (int32_t iic = 0; iic < ic; ++iic) {
        const T_in * __restrict src_plane = in + iic * plane_size;
        // Weight base for this (oc_idx, iic) slice.
        const T_in * __restrict wt_base = wts + iic * knl_plane + oc_idx * knl_vol;

        for (int32_t oy = 0; oy < oh; ++oy) {
            T_out * __restrict out_row = out + oy * ow;

            // Left border columns: some taps fall in the padding, so each
            // element keeps its bounds check.
            for (int32_t ox = 0; ox < ox_lo; ++ox) {
                float acc = 0.0f;
                GGML_CONV2D_UNROLL_TAPS
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= ih) {
                        continue;
                    }
                    GGML_CONV2D_UNROLL_TAPS
                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox * s0 + ikw * d0 - p0;
                        if (iiw >= 0 && iiw < iw) {
                            acc += static_cast<float>(src_plane[iih * iw + iiw]) *
                                   static_cast<float>(wt_base[ikh * kw + ikw]);
                        }
                    }
                }
                out_row[ox] =
                    static_cast<T_out>(accumulate ? static_cast<float>(out_row[ox]) + acc : acc);
            }

            // Interior columns [ox_lo, ox_hi), widest vector first. Stepping
            // 16 -> 8 -> 4 lanes matters because the interior is not a multiple
            // of V: at OW=28 it is 26 columns, so a V-only loop leaves ten of
            // them to the scalar tail below, and that tail then costs more than
            // the vectorized part it was supposed to trim.
            //
            // Peeling the padded top/bottom rows out of these loops (the
            // vertical counterpart of the ox_lo/ox_hi split, so interior rows
            // drop the iih bounds check) was tried and reverted: it outlined the
            // border row body into a separate call with a ~4.5 KB frame. On the
            // MNIST conv1 shape it measured 122.50/122.51/122.67 ms against
            // 116.88/116.89/116.90 for the code below (three runs each, median
            // of 25 iterations, per-run sd 0.02-0.18%), so the regression is
            // well outside the run-to-run noise.
            int32_t ox = ox_lo;
            if (vectorize) {
                for (; ox + V <= ox_hi; ox += V) {
                    vec_chunk.template operator()<V>(ox, oy, out_row, src_plane, wt_base);
                }
                // At most one V/2 chunk can follow: the loop above leaves
                // fewer than V columns. An `if` states that invariant.
                if (ox + V / 2 <= ox_hi) {
                    vec_chunk.template operator()<V / 2>(ox, oy, out_row, src_plane, wt_base);
                    ox += V / 2;
                }
                // No V/4 step: a 128-bit chunk returns wrong results here. With
                // IW=IH=14, IC=8 (MNIST conv2) the interior is 12 columns, so
                // the widths in use are V/2 then V/4, and that configuration
                // mismatches the CPU reference on 1174 elements; dropping the
                // V/4 step makes the same shape exact. V and V/2 (512- and
                // 256-bit) are correct on every shape tested. Not diagnosed
                // further -- it is the same family as the 512-bit
                // store_unaligned_v defect worked around below.
            }

            // Whatever the vector widths could not cover: the sub-V/4 interior
            // remainder plus the true right border, all bounds-checked.
            for (; ox < ow; ++ox) {
                float acc = 0.0f;
                GGML_CONV2D_UNROLL_TAPS
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= ih) {
                        continue;
                    }
                    GGML_CONV2D_UNROLL_TAPS
                    for (int32_t ikw = 0; ikw < kw; ++ikw) {
                        const int32_t iiw = ox * s0 + ikw * d0 - p0;
                        if (iiw >= 0 && iiw < iw) {
                            acc += static_cast<float>(src_plane[iih * iw + iiw]) *
                                   static_cast<float>(wt_base[ikh * kw + ikw]);
                        }
                    }
                }
                out_row[ox] =
                    static_cast<T_out>(accumulate ? static_cast<float>(out_row[ox]) + acc : acc);
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
 *
 * The shape and op_params come from the -DGGML_CONV2D_* compile definitions
 * conv_2d.py emits for this specialization; see conv_2d_impl.
 */
void ggml_op_conv_2d(const INPUT_DTYPE * __restrict in,
                     const INPUT_DTYPE * __restrict wts,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t oc_idx) {
    conv_2d_impl<INPUT_DTYPE, OUTPUT_DTYPE>(in, wts, out, oc_idx);
}

} // extern "C"
