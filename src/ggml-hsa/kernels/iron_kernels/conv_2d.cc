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

// This kernel is always built through conv_2d.py, which emits one
// specialization per configuration and passes every extent and op_param as a
// -DGGML_CONV2D_* literal. Fail with a readable message rather than a cascade
// of undefined-identifier errors (and an #if that would silently read a missing
// tap count as zero) if it is compiled without them.
#if !defined(GGML_CONV2D_IW) || !defined(GGML_CONV2D_IH) || !defined(GGML_CONV2D_IC) ||            \
    !defined(GGML_CONV2D_KW) || !defined(GGML_CONV2D_KH) || !defined(GGML_CONV2D_OW) ||            \
    !defined(GGML_CONV2D_OH) || !defined(GGML_CONV2D_S0) || !defined(GGML_CONV2D_S1) ||            \
    !defined(GGML_CONV2D_P0) || !defined(GGML_CONV2D_P1) || !defined(GGML_CONV2D_D0) ||            \
    !defined(GGML_CONV2D_D1)
#error "conv_2d.cc requires the -DGGML_CONV2D_* shape defines emitted by conv_2d.py"
#endif

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
#define GGML_CONV2D_UNROLL_TAPS AIE_LOOP_UNROLL_FULL
#else
#define GGML_CONV2D_UNROLL_TAPS
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

#if __AIEARCH__ == 20
    // Padded-row layout and the pieces the two aie2 vector paths below share; the per-row path's
    // comment explains the layout and why the taps are shuffled out of aligned loads.
    constexpr int32_t n_chunks = (ow + V - 1) / V;
    constexpr int32_t pw = (n_chunks + 1) * V;
    constexpr int32_t copy_lo = (p0 < pw) ? p0 : pw;           // first copied column
    constexpr int32_t copy_hi = (p0 + iw < pw) ? p0 + iw : pw; // one past the last
    // Zeroes a padded row's padding columns.
    [[maybe_unused]] const auto zero_pad = [](T_in * row) {
        for (int32_t j = 0; j < copy_lo; ++j) {
            row[j] = static_cast<T_in>(0.0f);
        }
        for (int32_t j = copy_hi; j < pw; ++j) {
            row[j] = static_cast<T_in>(0.0f);
        }
    };
    // Accumulates one padded row's kw taps into the V output columns at ox.
    [[maybe_unused]] const auto mac_row = [](aie::accum<accfloat, V> & acc, const T_in * row,
                                             int32_t ox, const T_in * __restrict wrow) {
        const aie::vector<T_in, V> lo = aie::load_v<V>(row + ox);
        const aie::vector<T_in, V> hi = aie::load_v<V>(row + ox + V);
        GGML_CONV2D_UNROLL_TAPS
        for (int32_t ikw = 0; ikw < kw; ++ikw) {
            const aie::vector<T_in, V> tap =
                (ikw * d0 == 0) ? lo : aie::shuffle_down_fill(lo, hi, ikw * d0);
            acc = aie::mac(acc, aie::broadcast<T_in, V>(wrow[ikw]), tap);
        }
    };
    // Stores acc's first `lanes` lanes at dst through an aligned bounce: unaligned 512-bit
    // stores corrupt their neighbours here (see vec_chunk below), and the partial last chunk must
    // not touch memory past the row.
    [[maybe_unused]] const auto store_lanes =
        [](T_out * __restrict dst, const aie::accum<accfloat, V> & acc, int32_t lanes) {
            alignas(aie::vector_decl_align) T_out chunk[V];
            aie::store_v(chunk, acc.template to_vector<T_out>());
            AIE_LOOP_NO_UNROLL
            for (int32_t l = 0; l < lanes; ++l) {
                dst[l] = chunk[l];
            }
        };
#endif

#if __AIEARCH__ ==                                                                                 \
    20 // aie2 only so far; aie2p keeps the border-peeled path below, unmeasured there.
    // Padded rows held in a ring, for all input channels at once, so each input row is copied once
    // per call (not once per output row that reads it) and every output chunk sums all channels
    // in registers before one store (no zeroed plane, no per-channel reload). The per-row path
    // further down does the same arithmetic one channel at a time, for shapes whose ring does not
    // fit in GGML_CONV2D_RING_BYTES (set by conv_2d.py from the stack size); see its comment for
    // the padded-row layout and why the taps are shuffled.
    if constexpr (s0 == 1 && (kw - 1) * d0 <= V) {
        // One output row's taps read input rows base, base + d1, ..., base + (kh - 1) * d1, with
        // base = oy * s1 - p1: a span of R rows, held in R slots. Slots are assigned relative to
        // base_slot, the slot of row `base`, and advance with it, rather than computed as r % R:
        // reducing by a constant that is not a power of two needs a 64-bit multiply, a runtime
        // call on the aie2 scalar unit. Where the ring starts does not matter.
        constexpr int32_t R = (kh - 1) * d1 + 1;
        constexpr int32_t s1_mod_r = s1 % R;
#ifndef GGML_CONV2D_RING_BYTES
#error "conv_2d.cc requires -DGGML_CONV2D_RING_BYTES, emitted by conv_2d.py"
#endif
        if constexpr (R * ic * pw * static_cast<int32_t>(sizeof(T_in)) <= GGML_CONV2D_RING_BYTES) {
            alignas(aie::vector_decl_align) T_in ring[ic][R][pw];
            for (int32_t c = 0; c < ic; ++c) {
                for (int32_t r = 0; r < R; ++r) {
                    zero_pad(ring[c][r]);
                }
            }

            const auto wrap = [](int32_t slot) { return slot >= R ? slot - R : slot; };
            int32_t next_row = 0;  // lowest input row not yet copied
            int32_t base_slot = 0; // slot of input row `base`
            for (int32_t oy = 0; oy < oh; ++oy) {
                const int32_t base = oy * s1 - p1;
                if (oy > 0) {
                    base_slot = wrap(base_slot + s1_mod_r);
                }
                const int32_t first = base > next_row ? base : next_row;
                const int32_t last = (base + R < ih ? base + R : ih); // one past
                for (int32_t r = first; r < last; ++r) {
                    const int32_t slot = wrap(base_slot + (r - base)); // r - base in [0, R)
                    for (int32_t c = 0; c < ic; ++c) {
                        const T_in * __restrict srow = in + c * plane_size + r * iw;
                        AIE_LOOP_NO_UNROLL
                        for (int32_t j = copy_lo; j < copy_hi; ++j) {
                            ring[c][slot][j] = srow[j - p0];
                        }
                    }
                }
                if (last > next_row) {
                    next_row = last;
                }

                T_out * __restrict out_row = out + oy * ow;
                for (int32_t ch = 0; ch < n_chunks; ++ch) {
                    const int32_t ox = ch * V;
                    const int32_t lanes = (ow - ox < V) ? ow - ox : V;
                    aie::accum<accfloat, V> acc;
                    acc.from_vector(aie::zeros<T_out, V>());
                    for (int32_t c = 0; c < ic; ++c) {
                        const T_in * __restrict wt_base = wts + c * knl_plane + oc_idx * knl_vol;
                        GGML_CONV2D_UNROLL_TAPS
                        for (int32_t ikh = 0; ikh < kh; ++ikh) {
                            const int32_t r = base + ikh * d1;
                            if (r < 0 || r >= ih) {
                                continue;
                            }
                            const int32_t slot = wrap(base_slot + ikh * d1); // ikh * d1 < R
                            mac_row(acc, ring[c][slot], ox, wt_base + ikh * kw);
                        }
                    }
                    store_lanes(out_row + ox, acc, lanes);
                }
            }

            event1();
            return;
        }
    }
#endif

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

#if __AIEARCH__ ==                                                                                 \
    20 // aie2 only so far; aie2p keeps the border-peeled path below, unmeasured there.
    if constexpr (s0 == 1 && (kw - 1) * d0 <= V) {
        // Every column in vectors, none in the bounds-checked scalar loops below: those paid a
        // runtime call (__mulsf3) per tap, and were 83% (MNIST conv1) and 94% (conv2) of the
        // kernel's time with their taps ablated. Each input row a tap reads is first copied into
        // a zero-padded row, so the padding columns read as zeros and every tap of every chunk
        // is a plain broadcast-weight MAC, with no reads outside buffers this kernel owns.
        //
        // prow[ikh][j] holds input column j - p0 of the row tap ikh reads (0 in the padding).
        // Output column ox reads prow[ikh][ox + ikw * d0]. A chunk at ox loads the two aligned
        // vectors at ox and ox + V and shifts each tap's window out of them, hence pw, and
        // (kw - 1) * d0 <= V. Unaligned 512-bit loads from these small stack rows instead
        // crash Peano's aie2 backend at some shapes: LLVM forms the shifted taps from one wide
        // load as lshr/trunc on i512, which the legalizer rejects ("unable to legalize
        // instruction: G_BITCAST <8 x s32> from s256", MNIST conv2; "cannot select
        // G_CONCAT_VECTORS" at 5x5). Reproduced in six lines of IR on llvm-aie b0d37423.
        alignas(aie::vector_decl_align) T_in prow[kh][pw];
        for (int32_t ikh = 0; ikh < kh; ++ikh) {
            zero_pad(prow[ikh]);
        }

        // The channel reduction stays the outermost loop, accumulating into the output plane
        // (see the comment on the loop below).
        for (int32_t iic = 0; iic < ic; ++iic) {
            const T_in * __restrict src_plane = in + iic * plane_size;
            const T_in * __restrict wt_base = wts + iic * knl_plane + oc_idx * knl_vol;

            for (int32_t oy = 0; oy < oh; ++oy) {
                T_out * __restrict out_row = out + oy * ow;

                bool row_in[kh];
                for (int32_t ikh = 0; ikh < kh; ++ikh) {
                    const int32_t iih = oy * s1 + ikh * d1 - p1;
                    row_in[ikh] = iih >= 0 && iih < ih;
                    if (row_in[ikh]) {
                        const T_in * __restrict srow = src_plane + iih * iw;
                        AIE_LOOP_NO_UNROLL
                        for (int32_t j = copy_lo; j < copy_hi; ++j) {
                            prow[ikh][j] = srow[j - p0];
                        }
                    }
                }

                for (int32_t c = 0; c < n_chunks; ++c) {
                    const int32_t ox = c * V;
                    const int32_t lanes = (ow - ox < V) ? ow - ox : V;
                    aie::accum<accfloat, V> acc;
                    if constexpr (accumulate) {
                        // The partial last chunk must not read past the row either.
                        alignas(aie::vector_decl_align) T_out chunk[V];
                        AIE_LOOP_NO_UNROLL
                        for (int32_t l = 0; l < lanes; ++l) {
                            chunk[l] = out_row[ox + l];
                        }
                        acc.from_vector(aie::load_v<V>(chunk));
                    } else {
                        acc.from_vector(aie::zeros<T_out, V>());
                    }
                    GGML_CONV2D_UNROLL_TAPS
                    for (int32_t ikh = 0; ikh < kh; ++ikh) {
                        if (!row_in[ikh]) {
                            continue;
                        }
                        mac_row(acc, prow[ikh], ox, wt_base + ikh * kw);
                    }
                    store_lanes(out_row + ox, acc, lanes);
                }
            }
        }

        event1();
        return;
    }
#endif

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
