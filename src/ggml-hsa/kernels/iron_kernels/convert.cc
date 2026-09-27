// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file convert.cc
 * @brief Element-wise dtype conversion (no shape change), for on-device GGML_OP_CPY casts.
 *
 * Converts a flat run of @p N elements from INPUT_DTYPE to OUTPUT_DTYPE. Unlike convert_pad this
 * does no padding: input and output are the same length and contiguous. Used to run a pure dtype
 * cast (e.g. the f32<->bf16 casts of the bf16 MNIST graph) on the device queue instead of a
 * host-side copy that would drain the queue.
 *
 * The f32 -> bf16 direction replicates the host ggml_compute_fp32_to_bf16 integer arithmetic
 * (round-to-nearest-even, NaN -> quiet) bit-for-bit, matching convert_pad. The bf16 -> f32
 * direction is an exact widening. f32 -> f32 (or bf16 -> bf16) is a plain copy. The f16 -> bf16
 * direction widens the raw f16 bits to f32 (exact) and then reuses the same bit-exact RNE
 * narrowing. The f32 -> f16 and bf16 -> f16 directions narrow with the bit-exact replica of the
 * host ggml_compute_fp32_to_fp16 instead; bf16 -> f16 additionally has a vector fast path for the
 * lanes that need no rounding. f16 arrives (and leaves) as i16 because IRON has no f16 element
 * type.
 */

#include <aie_api/aie.hpp>
#include <cstdint>
#include <cstring>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Converts @p N elements from INPUT_DTYPE to OUTPUT_DTYPE (same length, contiguous).
 *
 * @param[in]  in  Input array of @p N elements.
 * @param[out] out Output array of @p N elements.
 * @param[in]  N   Number of elements to convert.
 */
void ggml_hsa_convert(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    event0();

    // Row length is fixed per JIT-compiled kernel instance, so convert.py passes it as a -D define.
    // A compile-time trip count lets Peano fold it and software-pipeline the hot loop.
#ifdef CONVERT_N
    constexpr int32_t Nv = CONVERT_N;
    (void)N;
#else
    const int32_t Nv = N;
#endif

#ifdef CONVERT_F16_TO_BF16
    {
        // f16 -> bf16. IRON has no f16 element type, so the input arrives as raw i16 bit patterns.
        // Widen each to f32 exactly (no rounding is possible in that direction), then apply the
        // same bit-exact RNE narrowing the f32 -> bf16 path uses, which makes the composition
        // identical to the host's GGML_FP32_TO_BF16(GGML_FP16_TO_FP32(v)).
        for (int32_t i = 0; i < Nv; ++i) {
            uint16_t h = 0;
            std::memcpy(&h, &in[i], sizeof(h));
            const uint16_t lo = ::convert_f32_to_bf16_scalar(::convert_f16_bits_to_f32(h));
            std::memcpy(&out[i], &lo, sizeof(bf16));
        }
    }
#elif defined(CONVERT_BF16_TO_F16)
    {
        // bf16 -> f16. IRON has no f16 element type, so the result is stored into the i16 the
        // destination is streamed as. The scalar reference widens to f32 exactly (f32 has both more
        // exponent range and more mantissa bits, so no rounding is possible) and then applies the
        // bit-exact RNE narrowing, which makes it identical to the host's
        // GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(v)).
        //
        // Most lanes need none of that. A result that lands in the normal f16 range is a pure field
        // re-pack -- bf16's 7 mantissa bits fit in f16's 10, so nothing rounds -- and so is a
        // signed zero, and both vectorize with uniform shifts. A block whose lanes are all of that
        // kind takes the vector path; any other block (f16 subnormals, overflow, inf, NaN) falls
        // back to the scalar reference for the whole block, so the result is bit-identical either
        // way. The branch is per block of V, not per element.
        constexpr int32_t V = 512 / (sizeof(uint16_t) * 8);
        const int32_t nblk = Nv / V;
        const int32_t vend = nblk * V;

        // Flattened 1D tiling: each streamed tile starts at a vector-aligned base, so aligned
        // load/store are safe here (unlike convert_pad's per-row stride).
        for (int32_t blk = 0; blk < nblk; ++blk) {
            const int32_t base = blk * V;
            const aie::vector<uint16_t, V> bits =
                aie::vector_cast<uint16_t>(aie::load_v<V>(in + base));
            if (convert_bf16_to_f16_vector_covers<V>(bits).full()) {
                aie::store_v(out + base,
                             aie::vector_cast<OUTPUT_DTYPE>(convert_bf16_to_f16_vector<V>(bits)));
            } else {
                for (int32_t i = base; i < base + V; ++i) {
                    const uint16_t h = ::convert_f32_to_f16_scalar(static_cast<f32>(in[i]));
                    std::memcpy(&out[i], &h, sizeof(OUTPUT_DTYPE));
                }
            }
        }

        for (int32_t i = vend; i < Nv; ++i) {
            const uint16_t h = ::convert_f32_to_f16_scalar(static_cast<f32>(in[i]));
            std::memcpy(&out[i], &h, sizeof(OUTPUT_DTYPE));
        }
    }
#elif defined(CONVERT_F32_TO_F16)
    {
        // f32 -> f16, the bit-exact RNE replica of the host ggml_compute_fp32_to_fp16. Scalar: the
        // f16 subnormal range needs a per-lane variable shift, which aie_api has no vector form of
        // (only a uniform `unsigned shift`), and unlike bf16 -> f16 the normal range rounds too, so
        // there is no re-pack shortcut to vectorize instead.
        for (int32_t i = 0; i < Nv; ++i) {
            const uint16_t h = ::convert_f32_to_f16_scalar(in[i]);
            std::memcpy(&out[i], &h, sizeof(OUTPUT_DTYPE));
        }
    }
#elif defined(CONVERT_F32_TO_BF16)
    {
        // f32 -> bf16, bit-identical to the host reference (see convert_pad.cc for the derivation).
        constexpr int32_t V = 512 / (sizeof(f32) * 8);
        const int32_t nblk = Nv / V;
        const int32_t vend = nblk * V;

        AIE_PREPARE_FOR_PIPELINING
        // Only bind the trip-count hint when the tile spans at least one full vector; a tile with
        // CONVERT_N < V gives nblk == 0, for which AIE_LOOP_RANGE(0, 0) is an invalid Peano pragma
        // (min_iteration_count must be positive). Small tiles fall through to the scalar tail.
#if defined(CONVERT_N) && (CONVERT_N) >= 16
        AIE_LOOP_RANGE(nblk, nblk)
#endif
        // Flattened 1D tiling: each streamed tile starts at a vector-aligned base, so aligned
        // load/store are safe here (unlike convert_pad's per-row stride). Aligned store also dodges
        // the broken 16-bit unaligned vector store in this aie_api version.
        for (int32_t b = 0; b < nblk; ++b) {
            const int32_t i = b * V;
            const aie::vector<f32, V> fv = aie::load_v<V>(in + i);
            aie::store_v(out + i, convert_f32_to_bf16_vector<V>(fv));
        }

        for (int32_t i = vend; i < Nv; ++i) {
            const uint16_t hi = ::convert_f32_to_bf16_scalar(in[i]);
            std::memcpy(&out[i], &hi, sizeof(bf16));
        }
    }
#else
    {
        // bf16 -> f32 widening, or a same-dtype copy. Scalar static_cast handles all remaining
        // dtype pairs exactly (widening never loses bits; same-dtype is a plain copy).
        for (int32_t i = 0; i < Nv; ++i) {
            out[i] = static_cast<OUTPUT_DTYPE>(in[i]);
        }
    }
#endif

    event1();
}

} // extern "C"
