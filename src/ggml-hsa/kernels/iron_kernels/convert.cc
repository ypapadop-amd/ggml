// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file convert.cc
 * @brief Element-wise dtype conversion (no shape change), for on-device GGML_OP_CPY casts.
 *
 * Converts a flat run of CONVERT_N elements from INPUT_DTYPE to OUTPUT_DTYPE. Unlike convert_pad
 * this does no padding: input and output are the same length and contiguous. Used to run a pure
 * dtype cast (e.g. the f32<->bf16 casts of the bf16 MNIST graph, or the f16 <-> bf16 pre- and
 * post-processing of f16 tensors) on the device queue instead of a host-side copy that would drain
 * the queue.
 *
 * Every direction is bit-identical to the host: f32 -> bf16 replicates ggml_compute_fp32_to_bf16
 * (round-to-nearest-even, NaN -> quiet); bf16 -> f32 is an exact widening (the bf16 bits become the
 * high half of the f32); f32 -> f32 and bf16 -> bf16 are copies. f16 -> bf16 widens the raw f16
 * bits to f32 (exact) and then applies the same RNE narrowing; bf16 -> f16 narrows with the replica
 * of ggml_compute_fp32_to_fp16. The two 16-bit directions convert normal values and zeros in
 * vectors and leave any vector holding another class to a scalar fix-up pass. f16 arrives (and
 * leaves) as i16 because IRON has no f16 element type.
 *
 * The same code runs on aie2 and aie2p.
 */

#include <aie_api/aie.hpp>
#include <bit>
#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

// convert.py fixes both per JIT-compiled instance: the tile length, and the vector width, which
// divides it. Every trip count below is a compile-time constant, and every vector access is
// aligned: each tile buffer is a whole number of vectors, so each starts on a vector boundary.
// There are no unaligned stores, which would rewrite the 32-byte blocks around the tile's ends.
// CONVERT_LANES == 1 means no vector path.
#if !defined(CONVERT_N) || !defined(CONVERT_LANES)
#error "convert.cc needs -DCONVERT_N and -DCONVERT_LANES"
#endif

namespace {

constexpr std::int32_t Nv = CONVERT_N;
constexpr int VL = CONVERT_LANES;
static_assert(Nv % VL == 0, "the tile must be a whole number of vectors");

/// The vector width: VL, or, with no vector path (VL == 1), any width aie_api has, since the
/// vector code is then compiled but never runs.
constexpr int V = VL == 1 ? 16 : VL;

/// out[i] = vec_op(in[i]) over the tile, VL lanes at a time, or scalar_op(i) for each element.
template <typename TIn, typename TOut, typename VecOp, typename ScalarOp>
inline void
convert_tile(const TIn * __restrict in, TOut * __restrict out, VecOp vec_op, ScalarOp scalar_op) {
    if constexpr (VL == 1) {
        for (std::int32_t i = 0; i < Nv; ++i) {
            scalar_op(i);
        }
    } else {
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_NO_UNROLL
        for (std::int32_t i = 0; i < Nv; i += V) {
            aie::store_v(out + i, vec_op(aie::load_v<V>(in + i)));
        }
    }
}

/**
 * @brief A 16-bit -> 16-bit conversion whose vector form covers only some lanes.
 *
 * One branch-free pass converts every vector and records whether all lanes were inside
 * @p covers; only if not does a second pass find the vectors that were not and convert those
 * element by element with @p scalar_op, the reference. Typical data (normal values and zeros)
 * never takes the second pass.
 */
template <typename CoversOp, typename VecOp, typename ScalarOp>
inline void convert_tile_16(const std::uint16_t * __restrict in,
                            std::uint16_t * __restrict out,
                            CoversOp covers,
                            VecOp vec_op,
                            ScalarOp scalar_op) {
    if constexpr (VL == 1) {
        for (std::int32_t i = 0; i < Nv; ++i) {
            out[i] = scalar_op(in[i]);
        }
    } else {
        aie::mask<V> all_covered = aie::mask<V>(true);
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_NO_UNROLL
        for (std::int32_t i = 0; i < Nv; i += V) {
            const aie::vector<std::uint16_t, V> bits = aie::load_v<V>(in + i);
            aie::store_v(out + i, vec_op(bits));
            all_covered = all_covered & covers(bits);
        }
        if (all_covered.full()) {
            return;
        }
        for (std::int32_t base = 0; base < Nv; base += V) {
            if (!covers(aie::load_v<V>(in + base)).full()) {
                for (std::int32_t i = base; i < base + V; ++i) {
                    out[i] = scalar_op(in[i]);
                }
            }
        }
    }
}

} // namespace

extern "C" {

/**
 * @brief Converts CONVERT_N elements from INPUT_DTYPE to OUTPUT_DTYPE (same length, contiguous).
 *
 * @param[in]  in  Input array of CONVERT_N elements.
 * @param[out] out Output array of CONVERT_N elements.
 * @param[in]  N   Number of elements to convert; unused, the length is CONVERT_N.
 */
void ggml_hsa_convert(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    event0();
    (void)N;

#if defined(CONVERT_F16_TO_BF16) || defined(CONVERT_BF16_TO_F16)
    // Both sides are 16-bit; work on the raw bits.
    const auto * in16 = reinterpret_cast<const std::uint16_t *>(in);
    auto * out16 = reinterpret_cast<std::uint16_t *>(out);
#endif

#ifdef CONVERT_F16_TO_BF16
    // f16 -> bf16: normal f16s and zeros in vectors, anything else through the scalar reference,
    // identical to the host's GGML_FP32_TO_BF16(GGML_FP16_TO_FP32(v)).
    convert_tile_16(
        in16, out16, [](const auto & h) { return convert_f16_bits_to_bf16_vector_covers<V>(h); },
        [](const auto & h) { return convert_f16_bits_to_bf16_vector<V>(h); },
        [](std::uint16_t h) { return ::convert_f32_to_bf16_scalar(::convert_f16_bits_to_f32(h)); });
#elif defined(CONVERT_BF16_TO_F16)
    // bf16 -> f16: a result in the normal f16 range, or a signed zero, is a pure field re-pack;
    // anything else (f16 subnormals, overflow, inf, NaN) goes through the scalar reference,
    // identical to the host's GGML_FP32_TO_FP16(GGML_BF16_TO_FP32(v)).
    convert_tile_16(
        in16, out16, [](const auto & b) { return convert_bf16_to_f16_vector_covers<V>(b); },
        [](const auto & b) { return convert_bf16_to_f16_vector<V>(b); },
        [](std::uint16_t b) {
            return ::convert_f32_to_f16_scalar(std::bit_cast<f32>(std::uint32_t{b} << 16));
        });
#elif defined(CONVERT_F32_TO_BF16)
    // f32 -> bf16, bit-identical to the host reference.
    convert_tile(
        in, out,
        [](const aie::vector<f32, V> & v) { return convert_f32_to_bf16_vector_hi16<V>(v); },
        [&](std::int32_t i) {
            reinterpret_cast<std::uint16_t *>(out)[i] = ::convert_f32_to_bf16_scalar(in[i]);
        });
#elif defined(CONVERT_BF16_TO_F32)
    // bf16 -> f32, exact: the bf16 bits become the high half of each f32, under a zero low half.
    convert_tile(
        in, out,
        [](const aie::vector<bf16, V> & v) {
            const auto [lo, hi] = aie::interleave_zip(aie::zeros<std::uint16_t, V>(),
                                                      aie::vector_cast<std::uint16_t>(v), 1);
            return aie::vector_cast<f32>(aie::concat(lo, hi));
        },
        [&](std::int32_t i) { out[i] = static_cast<f32>(in[i]); });
#else
    // Same-dtype copy.
    convert_tile(
        in, out, [](const auto & v) { return v; }, [&](std::int32_t i) { out[i] = in[i]; });
#endif

    event1();
}

} // extern "C"
