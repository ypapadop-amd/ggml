// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>
#include <cstdint>
#include <cstring>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

// Templated on IN/OUT so the `if constexpr` mode branches below are genuinely dependent and the
// untaken branch (e.g. f32->bf16 conversion math when IN/OUT are both bf16) is discarded rather
// than type-checked; ggml_hsa_depad itself is not a template, so a non-dependent `if constexpr`
// there would still require every branch to compile for every dtype pairing.
template <typename IN, typename OUT>
static inline void
depad_impl(const IN * __restrict in, OUT * __restrict out, int32_t d0, int32_t d0pad) {
    event0();
    (void)d0pad;

    constexpr bool kConvertF32ToBf16 = std::is_same_v<IN, f32> && std::is_same_v<OUT, bf16>;
    // f32 -> bf16 vectorizes at f32 width (one f32 lane per output bf16 lane); other modes
    // vectorize at the matching IN/OUT width.
    constexpr int32_t V = kConvertF32ToBf16 ? 512 / (sizeof(f32) * 8) : 512 / (sizeof(OUT) * 8);

    // Row width is fixed per JIT-compiled instance; compile-time bound lets Peano fold the trip
    // count and pipeline the loop. Fall back to the runtime arg if undefined.
#ifdef DEPAD_D0
    constexpr int32_t d0v = DEPAD_D0;
    (void)d0;
#else
    const int32_t d0v = d0;
#endif

    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    // No AIE_LOOP_MIN_ITERATION_COUNT: d0 can be < V (e.g. fc2 d0=10), so nblk may be 0.
    AIE_PREPARE_FOR_PIPELINING
    // Only bind the range hint when nblk > 0; AIE_LOOP_RANGE(0, 0) is an invalid Peano pragma.
#if defined(DEPAD_D0) && (DEPAD_D0) >= 16
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
        if constexpr (kConvertF32ToBf16) {
            // f32 -> bf16, bit-identical to the host reference. Aligned store dodges the broken
            // 16-bit unaligned vector store in this aie_api version (output rows are dense bf16,
            // vector-aligned); input stays unaligned.
            const aie::vector<f32, V> fv = aie::load_unaligned_v<V>(in + i);
            aie::store_v(out + i, convert_f32_to_bf16_vector<V>(fv));
        } else {
            // f32 -> f32 or bf16 -> bf16: plain copy, no conversion.
            static_assert(std::is_same_v<IN, OUT>,
                          "Plain-copy depad requires matching IN/OUT types");
            const aie::vector<OUT, V> v = aie::load_unaligned_v<V>(in + i);
            aie::store_unaligned_v(out + i, v);
        }
    }

    for (int32_t i = vend; i < d0v; ++i) {
        if constexpr (kConvertF32ToBf16) {
            const uint16_t hi = convert_f32_to_bf16_scalar(in[i]);
            std::memcpy(&out[i], &hi, sizeof(bf16));
        } else {
            static_assert(std::is_same_v<IN, OUT>,
                          "Plain-copy depad requires matching IN/OUT types");
            out[i] = in[i];
        }
    }

    event1();
}

extern "C" {

/**
 * @brief Narrows one row from @p d0pad to @p d0 elements, with an optional dtype convert.
 *
 * @param[in]  in     Input padded row of @p d0pad elements (INPUT_DTYPE).
 * @param[out] out    Output dense row of @p d0 elements (OUTPUT_DTYPE).
 * @param[in]  d0     Number of valid elements to copy.
 * @param[in]  d0pad  Padded input row width (>= d0); only [0, d0) is read.
 */
void ggml_hsa_depad(const INPUT_DTYPE * __restrict in,
                    OUTPUT_DTYPE * __restrict out,
                    int32_t d0,
                    int32_t d0pad) {
    depad_impl<INPUT_DTYPE, OUTPUT_DTYPE>(in, out, d0, d0pad);
}

} // extern "C"
