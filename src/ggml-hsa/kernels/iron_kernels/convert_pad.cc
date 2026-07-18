// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>
#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Widens one row from @p d0 to @p d0pad elements, zero-filling the tail.
 * @param[in]  in     Input row of @p d0 elements (INPUT_DTYPE).
 * @param[out] out    Output row of @p d0pad elements (OUTPUT_DTYPE).
 * @param[in]  d0     Number of valid elements.
 * @param[in]  d0pad  Padded row width (>= d0).
 */
void ggml_hsa_convert_pad(const INPUT_DTYPE * __restrict in,
                          OUTPUT_DTYPE * __restrict out,
                          int32_t d0,
                          int32_t d0pad) {
    event0();

    // Row shape is fixed per JIT-compiled instance; compile-time bounds let Peano fold the
    // trip count and pipeline the loop. Fall back to the runtime args if undefined.
#ifdef CONVERT_PAD_D0
    constexpr int32_t d0v = CONVERT_PAD_D0;
    constexpr int32_t d0padv = CONVERT_PAD_D0PAD;
    (void)d0;
    (void)d0pad;
#else
    const int32_t d0v = d0;
    const int32_t d0padv = d0pad;
#endif

#ifdef CONVERT_PAD_PAD_ONLY
    static_assert(std::is_same_v<INPUT_DTYPE, OUTPUT_DTYPE>,
                  "CONVERT_PAD_PAD_ONLY requires matching input and output types");
    // bf16 -> bf16: pad only. Aligned store dodges the broken 16-bit unaligned vector store
    // (output rows are d0pad-wide, a tile multiple, so vector-aligned; input load stays
    // unaligned since the per-row fifo stride is not).
    constexpr int32_t V = 512 / (sizeof(OUTPUT_DTYPE) * 8);
    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    AIE_PREPARE_FOR_PIPELINING
    // Only bind the trip-count hint when the row spans a full vector; nblk == 0 (row < V) makes
    // AIE_LOOP_RANGE(0, 0) an invalid Peano pragma.
#if defined(CONVERT_PAD_D0) && (CONVERT_PAD_D0) >= 32
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
        const aie::vector<OUTPUT_DTYPE, V> v = aie::load_unaligned_v<V>(in + i);
        aie::store_v(out + i, v);
    }

    for (int32_t i = vend; i < d0v; ++i) {
        out[i] = in[i];
    }
#else
    constexpr int32_t V = 512 / (sizeof(f32) * 8);
    const int32_t nblk = d0v / V;
    const int32_t vend = nblk * V;

    // Vectorized f32 -> bf16, replicating ggml_compute_fp32_to_bf16's integer arithmetic
    // lane-wise for bit-identical results (not relying on hardware rounding/NaN handling).
    AIE_PREPARE_FOR_PIPELINING
#ifdef CONVERT_PAD_D0
    AIE_LOOP_RANGE(nblk, nblk)
#endif
    for (int32_t b = 0; b < nblk; ++b) {
        const int32_t i = b * V;
        const aie::vector<f32, V> fv = aie::load_unaligned_v<V>(in + i);
        // Output rows are d0pad-wide (tile-multiple, vector-aligned), so an aligned store is safe.
        aie::store_v(out + i, convert_f32_to_bf16_vector<V>(fv));
    }

    for (int32_t i = vend; i < d0v; ++i) {
        const uint16_t hi = convert_f32_to_bf16_scalar(in[i]);
        __builtin_memcpy(&out[i], &hi, sizeof(bf16));
    }
#endif

    // Scalar: 16-bit unaligned vector store is broken in this aie_api version, and d0 is not
    // vector-aligned.
    const OUTPUT_DTYPE zero = {};
    for (int32_t i = d0v; i < d0padv; ++i) {
        out[i] = zero;
    }

    event1();
}

} // extern "C"
