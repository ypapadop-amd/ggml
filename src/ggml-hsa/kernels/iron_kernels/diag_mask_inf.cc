// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file diag_mask_inf.cc
 * @brief Causal diagonal masking (GGML_OP_DIAG_MASK_INF) for AIE kernels.
 */

#include <stdint.h>

#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

extern "C" {

/**
 * @brief Applies the causal diagonal mask to one row.
 *
 * For an [nc, nr, nz] tensor, row j (within its z-slice) keeps columns
 * [0, n_past + j] from @p in and gets -inf in columns (n_past + j, nc). This
 * matches ggml_compute_forward_diag_mask_f32 with value = -inf: it masks column
 * i whenever i > n_past + j.
 *
 * Rows stream in slice-major order (nr consecutive rows per z-slice), so the row
 * index within the slice is recovered as j = tile_idx % nr, i.e. the causal
 * pattern repeats every nr rows regardless of how many z-slices there are.
 *
 * @param[in]  in       Input row of N float elements.
 * @param[out] out      Output row of N float elements (may alias @p in).
 * @param[in]  N        Row length (nc).
 * @param[in]  nr       Rows per z-slice (ne1).
 * @param[in]  n_past   Past tokens kept unmasked on every row.
 * @param[in]  tile_idx Global row index of this tile.
 */
void ggml_op_diag_mask_inf(const float * __restrict in,
                           float * __restrict out,
                           int32_t N,
                           int32_t nr,
                           int32_t n_past,
                           int32_t tile_idx) {
    event0();

    const int32_t j = tile_idx % nr;

    // Columns [0, keep) are copied from the input; [keep, N) are masked to -inf.
    int32_t keep = n_past + j + 1;
    if (keep < 0) {
        keep = 0;
    }
    if (keep > N) {
        keep = N;
    }

#if __AIEARCH__ == 20 // aie2 only so far; aie2p keeps the scalar loops below, unmeasured there.
    // One pass in vectors: each lane is kept or masked by comparing its column with keep, where
    // the scalar copy below ran at II 10 per element. Rows need not start vector-aligned, so the
    // accesses are unaligned; the last N % V columns stay scalar.
    constexpr int32_t V = 16;
    const int32_t vend = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V - 1});
    alignas(aie::vector_decl_align) static constexpr int32_t lane_index[V] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const aie::vector<int32_t, V> lanes = aie::load_v<V>(lane_index);
    const aie::vector<float, V> neg_inf =
        aie::broadcast<float, V>(-std::numeric_limits<float>::infinity());
    const auto body = [&](int32_t i) {
        const auto masked = aie::ge(aie::add(lanes, i), keep);
        aie::store_unaligned_v(out + i,
                               aie::select(aie::load_unaligned_v<V>(in + i), neg_inf, masked));
    };
    GGML_AIE_LOOP_MIN_TRIPS(4, i, vend, V, body(i);)
    for (int32_t i = vend; i < N; ++i) {
        out[i] = i < keep ? in[i] : -std::numeric_limits<float>::infinity();
    }
#else
    for (int32_t i = 0; i < keep; ++i) {
        out[i] = in[i];
    }
    for (int32_t i = keep; i < N; ++i) {
        out[i] = -std::numeric_limits<float>::infinity();
    }
#endif

    event1();
}

} // extern "C"
