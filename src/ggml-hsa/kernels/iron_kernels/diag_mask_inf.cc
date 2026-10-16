// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file diag_mask_inf.cc
 * @brief Causal diagonal masking (GGML_OP_DIAG_MASK_INF) for AIE kernels.
 */

#include <stdint.h>

#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

extern "C" {

/**
 * @brief Applies the causal diagonal mask to a tile of whole rows.
 *
 * For an [nc, nr, nz] tensor, row j (within its z-slice) keeps columns
 * [0, n_past + j] from @p in and gets -inf in columns (n_past + j, nc). This
 * matches ggml_compute_forward_diag_mask_f32 with value = -inf: it masks column
 * i whenever i > n_past + j.
 *
 * Rows stream in slice-major order (nr consecutive rows per z-slice), so the row
 * index within the slice is j = row % nr: the causal pattern repeats every nr rows
 * regardless of how many z-slices there are. It is reduced once per tile and then
 * advanced row by row, since a modulo by a run-time value is a runtime call.
 *
 * A row is one pass of 16-lane vectors, each lane kept or set to -inf by comparing its column
 * with the row's cut-off, walking cursors (Peano does not strength-reduce base + i). The last
 * N % 16 columns are one more vector ending at the row's end, which rewrites the columns it
 * shares with the last whole vector with the same values.
 *
 * Rows in a tile need not start aligned for a vector access. A row or tail that does is loaded
 * and stored aligned. One that does not is loaded unaligned and stored with store_v_any, which
 * writes only the 32-byte blocks holding the vector (aie::store_unaligned_v would also rewrite
 * the block after a vector starting on one: past the end of the tile on its last row, while the
 * DMA may be filling the next buffer).
 *
 * @param[in]  in     Input tile of rows x N float elements.
 * @param[out] out    Output tile of rows x N float elements.
 * @param[in]  N      Row length (nc).
 * @param[in]  nr     Rows per z-slice (ne1).
 * @param[in]  n_past Past tokens kept unmasked on every row.
 * @param[in]  row0   Global index of the tile's first row.
 * @param[in]  rows   Rows in the tile.
 */
void ggml_op_diag_mask_inf(const float * __restrict in,
                           float * __restrict out,
                           int32_t N,
                           int32_t nr,
                           int32_t n_past,
                           int32_t row0,
                           int32_t rows) {
    event0();

    constexpr int32_t V = 16;
    // Alignment of a vector load/store (32 B on aie2, 64 B on aie2p).
    constexpr uintptr_t vector_align_mask = aie::vector_ldst_align_v<float, V> - 1;
    const int32_t vend = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V - 1});
    alignas(aie::vector_decl_align) static constexpr int32_t lane_index[V] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const aie::vector<int32_t, V> lanes = aie::load_v<V>(lane_index);
    const aie::vector<float, V> neg_inf =
        aie::broadcast<float, V>(-std::numeric_limits<float>::infinity());

    // How a vector starting at in + i / out + i may be accessed: aligned, or unaligned loads
    // with store_v_any at out + i's alignment.
    enum class access { aligned, unaligned_aligned_store, unaligned };
    const auto access_at = [&](int32_t i) {
        const auto ip = reinterpret_cast<uintptr_t>(in + i);
        const auto op = reinterpret_cast<uintptr_t>(out + i);
        if (((ip | op) & vector_align_mask) == 0) {
            return access::aligned;
        }
        return store_v_any_aligned<V>(out + i) ? access::unaligned_aligned_store
                                               : access::unaligned;
    };
    using aligned_store = std::integral_constant<store_alignment, store_alignment::aligned>;
    using unaligned_store = std::integral_constant<store_alignment, store_alignment::unaligned>;

    int32_t j = row0 % nr;
    for (int32_t r = 0; r < rows; ++r) {
        // Columns [0, keep) are copied from the input; [keep, N) are masked to -inf.
        int32_t keep = n_past + j + 1;
        if (keep < 0) {
            keep = 0;
        }
        if (keep > N) {
            keep = N;
        }
        if (vend == 0) { // narrower than a vector
            for (int32_t i = 0; i < N; ++i) {
                out[i] = i < keep ? in[i] : -std::numeric_limits<float>::infinity();
            }
        } else {
            const auto row_loop = [&](auto aligned, auto store) {
                const float * __restrict pi = in;
                float * __restrict po = out;
                aie::vector<int32_t, V> cols = lanes;
                const auto body = [&]() {
                    aie::vector<float, V> v;
                    if constexpr (decltype(aligned)::value) {
                        v = aie::load_v<V>(pi);
                    } else {
                        v = aie::load_unaligned_v<V>(pi);
                    }
                    const aie::vector<float, V> res = aie::select(v, neg_inf, aie::ge(cols, keep));
                    if constexpr (decltype(aligned)::value) {
                        aie::store_v(po, res);
                    } else {
                        store_v_any<decltype(store)::value>(po, res);
                    }
                    pi += V;
                    po += V;
                    cols = aie::add(cols, V);
                };
                GGML_AIE_LOOP_MIN_TRIPS(4, i, vend, V, body();)
            };
            const access row_access = access_at(0);
            if (row_access == access::aligned) {
                row_loop(std::true_type{}, aligned_store{});
            } else if (row_access == access::unaligned_aligned_store) {
                row_loop(std::false_type{}, aligned_store{});
            } else {
                row_loop(std::false_type{}, unaligned_store{});
            }

            const int32_t t = N - V; // the tail vector
            if (vend != N) {
                const auto mask = aie::ge(aie::add(lanes, t), keep);
                if (access_at(t) == access::aligned) {
                    aie::store_v(out + t, aie::select(aie::load_v<V>(in + t), neg_inf, mask));
                } else {
                    store_v_any(out + t,
                                aie::select(aie::load_unaligned_v<V>(in + t), neg_inf, mask));
                }
            }
        }

        in += N;
        out += N;
        if (++j == nr) {
            j = 0;
        }
    }

    event1();
}

} // extern "C"
