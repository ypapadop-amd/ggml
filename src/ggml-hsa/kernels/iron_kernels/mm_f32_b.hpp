// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

/**
 * @file mm_f32_b.hpp
 * @brief Core-side helpers for a GEMM that streams an f32 B and converts it to bf16 on the core,
 * shared by aie2/mm.cc and aie2p/mm.cc.
 */

#include <cstdint>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

/**
 * @brief Converts the streamed f32 B tile to bf16.
 *
 * The conversion is the bit-exact round-to-nearest-even used by the host and the CONVERT_PAD
 * kernel, so the product is identical to converting B beforehand. It is a flat element-wise pass:
 * the DMA has already applied the mmul blocking to the f32 tile, and the conversion preserves
 * element order.
 *
 * @tparam K Tile K dimension.
 * @tparam N Tile N dimension.
 *
 * @param[in]  b_in      B tile (f32, K x N).
 * @param[out] b_scratch B tile converted to bf16 (K x N).
 */
template <std::int32_t K, std::int32_t N>
inline void convert_b_tile(const float * __restrict b_in, bfloat16 * __restrict b_scratch) {
    constexpr std::int32_t V = 512 / (sizeof(float) * 8);
    constexpr std::int32_t nblk = (K * N) / V;
    static_assert((K * N) % V == 0, "B tile must be a whole number of vectors");

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_RANGE(nblk, nblk)
    for (std::int32_t b = 0; b < nblk; ++b) {
        const aie::vector<float, V> fv = aie::load_v<V>(b_in + b * V);
        aie::store_v(b_scratch + b * V, convert_f32_to_bf16_vector<V>(fv));
    }
}

/**
 * @brief Zeroes the converted B tile's elements whose K index is @p k_valid or more.
 *
 * On the last K tile of a GEMM whose B holds fewer than K elements per column, the DMA reads past
 * each column's end (into the next column, or the allocation's slack). Those elements must not
 * contribute, and zeroing them -- rather than relying on A's zero padding -- also keeps a NaN or
 * inf read from the next column out of this one.
 *
 * gemm.py streams column-major B as (n/t, t*k), (k/s, s), (t, k), (s, 1), so the tile holds
 * [n/t][k/s][t][s] blocks and an element's K index within the tile is sb * s + jj.
 *
 * @tparam K       Tile K dimension.
 * @tparam N       Tile N dimension.
 * @tparam s       bf16 mmul K dimension.
 * @tparam t       bf16 mmul N dimension.
 * @tparam k_valid Number of valid elements along K in this tile.
 *
 * @param[in,out] b_scratch Converted B tile (bf16, K x N).
 */
template <std::int32_t K, std::int32_t N, std::int32_t s, std::int32_t t, std::int32_t k_valid>
inline void zero_b_k_tail(bfloat16 * __restrict b_scratch) {
    static_assert(k_valid > 0 && k_valid < K, "the K tail must be a partial tile");

    for (std::int32_t tb = 0; tb < N / t; ++tb) {
        for (std::int32_t sb = k_valid / s; sb < K / s; ++sb) {
            const std::int32_t first = sb == k_valid / s ? k_valid % s : 0;
            for (std::int32_t i = 0; i < t; ++i) {
                bfloat16 * block = b_scratch + ((tb * (K / s) + sb) * t + i) * s;
                for (std::int32_t jj = first; jj < s; ++jj) {
                    block[jj] = static_cast<bfloat16>(0.0f);
                }
            }
        }
    }
}
