// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

extern "C" {

#ifdef GGML_OP_COUNT_EQUAL

/**
 * Count equal operation: counts elements that are equal between two tensors.
 *
 * Processes data in tiles. On the first tile (tile_idx == 0), initializes the
 * output buffer to 0. Each tile reads the accumulated count from the output
 * buffer, adds its local count, and writes back.
 *
 * The output buffer is passed as int32_t[2] due to IRON not supporting i64
 * in ObjectFifos, but we access it as int64_t through pointer casting.
 *
 * @param in0 First input tile
 * @param in1 Second input tile
 * @param out Output buffer (2 x int32 = 1 x int64), used as accumulator
 * @param tile_size Size of full tiles (TILE_SIZE)
 * @param tile_idx Current tile index (0-based)
 * @param num_tiles Total number of tiles
 * @param last_tile_size Size of the last tile (may be smaller than tile_size)
 */
void ggml_op_count_equal(
    const INPUT_DTYPE * __restrict in0,
    const INPUT_DTYPE * __restrict in1,
    int32_t * __restrict out,  // Actually int64_t, cast due to IRON limitations
    int32_t tile_size,
    int32_t tile_idx,
    int32_t num_tiles,
    int32_t last_tile_size
) {
    event0();

    // Cast output buffer to int64_t for accumulation
    int64_t * out64 = reinterpret_cast<int64_t *>(out);

    // Initialize accumulator on first tile
    if (tile_idx == 0) {
        out64[0] = 0;
    }

    // Determine actual size for this tile
    const int32_t actual_size = (tile_idx == num_tiles - 1) ? last_tile_size : tile_size;

    // Count equal elements using vectorized comparison where possible
    constexpr int VEC_SIZE = 16;  // 16 x int32 = 512 bits
    const int num_full_iters = actual_size / VEC_SIZE;
    const int tail_start = num_full_iters * VEC_SIZE;

    int32_t local_count = 0;

    // Vectorized loop
    const INPUT_DTYPE * __restrict p0 = in0;
    const INPUT_DTYPE * __restrict p1 = in1;

    for (int i = 0; i < num_full_iters; i++) {
        aie::vector<INPUT_DTYPE, VEC_SIZE> v0 = aie::load_v<VEC_SIZE>(p0);
        aie::vector<INPUT_DTYPE, VEC_SIZE> v1 = aie::load_v<VEC_SIZE>(p1);
        p0 += VEC_SIZE;
        p1 += VEC_SIZE;

        // Compare vectors - returns mask where elements are equal
        auto mask = aie::eq(v0, v1);

        // Count set bits in mask (number of equal elements)
        for (int j = 0; j < VEC_SIZE; j++) {
            if (mask.test(j)) {
                local_count++;
            }
        }
    }

    // Scalar tail
    for (int i = tail_start; i < actual_size; i++) {
        if (in0[i] == in1[i]) {
            local_count++;
        }
    }

    // Accumulate into output buffer
    out64[0] += local_count;

    event1();
}

#endif // GGML_OP_COUNT_EQUAL

} // extern "C"
