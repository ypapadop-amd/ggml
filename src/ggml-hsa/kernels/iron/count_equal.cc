// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstring>

#include "ggml-aie.hpp"

extern "C" {

/**
 * Count equal operation: counts elements that are equal between two tensors.
 *
 * Processes data in tiles. On the first tile (tile_idx == 0), initializes the
 * output buffer to 0. Each tile reads the accumulated count from the output
 * buffer, adds its local count, and writes back.
 *
 * The output buffer is passed as int32_t[2] due to IRON not supporting i64
 * in ObjectFifos, but we access it as int64_t through casting.
 *
 * @param in0 First input tile
 * @param in1 Second input tile
 * @param out Output buffer (2 x int32 = 1 x int64), used as accumulator
 * @param tile_size Number of elements in this tile
 * @param tile_idx Current tile index (0-based)
 */
void ggml_op_count_equal(const INPUT_DTYPE * __restrict in0,
                         const INPUT_DTYPE * __restrict in1,
                         int32_t * __restrict out, // Actually int64_t, cast due to IRON limitations
                         int32_t tile_size,
                         int32_t tile_idx) {
    event0();

    // Initialize accumulator on first tile
    if (tile_idx == 0) {
        out[0] = 0;
        out[1] = 0;
    }

    // Count equal elements using vectorized comparison where possible
    constexpr int VEC_SIZE = 16; // 16 x int32 = 512 bits
    const int num_full_iters = tile_size / VEC_SIZE;
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
    for (int i = tail_start; i < tile_size; i++) {
        if (in0[i] == in1[i]) {
            local_count++;
        }
    }

    // Accumulate into output buffer
    int64_t out64 = 0;
    std::memcpy(&out64, out, sizeof(int64_t)); // Read current count (as int64_t)
    out64 += local_count;                      // Add local count
    std::memcpy(out, &out64, sizeof(int64_t)); // Write back

    event1();
}

} // extern "C"
