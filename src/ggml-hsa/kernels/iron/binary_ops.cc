// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"

template <typename T0, typename T1, typename TOut, typename Size, typename BinaryOp>
void transform_binary_n(const T0 * __restrict in0,
                        const T1 * __restrict in1,
                        Size count,
                        TOut * __restrict out,
                        BinaryOp op) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in0[i], in1[i]);
    }
    event1();
}

extern "C" {

#ifdef GGML_OP_ADD

void ggml_op_add(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a + b; });
}

#endif // GGML_OP_ADD

#ifdef GGML_OP_SUB

void ggml_op_sub(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a - b; });
}

#endif // GGML_OP_SUB

#ifdef GGML_OP_MUL

void ggml_op_mul(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a * b; });
}

#endif // GGML_OP_MUL

#ifdef GGML_OP_DIV

void ggml_op_div(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a / b; });
}

#endif // GGML_OP_DIV

#ifdef GGML_OP_ADD_BROADCAST

void ggml_op_add_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t ne0_src1) {
    event0();

    // Compute starting offset within src1 for this tile.
    // global_offset = (tile_idx * tile_size) % ne0_src1
    // To avoid 64-bit division, we compute: offset = (tile_idx % (ne0_src1/gcd)) * tile_size % ne0_src1
    // For simplicity, we just track the running index within src1.
    int32_t idx_src1 = (tile_idx * (tile_size % ne0_src1)) % ne0_src1;

    for (int32_t i = 0; i < tile_size; ++i) {
        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) + static_cast<float>(in1[idx_src1])
        );
        // Increment and wrap
        idx_src1++;
        if (idx_src1 >= ne0_src1) {
            idx_src1 = 0;
        }
    }

    event1();
}

#endif // GGML_OP_ADD_BROADCAST

#ifdef GGML_OP_SUB_BROADCAST

void ggml_op_sub_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t ne0_src1) {
    event0();

    int32_t idx_src1 = (tile_idx * (tile_size % ne0_src1)) % ne0_src1;

    for (int32_t i = 0; i < tile_size; ++i) {
        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) - static_cast<float>(in1[idx_src1])
        );
        idx_src1++;
        if (idx_src1 >= ne0_src1) {
            idx_src1 = 0;
        }
    }

    event1();
}

#endif // GGML_OP_SUB_BROADCAST

#ifdef GGML_OP_MUL_BROADCAST

void ggml_op_mul_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t ne0_src1) {
    event0();

    int32_t idx_src1 = (tile_idx * (tile_size % ne0_src1)) % ne0_src1;

    for (int32_t i = 0; i < tile_size; ++i) {
        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) * static_cast<float>(in1[idx_src1])
        );
        idx_src1++;
        if (idx_src1 >= ne0_src1) {
            idx_src1 = 0;
        }
    }

    event1();
}

#endif // GGML_OP_MUL_BROADCAST

#ifdef GGML_OP_DIV_BROADCAST

void ggml_op_div_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t ne0_src1) {
    event0();

    int32_t idx_src1 = (tile_idx * (tile_size % ne0_src1)) % ne0_src1;

    for (int32_t i = 0; i < tile_size; ++i) {
        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) / static_cast<float>(in1[idx_src1])
        );
        idx_src1++;
        if (idx_src1 >= ne0_src1) {
            idx_src1 = 0;
        }
    }

    event1();
}

#endif // GGML_OP_DIV_BROADCAST

} // extern "C"
