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
                           int32_t ne0_src1,
                           int32_t ne1_src1,
                           int32_t ne2_src1,
                           int32_t ne3_src1,
                           int32_t ne0_dst,
                           int32_t ne1_dst,
                           int32_t ne2_dst) {
    event0();

    int32_t global_offset = tile_idx * tile_size;

    // src1 strides (contiguous layout)
    int32_t s1 = ne0_src1;
    int32_t s2 = ne0_src1 * ne1_src1;
    int32_t s3 = ne0_src1 * ne1_src1 * ne2_src1;

    // dst strides for coordinate decomposition
    int32_t d1 = ne0_dst;
    int32_t d2 = ne0_dst * ne1_dst;

    for (int32_t i = 0; i < tile_size; ++i) {
        int32_t g = global_offset + i;

        // Decompose into 4D dst coordinates
        int32_t i0 = g % ne0_dst;
        int32_t i1 = (g / d1) % ne1_dst;
        int32_t i2 = (g / d2) % ne2_dst;
        int32_t i3 = g / (d2 * ne2_dst);

        // Apply broadcast modulo
        int32_t j0 = i0 % ne0_src1;
        int32_t j1 = i1 % ne1_src1;
        int32_t j2 = i2 % ne2_src1;
        int32_t j3 = i3 % ne3_src1;

        // src1 index
        int32_t idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;

        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) + static_cast<float>(in1[idx_src1])
        );
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
                           int32_t ne0_src1,
                           int32_t ne1_src1,
                           int32_t ne2_src1,
                           int32_t ne3_src1,
                           int32_t ne0_dst,
                           int32_t ne1_dst,
                           int32_t ne2_dst) {
    event0();

    int32_t global_offset = tile_idx * tile_size;

    // src1 strides (contiguous layout)
    int32_t s1 = ne0_src1;
    int32_t s2 = ne0_src1 * ne1_src1;
    int32_t s3 = ne0_src1 * ne1_src1 * ne2_src1;

    // dst strides for coordinate decomposition
    int32_t d1 = ne0_dst;
    int32_t d2 = ne0_dst * ne1_dst;

    for (int32_t i = 0; i < tile_size; ++i) {
        int32_t g = global_offset + i;

        // Decompose into 4D dst coordinates
        int32_t i0 = g % ne0_dst;
        int32_t i1 = (g / d1) % ne1_dst;
        int32_t i2 = (g / d2) % ne2_dst;
        int32_t i3 = g / (d2 * ne2_dst);

        // Apply broadcast modulo
        int32_t j0 = i0 % ne0_src1;
        int32_t j1 = i1 % ne1_src1;
        int32_t j2 = i2 % ne2_src1;
        int32_t j3 = i3 % ne3_src1;

        // src1 index
        int32_t idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;

        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) - static_cast<float>(in1[idx_src1])
        );
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
                           int32_t ne0_src1,
                           int32_t ne1_src1,
                           int32_t ne2_src1,
                           int32_t ne3_src1,
                           int32_t ne0_dst,
                           int32_t ne1_dst,
                           int32_t ne2_dst) {
    event0();

    int32_t global_offset = tile_idx * tile_size;

    // src1 strides (contiguous layout)
    int32_t s1 = ne0_src1;
    int32_t s2 = ne0_src1 * ne1_src1;
    int32_t s3 = ne0_src1 * ne1_src1 * ne2_src1;

    // dst strides for coordinate decomposition
    int32_t d1 = ne0_dst;
    int32_t d2 = ne0_dst * ne1_dst;

    for (int32_t i = 0; i < tile_size; ++i) {
        int32_t g = global_offset + i;

        // Decompose into 4D dst coordinates
        int32_t i0 = g % ne0_dst;
        int32_t i1 = (g / d1) % ne1_dst;
        int32_t i2 = (g / d2) % ne2_dst;
        int32_t i3 = g / (d2 * ne2_dst);

        // Apply broadcast modulo
        int32_t j0 = i0 % ne0_src1;
        int32_t j1 = i1 % ne1_src1;
        int32_t j2 = i2 % ne2_src1;
        int32_t j3 = i3 % ne3_src1;

        // src1 index
        int32_t idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;

        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) * static_cast<float>(in1[idx_src1])
        );
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
                           int32_t ne0_src1,
                           int32_t ne1_src1,
                           int32_t ne2_src1,
                           int32_t ne3_src1,
                           int32_t ne0_dst,
                           int32_t ne1_dst,
                           int32_t ne2_dst) {
    event0();

    int32_t global_offset = tile_idx * tile_size;

    // src1 strides (contiguous layout)
    int32_t s1 = ne0_src1;
    int32_t s2 = ne0_src1 * ne1_src1;
    int32_t s3 = ne0_src1 * ne1_src1 * ne2_src1;

    // dst strides for coordinate decomposition
    int32_t d1 = ne0_dst;
    int32_t d2 = ne0_dst * ne1_dst;

    for (int32_t i = 0; i < tile_size; ++i) {
        int32_t g = global_offset + i;

        // Decompose into 4D dst coordinates
        int32_t i0 = g % ne0_dst;
        int32_t i1 = (g / d1) % ne1_dst;
        int32_t i2 = (g / d2) % ne2_dst;
        int32_t i3 = g / (d2 * ne2_dst);

        // Apply broadcast modulo
        int32_t j0 = i0 % ne0_src1;
        int32_t j1 = i1 % ne1_src1;
        int32_t j2 = i2 % ne2_src1;
        int32_t j3 = i3 % ne3_src1;

        // src1 index
        int32_t idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;

        out[i] = static_cast<OUTPUT_DTYPE>(
            static_cast<float>(in0[i]) / static_cast<float>(in1[idx_src1])
        );
    }

    event1();
}

#endif // GGML_OP_DIV_BROADCAST

} // extern "C"
