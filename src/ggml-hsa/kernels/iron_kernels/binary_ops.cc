// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file binary_ops.cc
 * @brief Element-wise binary operations for AIE kernels.
 */

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

/**
 * @brief out[i] = op(in0[i], in1[i]) for count elements.
 * @param[in]  in0   First input array of count elements.
 * @param[in]  in1   Second input array of count elements.
 * @param[in]  count Number of elements to process.
 * @param[out] out   Output array of count elements.
 * @param[in]  op    Binary operation to apply.
 */
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

/**
 * @brief Applies a binary operation with NumPy-style broadcasting.
 *
 * Handles broadcasting of src1 (in1) to match the shape of src0/dst (in0/out).
 * Tiles are processed sequentially; the global element index is computed from
 * tile_idx and tile_size to determine the appropriate src1 index via modulo.
 *
 * Scalar: each element's global index is decomposed into 4D dst coordinates and
 * reduced modulo the src1 shape. The row-bias case is split into the dedicated,
 * vectorized ggml_op_add_bias below because a runtime-dimension modulo/divide here
 * lowers to a __divsi3 call per element, which is too costly for that hot path.
 *
 * @param[in]  in0       First input tile (tile_size elements, contiguous from src0).
 * @param[in]  in1       Second input array (full broadcasted tensor).
 * @param[out] out       Output tile (tile_size elements).
 * @param[in]  tile_size Number of elements in this tile.
 * @param[in]  tile_idx  Index of the current tile (0-based).
 * @param[in]  src1_ne0  src1 dimension 0 (innermost).
 * @param[in]  src1_ne1  src1 dimension 1.
 * @param[in]  src1_ne2  src1 dimension 2.
 * @param[in]  src1_ne3  src1 dimension 3 (outermost).
 * @param[in]  dst_ne0   dst dimension 0 (innermost).
 * @param[in]  dst_ne1   dst dimension 1.
 * @param[in]  dst_ne2   dst dimension 2.
 * @param[in]  op        Binary operation to apply.
 */
template <typename T0, typename T1, typename TOut, typename Size, typename BinaryOp>
void transform_binary_broadcast_n(const T0 * __restrict in0,
                                  const T1 * __restrict in1,
                                  TOut * __restrict out,
                                  Size tile_size,
                                  Size tile_idx,
                                  Size src1_ne0,
                                  Size src1_ne1,
                                  Size src1_ne2,
                                  Size src1_ne3,
                                  Size dst_ne0,
                                  Size dst_ne1,
                                  Size dst_ne2,
                                  BinaryOp op) {
    event0();

    auto global_offset = tile_idx * tile_size;

    // src1 strides (contiguous layout)
    auto s1 = src1_ne0;
    auto s2 = src1_ne0 * src1_ne1;
    auto s3 = src1_ne0 * src1_ne1 * src1_ne2;

    // dst strides for coordinate decomposition
    auto d1 = dst_ne0;
    auto d2 = dst_ne0 * dst_ne1;

    for (auto i = 0; i < tile_size; ++i) {
        auto g = global_offset + i;

        // Decompose into 4D dst coordinates
        auto i0 = g % dst_ne0;
        auto i1 = (g / d1) % dst_ne1;
        auto i2 = (g / d2) % dst_ne2;
        auto i3 = g / (d2 * dst_ne2);

        // Apply broadcast modulo
        auto j0 = i0 % src1_ne0;
        auto j1 = i1 % src1_ne1;
        auto j2 = i2 % src1_ne2;
        auto j3 = i3 % src1_ne3;

        // src1 index
        auto idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;

        out[i] = op(in0[i], in1[idx_src1]);
    }

    event1();
}

/**
 * @brief Applies a binary operation between one dst row and a single src1 row that is
 * reused across every dst row.
 *
 * Row-tiled counterpart of @c transform_binary_broadcast_n for the case where src1 is a
 * single row (src1_ne0 == dst_ne0, all higher dims 1). Because the tile is exactly one
 * dst row, the src1 index equals the src0 index, so none of that function's coordinate
 * decomposition or broadcast modulo is needed: seven runtime divisions per element (each
 * a __divsi3 call) collapse to none, and the body vectorizes.
 *
 * @tparam T0        Element type of the first input.
 * @tparam T1        Element type of the second input.
 * @tparam TOut      Element type of the output.
 * @tparam VecOp     Callable applied to a pair of vectors.
 * @tparam ScalarOp  Callable applied to a pair of elements, for the tail.
 *
 * @param[in]  src0      First input row of N elements.
 * @param[in]  src1      Reused row of N elements.
 * @param[out] out       Output row of N elements.
 * @param[in]  N         Elements per row (== ne0).
 * @param[in]  vec_op    Vector operation to apply.
 * @param[in]  scalar_op Scalar operation to apply to the tail.
 */
template <typename T0, typename T1, typename TOut, typename VecOp, typename ScalarOp>
void transform_binary_row_n(const T0 * __restrict src0,
                            const T1 * __restrict src1,
                            TOut * __restrict out,
                            int32_t N,
                            VecOp vec_op,
                            ScalarOp scalar_op) {
    static_assert(std::is_same_v<T0, T1> && std::is_same_v<T0, TOut>,
                  "the vector body operates on the operands directly, with no per-element "
                  "cast, so all three types must match");

    event0();

    constexpr int32_t V = 512 / (sizeof(TOut) * 8);
    const int32_t vend = (N / V) * V; // division by constexpr V → inline shift, once

    // Unaligned loads/stores: the IRON design streams rows through double-buffered
    // fifos whose per-row object stride (N elements) need not be vector-aligned, so
    // aligned load_v/store_v would corrupt alternate (ping-pong) rows.
    // No AIE_LOOP_MIN_ITERATION_COUNT: when N < V (e.g. a 10-wide row) vend is 0,
    // and promising >=1 iteration makes the pipelined prologue run the body on too few
    // elements. AIE_PREPARE_FOR_PIPELINING alone suffices for the N >> V rows.
    AIE_PREPARE_FOR_PIPELINING
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<T0, V> a = aie::load_unaligned_v<V>(src0 + i);
        aie::vector<T1, V> b = aie::load_unaligned_v<V>(src1 + i);
        aie::store_unaligned_v(out + i, vec_op(a, b));
    }

    for (int32_t i = vend; i < N; ++i) {
        out[i] = scalar_op(src0[i], src1[i]);
    }

    event1();
}

extern "C" {

#ifdef GGML_OP_ADD

/**
 * @brief Element-wise addition: out[i] = in0[i] + in1[i].
 *
 * @param[in]  in0 First input array of N elements.
 * @param[in]  in1 Second input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_add(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(a + b);
    });
}

#endif // GGML_OP_ADD

#ifdef GGML_OP_SUB

/**
 * @brief Element-wise subtraction: out[i] = in0[i] - in1[i].
 *
 * @param[in]  in0 First input array of N elements.
 * @param[in]  in1 Second input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_sub(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(a - b);
    });
}

#endif // GGML_OP_SUB

#ifdef GGML_OP_MUL

/**
 * @brief Element-wise multiplication: out[i] = in0[i] * in1[i].
 *
 * @param[in]  in0 First input array of N elements.
 * @param[in]  in1 Second input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_mul(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(a * b);
    });
}

#endif // GGML_OP_MUL

#ifdef GGML_OP_DIV

/**
 * @brief Element-wise division: out[i] = in0[i] / in1[i].
 *
 * @param[in]  in0 First input array of N elements (dividend).
 * @param[in]  in1 Second input array of N elements (divisor).
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_div(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(a / b);
    });
}

#endif // GGML_OP_DIV

#ifdef GGML_OP_ADD_BROADCAST

/**
 * @brief Addition with broadcasting: out[i] = in0[i] + in1[broadcast_idx].
 *
 * Broadcasts in1 to match in0's shape using NumPy-style broadcasting rules.
 *
 * @param[in]  in0       First input tile (tile_size elements).
 * @param[in]  in1       Second input array (broadcasted, may be smaller).
 * @param[out] out       Output tile (tile_size elements).
 * @param[in]  tile_size Number of elements in this tile.
 * @param[in]  tile_idx  Index of the current tile (0-based).
 * @param[in]  src1_ne0  src1 dimension 0.
 * @param[in]  src1_ne1  src1 dimension 1.
 * @param[in]  src1_ne2  src1 dimension 2.
 * @param[in]  src1_ne3  src1 dimension 3.
 * @param[in]  dst_ne0   dst dimension 0.
 * @param[in]  dst_ne1   dst dimension 1.
 * @param[in]  dst_ne2   dst dimension 2.
 */
void ggml_op_add_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t src1_ne0,
                           int32_t src1_ne1,
                           int32_t src1_ne2,
                           int32_t src1_ne3,
                           int32_t dst_ne0,
                           int32_t dst_ne1,
                           int32_t dst_ne2) {
    transform_binary_broadcast_n(
        in0, in1, out, tile_size, tile_idx, src1_ne0, src1_ne1, src1_ne2, src1_ne3, dst_ne0,
        dst_ne1, dst_ne2,
        [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a + b); });
}

#endif // GGML_OP_ADD_BROADCAST

#ifdef GGML_OP_SUB_BROADCAST

/**
 * @brief Subtraction with broadcasting: out[i] = in0[i] - in1[broadcast_idx].
 *
 * Broadcasts in1 to match in0's shape using NumPy-style broadcasting rules.
 *
 * @param[in]  in0       First input tile (tile_size elements).
 * @param[in]  in1       Second input array (broadcasted, may be smaller).
 * @param[out] out       Output tile (tile_size elements).
 * @param[in]  tile_size Number of elements in this tile.
 * @param[in]  tile_idx  Index of the current tile (0-based).
 * @param[in]  src1_ne0  src1 dimension 0.
 * @param[in]  src1_ne1  src1 dimension 1.
 * @param[in]  src1_ne2  src1 dimension 2.
 * @param[in]  src1_ne3  src1 dimension 3.
 * @param[in]  dst_ne0   dst dimension 0.
 * @param[in]  dst_ne1   dst dimension 1.
 * @param[in]  dst_ne2   dst dimension 2.
 */
void ggml_op_sub_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t src1_ne0,
                           int32_t src1_ne1,
                           int32_t src1_ne2,
                           int32_t src1_ne3,
                           int32_t dst_ne0,
                           int32_t dst_ne1,
                           int32_t dst_ne2) {
    transform_binary_broadcast_n(
        in0, in1, out, tile_size, tile_idx, src1_ne0, src1_ne1, src1_ne2, src1_ne3, dst_ne0,
        dst_ne1, dst_ne2,
        [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a - b); });
}

#endif // GGML_OP_SUB_BROADCAST

#ifdef GGML_OP_MUL_BROADCAST

/**
 * @brief Multiplication with broadcasting: out[i] = in0[i] * in1[broadcast_idx].
 *
 * Broadcasts in1 to match in0's shape using NumPy-style broadcasting rules.
 *
 * @param[in]  in0       First input tile (tile_size elements).
 * @param[in]  in1       Second input array (broadcasted, may be smaller).
 * @param[out] out       Output tile (tile_size elements).
 * @param[in]  tile_size Number of elements in this tile.
 * @param[in]  tile_idx  Index of the current tile (0-based).
 * @param[in]  src1_ne0  src1 dimension 0.
 * @param[in]  src1_ne1  src1 dimension 1.
 * @param[in]  src1_ne2  src1 dimension 2.
 * @param[in]  src1_ne3  src1 dimension 3.
 * @param[in]  dst_ne0   dst dimension 0.
 * @param[in]  dst_ne1   dst dimension 1.
 * @param[in]  dst_ne2   dst dimension 2.
 */
void ggml_op_mul_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t src1_ne0,
                           int32_t src1_ne1,
                           int32_t src1_ne2,
                           int32_t src1_ne3,
                           int32_t dst_ne0,
                           int32_t dst_ne1,
                           int32_t dst_ne2) {
    transform_binary_broadcast_n(
        in0, in1, out, tile_size, tile_idx, src1_ne0, src1_ne1, src1_ne2, src1_ne3, dst_ne0,
        dst_ne1, dst_ne2,
        [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a * b); });
}

#endif // GGML_OP_MUL_BROADCAST

#ifdef GGML_OP_DIV_BROADCAST

/**
 * @brief Division with broadcasting: out[i] = in0[i] / in1[broadcast_idx].
 *
 * Broadcasts in1 to match in0's shape using NumPy-style broadcasting rules.
 *
 * @param[in]  in0       First input tile (dividend, tile_size elements).
 * @param[in]  in1       Second input array (divisor, broadcasted).
 * @param[out] out       Output tile (tile_size elements).
 * @param[in]  tile_size Number of elements in this tile.
 * @param[in]  tile_idx  Index of the current tile (0-based).
 * @param[in]  src1_ne0  src1 dimension 0.
 * @param[in]  src1_ne1  src1 dimension 1.
 * @param[in]  src1_ne2  src1 dimension 2.
 * @param[in]  src1_ne3  src1 dimension 3.
 * @param[in]  dst_ne0   dst dimension 0.
 * @param[in]  dst_ne1   dst dimension 1.
 * @param[in]  dst_ne2   dst dimension 2.
 */
void ggml_op_div_broadcast(const INPUT0_DTYPE * __restrict in0,
                           const INPUT1_DTYPE * __restrict in1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t tile_size,
                           int32_t tile_idx,
                           int32_t src1_ne0,
                           int32_t src1_ne1,
                           int32_t src1_ne2,
                           int32_t src1_ne3,
                           int32_t dst_ne0,
                           int32_t dst_ne1,
                           int32_t dst_ne2) {
    transform_binary_broadcast_n(
        in0, in1, out, tile_size, tile_idx, src1_ne0, src1_ne1, src1_ne2, src1_ne3, dst_ne0,
        dst_ne1, dst_ne2,
        [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a / b); });
}

#endif // GGML_OP_DIV_BROADCAST

// Row-broadcast fast paths: src1 is a single row (ne0 elements) reused across every dst
// row. The Python dispatch gates all four on matching input/output types (see
// transform_binary_row_n) and falls back to the generic broadcast kernels otherwise.

#ifdef GGML_OP_ADD_ROW

/**
 * @brief out[i] = src0[i] + src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * @param[in]  src0 First input row of N elements.
 * @param[in]  src1 Reused row of N elements.
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 */
void ggml_op_add_row(const INPUT0_DTYPE * __restrict src0,
                     const INPUT1_DTYPE * __restrict src1,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N) {
    transform_binary_row_n(
        src0, src1, out, N, [](auto a, auto b) { return aie::add(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a + b); });
}

#endif // GGML_OP_ADD_ROW

#ifdef GGML_OP_SUB_ROW

/**
 * @brief out[i] = src0[i] - src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * @param[in]  src0 First input row of N elements.
 * @param[in]  src1 Reused row of N elements.
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 */
void ggml_op_sub_row(const INPUT0_DTYPE * __restrict src0,
                     const INPUT1_DTYPE * __restrict src1,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N) {
    transform_binary_row_n(
        src0, src1, out, N, [](auto a, auto b) { return aie::sub(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a - b); });
}

#endif // GGML_OP_SUB_ROW

#ifdef GGML_OP_MUL_ROW

/**
 * @brief out[i] = src0[i] * src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * aie::mul yields an accumulator, so the vector result is narrowed back to the operand
 * type before the store (same shape as the SCALE kernel).
 *
 * @param[in]  src0 First input row of N elements.
 * @param[in]  src1 Reused row of N elements.
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 */
void ggml_op_mul_row(const INPUT0_DTYPE * __restrict src0,
                     const INPUT1_DTYPE * __restrict src1,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N) {
    transform_binary_row_n(
        src0, src1, out, N,
        [](auto a, auto b) { return aie::mul(a, b).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a * b); });
}

#endif // GGML_OP_MUL_ROW

#ifdef GGML_OP_DIV_ROW

/**
 * @brief out[i] = src0[i] / src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * Scalar body, unlike the other three: there is no float vector divide instruction, and
 * aie::div is a multiply by aie::inv (an approximate reciprocal), so a vectorized body
 * would not match the CPU reference the device tests compare against element for element.
 * Row-tiling still removes the seven per-element broadcast divisions, leaving only the one
 * division the operation requires. If a looser tolerance is ever acceptable here, the
 * aie::div path is the obvious next step.
 *
 * @param[in]  src0 Dividend row of N elements.
 * @param[in]  src1 Reused divisor row of N elements.
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 */
void ggml_op_div_row(const INPUT0_DTYPE * __restrict src0,
                     const INPUT1_DTYPE * __restrict src1,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N) {
    transform_binary_n(src0, src1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(a / b);
    });
}

#endif // GGML_OP_DIV_ROW

} // extern "C"
