// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file binary_ops.cc
 * @brief Element-wise binary operations for AIE kernels.
 */

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"
#include "transform.h"

/**
 * @brief Applies a binary operation with NumPy-style broadcasting.
 *
 * Handles broadcasting of src1 (in1) to match the shape of src0/dst (in0/out).
 * Tiles are processed sequentially; the global element index is computed from
 * tile_idx and tile_size to determine the appropriate src1 index via modulo.
 *
 * Scalar: each element's global index is decomposed into 4D dst coordinates and
 * reduced modulo the src1 shape. A src1 that is a single row reused across every dst
 * row is handled by the vectorized ggml_op_*_row kernels below instead, because a
 * runtime-dimension modulo/divide here lowers to a __divsi3 call per element, which is
 * too costly for that hot path.
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

#if defined(GGML_OP_ADD_ROWS) || defined(GGML_OP_SUB_ROWS) || defined(GGML_OP_MUL_ROWS) ||         \
    defined(GGML_OP_DIV_ROWS) || defined(GGML_OP_ADD_SCALAR_ROWS) ||                               \
    defined(GGML_OP_SUB_SCALAR_ROWS) || defined(GGML_OP_MUL_SCALAR_ROWS) ||                        \
    defined(GGML_OP_DIV_SCALAR_ROWS)

/**
 * @brief The src1 row that dst row @p row reads, for a src1 of several rows.
 *
 * GGML repeats src1 along every dimension it is shorter in: dst row (i1, i2, i3) reads src1 row
 * (i1 % ne1, i2 % ne2, i3 % ne3). The shapes are compile-time (-DGGML_BCAST_*), so the divisions
 * are by constants; this runs once per row, not per element.
 *
 * @param[in] row dst row index (over dims 1..3).
 * @return src1 row index.
 */
inline int32_t bcast_src1_row(int32_t row) {
    const auto r = static_cast<uint32_t>(row);
    const uint32_t i1 = r % GGML_BCAST_DST_NE1;
    const uint32_t i2 = (r / GGML_BCAST_DST_NE1) % GGML_BCAST_DST_NE2;
    const uint32_t i3 = r / (GGML_BCAST_DST_NE1 * GGML_BCAST_DST_NE2);
    return static_cast<int32_t>(
        i1 % GGML_BCAST_SRC1_NE1 +
        GGML_BCAST_SRC1_NE1 *
            (i2 % GGML_BCAST_SRC1_NE2 + GGML_BCAST_SRC1_NE2 * (i3 % GGML_BCAST_SRC1_NE3)));
}

#endif

#if defined(GGML_OP_ADD_SCALAR_ROWS) || defined(GGML_OP_SUB_SCALAR_ROWS) ||                        \
    defined(GGML_OP_MUL_SCALAR_ROWS) || defined(GGML_OP_DIV_SCALAR_ROWS)

/**
 * @brief out = op(src0, s) over @p rows dst rows of @p N elements, s the one src1 element each
 * row reads.
 *
 * For a src1 that is a single element along the leading dims it repeats in (a conv bias
 * [1, 1, C]): the design folds those dims into rows of N elements, so every row reads one src1
 * element, bcast_src1_row(row) of src1 viewed as rows of one element.
 *
 * @param[in]  src0     The tile's rows of src0, @p rows * @p N elements.
 * @param[in]  src1     All of src1.
 * @param[out] out      The tile's rows of dst.
 * @param[in]  N        Elements per row.
 * @param[in]  rows     Rows in the tile.
 * @param[in]  row0     dst row index of the tile's first row.
 * @param[in]  vec_op   op(vector, scalar), as transform_vector_walk_n's vec_op.
 * @param[in]  scalar_op op(element, scalar).
 */
template <typename VecOp, typename ScalarOp>
void transform_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                           const INPUT1_DTYPE * __restrict src1,
                           OUTPUT_DTYPE * __restrict out,
                           int32_t N,
                           int32_t rows,
                           int32_t row0,
                           VecOp vec_op,
                           ScalarOp scalar_op) {
    static_assert(std::is_same_v<INPUT1_DTYPE, OUTPUT_DTYPE>,
                  "the vector op takes src1's element as is: the design gates this on one dtype");
    for (int32_t r = 0; r < rows; ++r) {
        const auto s = src1[bcast_src1_row(row0 + r)];
        transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
            out + r * N, N, [s, vec_op](auto a) { return vec_op(a, s); },
            [s, scalar_op](auto a) { return scalar_op(a, s); }, src0 + r * N);
    }
}

#endif

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
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::add(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a + b); }, in0, in1);
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
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::sub(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a - b); }, in0, in1);
}

#endif // GGML_OP_SUB

#ifdef GGML_OP_MUL

/**
 * @brief Element-wise multiplication: out[i] = in0[i] * in1[i].
 *
 * Same accuracy caveat as ggml_op_mul_row: on aie2 (NPU1) there is no native fp32 multiplier,
 * so the vector body's aie::mul lowers to Peano's bf16 triple-product emulation and is not
 * bit-identical to the scalar `a * b` the tail uses -- about one ulp of the operand scale. The
 * scalar path (integers, and any type mismatch) stays exact.
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
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::mul(a, b).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a * b); }, in0, in1);
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
    transform_n(
        out, N, [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a / b); },
        in0, in1);
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
// the shared transform_vector_walk_n) and falls back to the generic broadcast kernels otherwise.

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
    transform_vector_walk_n(
        out, N, [](auto a, auto b) { return aie::add(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a + b); }, src0, src1);
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
    transform_vector_walk_n(
        out, N, [](auto a, auto b) { return aie::sub(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a - b); }, src0, src1);
}

#endif // GGML_OP_SUB_ROW

#ifdef GGML_OP_MUL_ROW

/**
 * @brief out[i] = src0[i] * src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * aie::mul yields an accumulator, so the vector result is narrowed back to the operand
 * type before the store (same shape as the SCALE kernel).
 *
 * On aie2 (NPU1) there is no native fp32 multiplier: aie::mul lowers to ::mul_elem_16_conf,
 * which Peano defines in aiev2/aiev2_core.h (clang 22 and later) as a bf16 triple-product
 * emulation, so the vector path is not bit-identical to the scalar `a * b` the tail uses. Measured
 * against the CPU it costs one ulp of the operand scale (test-broadcast-hsa reports it), which is
 * the same order as the ADD and SUB row kernels above -- those are within one ulp too, on both
 * their vector and scalar paths. Only DIV below, and this kernel's own scalar tail, come out exact.
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
    transform_vector_walk_n(
        out, N, [](auto a, auto b) { return aie::mul(a, b).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a * b); }, src0, src1);
}

#endif // GGML_OP_MUL_ROW

#ifdef GGML_OP_DIV_ROW

/**
 * @brief out[i] = src0[i] / src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * Scalar body, unlike the other three: there is no float vector divide instruction, and
 * aie::div is a multiply by aie::inv, an approximate reciprocal whose error is orders of
 * magnitude larger than the sub-ulp drift the vectorized ADD/SUB/MUL kernels above accept.
 * That gap, not bit-exactness, is the reason to stay scalar: none of the fp32 kernels here
 * is bit-identical to the CPU (aie2 has no native fp32 ALU), they are simply all within one
 * ulp, and aie::div would not be. Row-tiling still removes the seven per-element broadcast
 * divisions, leaving only the one division the operation requires. If a tolerance that
 * loose is ever acceptable, the aie::div path is the obvious next step.
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
    transform_n(
        out, N, [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a / b); },
        src0, src1);
}

#endif // GGML_OP_DIV_ROW

// Row-tiled broadcast of a src1 with several rows of ne0 elements (each dst row reads the src1
// row bcast_src1_row gives). Unlike the _ROW kernels above, src1 may differ in type: the walk
// then runs its scalar path. The vector body is limited to floating point, as in the
// element-wise kernels, since aie::mul saturates integers where `a * b` wraps.

#ifdef GGML_OP_ADD_ROWS

/**
 * @brief out[i] = src0[i] + src1[src1 row of this dst row][i], for one dst row.
 *
 * @param[in]  src0 dst row's src0 row of N elements.
 * @param[in]  src1 All of src1 (rows of N elements).
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 * @param[in]  row0 First dst row of this worker's band.
 * @param[in]  i    Row within the band.
 */
void ggml_op_add_rows(const INPUT0_DTYPE * __restrict src0,
                      const INPUT1_DTYPE * __restrict src1,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t row0,
                      int32_t i) {
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::add(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a + b); }, src0,
        src1 + bcast_src1_row(row0 + i) * N);
}

#endif // GGML_OP_ADD_ROWS

#ifdef GGML_OP_SUB_ROWS

/// As ggml_op_add_rows, for out[i] = src0[i] - src1 row[i].
void ggml_op_sub_rows(const INPUT0_DTYPE * __restrict src0,
                      const INPUT1_DTYPE * __restrict src1,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t row0,
                      int32_t i) {
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::sub(a, b); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a - b); }, src0,
        src1 + bcast_src1_row(row0 + i) * N);
}

#endif // GGML_OP_SUB_ROWS

#ifdef GGML_OP_MUL_ROWS

/// As ggml_op_add_rows, for out[i] = src0[i] * src1 row[i] (accuracy as ggml_op_mul_row).
void ggml_op_mul_rows(const INPUT0_DTYPE * __restrict src0,
                      const INPUT1_DTYPE * __restrict src1,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t row0,
                      int32_t i) {
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::mul(a, b).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a * b); }, src0,
        src1 + bcast_src1_row(row0 + i) * N);
}

#endif // GGML_OP_MUL_ROWS

#ifdef GGML_OP_DIV_ROWS

/// As ggml_op_add_rows, for out[i] = src0[i] / src1 row[i]; scalar, as ggml_op_div_row.
void ggml_op_div_rows(const INPUT0_DTYPE * __restrict src0,
                      const INPUT1_DTYPE * __restrict src1,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t row0,
                      int32_t i) {
    transform_n(
        out, N, [](auto a, auto b) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a / b); },
        src0, src1 + bcast_src1_row(row0 + i) * N);
}

#endif // GGML_OP_DIV_ROWS

// Row-tiled broadcast of a src1 that is one element per dst row (see transform_scalar_rows).
// The design takes this path on aie2p only.

#ifdef GGML_OP_ADD_SCALAR_ROWS

/**
 * @brief out = src0 + s for @p rows dst rows of @p N elements, s the src1 element of each row.
 *
 * @param[in]  src0  The tile's rows of src0.
 * @param[in]  src1  All of src1.
 * @param[out] out   The tile's rows of dst.
 * @param[in]  N     Elements per row.
 * @param[in]  rows  Rows per tile.
 * @param[in]  row0  First dst row of this worker's band.
 * @param[in]  i     Tile within the band.
 */
void ggml_op_add_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                             const INPUT1_DTYPE * __restrict src1,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N,
                             int32_t rows,
                             int32_t row0,
                             int32_t i) {
    transform_scalar_rows(
        src0, src1, out, N, rows, row0 + i * rows, [](auto a, auto s) { return aie::add(a, s); },
        [](auto a, auto s) { return static_cast<OUTPUT_DTYPE>(a + s); });
}

#endif // GGML_OP_ADD_SCALAR_ROWS

#ifdef GGML_OP_SUB_SCALAR_ROWS

/// As ggml_op_add_scalar_rows, for out = src0 - s.
void ggml_op_sub_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                             const INPUT1_DTYPE * __restrict src1,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N,
                             int32_t rows,
                             int32_t row0,
                             int32_t i) {
    transform_scalar_rows(
        src0, src1, out, N, rows, row0 + i * rows, [](auto a, auto s) { return aie::sub(a, s); },
        [](auto a, auto s) { return static_cast<OUTPUT_DTYPE>(a - s); });
}

#endif // GGML_OP_SUB_SCALAR_ROWS

#ifdef GGML_OP_MUL_SCALAR_ROWS

/// As ggml_op_add_scalar_rows, for out = src0 * s (accuracy as ggml_op_mul_row).
void ggml_op_mul_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                             const INPUT1_DTYPE * __restrict src1,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N,
                             int32_t rows,
                             int32_t row0,
                             int32_t i) {
    transform_scalar_rows(
        src0, src1, out, N, rows, row0 + i * rows,
        [](auto a, auto s) { return aie::mul(a, s).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto s) { return static_cast<OUTPUT_DTYPE>(a * s); });
}

#endif // GGML_OP_MUL_SCALAR_ROWS

#ifdef GGML_OP_DIV_SCALAR_ROWS

/// As ggml_op_add_scalar_rows, for out = src0 / s; scalar, as ggml_op_div_row.
void ggml_op_div_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                             const INPUT1_DTYPE * __restrict src1,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N,
                             int32_t rows,
                             int32_t row0,
                             int32_t i) {
    row0 += i * rows;
    for (int32_t r = 0; r < rows; ++r) {
        const auto s = src1[bcast_src1_row(row0 + r)];
        transform_n(
            out + r * N, N,
            [s](auto a) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(a / s); }, src0 + r * N);
    }
}

#endif // GGML_OP_DIV_SCALAR_ROWS

} // extern "C"
