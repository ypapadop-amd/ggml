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
 * @brief Bits per vector step of the MUL kernels (transform_vector_walk_n's VectorBits).
 *
 * There is no f32 vector multiplier: aie::mul on f32 is emulated on bf16 limbs, and on aie2p a
 * step of 32 lanes costs about what one of 16 does (static II 61 against 58 for the row kernel's
 * aligned loop), so f32 steps 1024 bits there. Other types, and aie2, step one register.
 */
template <typename T>
constexpr unsigned mul_vector_bits = __AIEARCH__ != 20 && std::is_same_v<T, float> ? 1024 : 512;

/// Whether the DIV kernels divide f32 in vectors (div_f32_rne); aie2 keeps the scalar division.
template <typename TOut, typename T0, typename T1>
constexpr bool div_vector_v = __AIEARCH__ != 20 && std::is_same_v<TOut, float> &&
                              std::is_same_v<T0, float> && std::is_same_v<T1, float>;

/**
 * @brief a / b on 16 f32 lanes, correctly rounded (to nearest, ties to even), by integer long
 * division of the significands.
 *
 * There is no vector divide, and aie::div is an approximate reciprocal times a multiply; the
 * scalar division is the __divsf3 libcall. Here the quotient of the two 24-bit significands is
 * found one bit per step, 25 steps (24 bits and the rounding bit), each a compare of the
 * remainder with the divisor's significand, a select of the remainder and of the quotient bit.
 * With the remainder as the sticky bit that rounds exactly as IEEE division does, so every lane
 * it computes equals the CPU's quotient bit for bit.
 *
 * Lanes whose operands or quotient leave the normal range (a subnormal, infinite or NaN operand,
 * a zero divisor, or a quotient exponent outside [1, 254]) are not computed; they are set in
 * @p bad and the caller divides those in scalar. A zero dividend gives a signed zero.
 *
 * @param[in]     a   Dividends.
 * @param[in]     b   Divisors.
 * @param[in,out] bad Or-ed with the lanes not computed.
 * @return a / b in the lanes not in @p bad.
 */
__attribute__((always_inline)) inline aie::vector<float, 16>
div_f32_rne(aie::vector<float, 16> a, aie::vector<float, 16> b, aie::mask<16> & bad) {
    using v32 = aie::vector<std::int32_t, 16>;
    constexpr std::int32_t implicit = 1 << 23;
    constexpr std::int32_t exp_field = 0x7f800000;
    const v32 ua = a.cast_to<std::int32_t>();
    const v32 ub = b.cast_to<std::int32_t>();
    const v32 ea = aie::bit_and(exp_field, ua); // exponents in place, as << 23
    const v32 eb = aie::bit_and(exp_field, ub);
    const v32 sign = aie::bit_and(std::int32_t(0x80000000u), aie::bit_xor(ua, ub));
    const v32 ma = aie::bit_or(implicit, aie::bit_and(implicit - 1, ua));
    const v32 mb = aie::bit_or(implicit, aie::bit_and(implicit - 1, ub));

    // Scale the dividend's significand into [mb, 2 mb), so the quotient's leading bit is 1.
    const auto below = aie::lt(ma, mb);
    v32 r = aie::select(ma, aie::add(ma, ma), below);
    // (e - 128) << 23 for the quotient's biased exponent e = ea - eb + 127 - below, before a
    // carry out of the rounding (offset by 128 so it cannot overflow).
    const v32 eo = aie::add(aie::sub(ea, eb), aie::select(-implicit, -2 * implicit, below));

    // One quotient bit per step: mant collects the 24 significand bits, r is the remainder,
    // doubled. A 25th step gives the round bit, and what remains the sticky bit.
    v32 mant = aie::zeros<std::int32_t, 16>();
    AIE_LOOP_UNROLL_FULL
    for (int step = 0; step < 24; ++step) {
        const auto ge = aie::ge(r, mb);
        const v32 m2 = aie::add(mant, mant);
        mant = aie::select(m2, aie::add(m2, std::int32_t{1}), ge);
        r = aie::select(r, aie::sub(r, mb), ge);
        r = aie::add(r, r);
    }
    const auto round = aie::ge(r, mb);
    const auto odd_or_sticky = aie::neq(aie::bit_and(std::int32_t{1}, mant), std::int32_t{0}) |
                               aie::neq(r, mb); // with round set, sticky is r - mb != 0
    const v32 rounded =
        aie::add(mant, aie::select(std::int32_t{0}, std::int32_t{1}, round & odd_or_sticky));
    // (e - 1) << 23 plus the significand with its implicit bit: a carry to 2^24 bumps the exponent.
    v32 bits = aie::bit_or(sign, aie::add(aie::add(eo, 127 * implicit), rounded));

    const auto a_zero = aie::eq(aie::add(ua, ua), std::int32_t{0}); // +0 or -0
    bits = aie::select(bits, sign, a_zero);
    const auto special =
        aie::eq(ea, exp_field) | aie::eq(eb, std::int32_t{0}) | aie::eq(eb, exp_field);
    // e outside [1, 254]: e - 128 outside [-127, 126]. (At 254 a carry out of the rounding
    // gives the exponent 255 and a zero fraction, the infinity IEEE rounds to.)
    const auto out_of_range =
        aie::eq(ea, std::int32_t{0}) | aie::lt(eo, -127 * implicit) | aie::gt(eo, 126 * implicit);
    bad = bad | special | (out_of_range & ~a_zero);
    return bits.cast_to<float>();
}

/**
 * @brief out = in / divisor over N elements: when div_vector_v, in vectors of div_f32_rne with a
 * scalar tail, else (and for the whole range when any lane falls outside div_f32_rne's range) by
 * the scalar division.
 *
 * A lane outside the vector path's range is rare, so the whole range is redone in scalar when
 * one occurs, keeping the vector loop free of branches.
 *
 * @param[out] out     Output array of N elements.
 * @param[in]  N       Number of elements.
 * @param[in]  in      Dividends, N elements.
 * @param[in]  divisor The divisors: an array of N elements, or one value for all of them.
 */
template <typename TOut, typename TIn, typename Divisor>
void div_n(TOut * __restrict out, int32_t N, const TIn * __restrict in, Divisor divisor) {
    constexpr bool per_element = std::is_pointer_v<Divisor>;
    using TDiv = std::remove_cv_t<std::remove_pointer_t<Divisor>>;
    const auto scalar_div = [](auto a, auto b) -> TOut { return static_cast<TOut>(a / b); };
    const auto scalar = [&]() {
        if constexpr (per_element) {
            transform_n(out, N, scalar_div, in, divisor);
        } else {
            transform_n(
                out, N, [divisor, scalar_div](auto a) { return scalar_div(a, divisor); }, in);
        }
    };
    if constexpr (div_vector_v<TOut, TIn, TDiv>) {
        // A vector is ~230 bundles of div_f32_rne (one vector-ALU op each), which the loads and
        // stores hide in: load unaligned and store through store_v_any, whatever the alignment.
        constexpr int32_t V = 16;
        auto bad = aie::mask<V>::from_uint32(0u);
        const TIn * __restrict a = in;
        TOut * __restrict o = out;
        [[maybe_unused]] const TDiv * __restrict b = nullptr;
        [[maybe_unused]] aie::vector<TDiv, V> bv;
        if constexpr (per_element) {
            b = divisor;
        } else {
            bv = aie::broadcast<TDiv, V>(divisor);
        }
        const int32_t vend = N / V * V;
        for (int32_t i = 0; i < vend; i += V) {
            if constexpr (per_element) {
                bv = aie::load_unaligned_v<V>(b);
                b += V;
            }
            store_v_any(o, div_f32_rne(aie::load_unaligned_v<V>(a), bv, bad));
            a += V;
            o += V;
        }
        for (int32_t i = vend; i < N; ++i) {
            if constexpr (per_element) {
                out[i] = scalar_div(in[i], divisor[i]);
            } else {
                out[i] = scalar_div(in[i], divisor);
            }
        }
        if (!bad.empty()) {
            scalar();
        }
    } else {
        scalar();
    }
}

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
 * @tparam VectorBits As transform_vector_walk_n's.
 */
template <unsigned VectorBits = 512, typename VecOp, typename ScalarOp>
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
        transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>, VectorBits>(
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
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>,
                            mul_vector_bits<OUTPUT_DTYPE>>(
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
    div_n(out, N, in0, in1);
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
    transform_vector_walk_n</*Aligned=*/false, /*EnableVector=*/true,
                            mul_vector_bits<OUTPUT_DTYPE>>(
        out, N, [](auto a, auto b) { return aie::mul(a, b).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto b) { return static_cast<OUTPUT_DTYPE>(a * b); }, src0, src1);
}

#endif // GGML_OP_MUL_ROW

#ifdef GGML_OP_DIV_ROW

/**
 * @brief out[i] = src0[i] / src1[i] for one dst row; src1 is reused across all dst rows.
 *
 * There is no float vector divide instruction, and aie::div is a multiply by aie::inv, an
 * approximate reciprocal orders of magnitude off one ulp. So the division is either scalar (aie2,
 * and other types) or, for f32 on aie2p, an exact integer long division in vectors (div_n,
 * div_f32_rne), bit-identical to the scalar one. Row-tiling removes the seven per-element
 * broadcast divisions, leaving only the one the operation requires.
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
    div_n(out, N, src0, src1);
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
    transform_vector_walk_n</*Aligned=*/false, is_floating_point_v<OUTPUT_DTYPE>,
                            mul_vector_bits<OUTPUT_DTYPE>>(
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
    div_n(out, N, src0, src1 + bcast_src1_row(row0 + i) * N);
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
    transform_scalar_rows<mul_vector_bits<OUTPUT_DTYPE>>(
        src0, src1, out, N, rows, row0 + i * rows,
        [](auto a, auto s) { return aie::mul(a, s).template to_vector<OUTPUT_DTYPE>(); },
        [](auto a, auto s) { return static_cast<OUTPUT_DTYPE>(a * s); });
}

#endif // GGML_OP_MUL_SCALAR_ROWS

#ifdef GGML_OP_DIV_SCALAR_ROWS

/// As ggml_op_add_scalar_rows, for out = src0 / s (div_n).
void ggml_op_div_scalar_rows(const INPUT0_DTYPE * __restrict src0,
                             const INPUT1_DTYPE * __restrict src1,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N,
                             int32_t rows,
                             int32_t row0,
                             int32_t i) {
    row0 += i * rows;
    for (int32_t r = 0; r < rows; ++r) {
        div_n(out + r * N, N, src0 + r * N, src1[bcast_src1_row(row0 + r)]);
    }
}

#endif // GGML_OP_DIV_SCALAR_ROWS

} // extern "C"
