// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file transform.h
 * @brief Arity-generic element-wise transform helpers shared by the AIE kernels.
 *
 * Every element-wise kernel in this directory has the same shape: walk N elements, apply an
 * op, store. The only thing that varies is how many inputs it reads and whether it has a
 * vector body. These two helpers cover both axes, so a unary and a binary kernel share one
 * implementation instead of each carrying its own copy.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

/**
 * @brief Whether the Python builder opted this kernel in to a vector body.
 *
 * A vector body is only worth having if the kernel streams more than one vector register per
 * object-fifo round trip; with the one-register tile it pays an acquire/release per vector,
 * which dominates the compute it just saved. The Python op builders define
 * GGML_VECTORIZED_TILING on the paths whose tile is chosen for that (the L1-budgeted tile, or a
 * whole row), and @ref transform_vector_n asserts on it, so a vector body can never be paired
 * with the one-register tile by accident.
 *
 * Note this is permission, not a measurement of the tile: the row path's tile is ne0, which for
 * a narrow row can still be below one vector register. That is intended -- the row kernels were
 * measured as a win at those widths -- and the vector body degrades to its scalar tail there.
 *
 * Templated so the static_assert below is dependent, and therefore checked per instantiation
 * rather than when the template is defined.
 */
template <typename>
inline constexpr bool vectorized_tiling_v =
#ifdef GGML_VECTORIZED_TILING
    true;
#else
    false;
#endif

/**
 * @brief Applies an op element-wise over N elements: out[i] = op(in[i]...).
 *
 * Scalar for every element. Use @ref transform_vector_n instead when the op has an aie::
 * vector equivalent.
 *
 * @tparam TOut Output element type.
 * @tparam Size Integer type of the element count.
 * @tparam Op   Callable taking one element per input.
 * @tparam TIn  Input element types, one per input array.
 *
 * @param[out] out   Output array of count elements.
 * @param[in]  count Number of elements to process.
 * @param[in]  op    Operation to apply.
 * @param[in]  in    One input array of count elements per operand.
 */
template <typename TOut, typename Size, typename Op, typename... TIn>
void transform_n(TOut * __restrict out, Size count, Op op, const TIn * __restrict... in) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in[i]...);
    }
    event1();
}

/**
 * @brief Applies an op over N elements with a vector body and a scalar tail.
 *
 * The vector path is gated on every input sharing the output type, because it operates on the
 * loaded vectors with no per-element cast; a widening or narrowing op falls back to the scalar
 * path for the whole range. The tail covers any remainder when N is not a whole number of
 * vectors, and covers everything when there is no vector path.
 *
 * @tparam Aligned  Whether the tile base is 512-bit aligned, so aligned loads/stores are safe.
 *                  False (the default) is always correct. True is only valid when the streamed
 *                  tile is a whole number of vector registers; in particular a design that
 *                  streams rows through double-buffered fifos whose per-row object stride need
 *                  not be vector-aligned must leave this false, or aligned accesses corrupt
 *                  alternate (ping-pong) rows.
 * @tparam EnableVector Set false to force the scalar path for the whole range. Use it when the
 *                  vector and scalar formulations of an op disagree for some element type --
 *                  aie::mul narrows through an accumulator, which saturates for integers where
 *                  the scalar `a * b` wraps -- so the op can keep its vector body for the types
 *                  where the two agree without changing results for the types where they do not.
 * @tparam TOut     Output element type.
 * @tparam VecOp    Callable taking one vector per input.
 * @tparam ScalarOp Callable taking one element per input, for the tail and the scalar path.
 * @tparam TIn      Input element types, one per input array.
 *
 * @param[out] out       Output array of N elements.
 * @param[in]  N         Number of elements to process.
 * @param[in]  vec_op    Vector operation to apply.
 * @param[in]  scalar_op Scalar operation to apply.
 * @param[in]  in        One input array of N elements per operand.
 */
template <bool Aligned = false,
          bool EnableVector = true,
          typename TOut,
          typename VecOp,
          typename ScalarOp,
          typename... TIn>
void transform_vector_n(TOut * __restrict out,
                        int32_t N,
                        VecOp vec_op,
                        ScalarOp scalar_op,
                        const TIn * __restrict... in) {
    static_assert(vectorized_tiling_v<TOut>,
                  "this op has a vectorized body but was compiled without "
                  "GGML_VECTORIZED_TILING, so it streams one vector register per object-fifo "
                  "round trip: give it the large tile in the op's Python builder");

    event0();

    int32_t vend = 0;

    if constexpr (EnableVector && (std::is_same_v<TIn, TOut> && ...)) {
        constexpr int32_t V = 512 / (sizeof(TOut) * 8);
        vend = (N / V) * V; // division by constexpr V -> inline shift, once

        // No AIE_LOOP_MIN_ITERATION_COUNT: a tile narrower than V leaves vend 0, and promising
        // >= 1 iteration would make the pipelined prologue run the body on too few elements.
        AIE_PREPARE_FOR_PIPELINING
        for (int32_t i = 0; i < vend; i += V) {
            if constexpr (Aligned) {
                aie::store_v(out + i, vec_op(aie::load_v<V>(in + i)...));
            } else {
                aie::store_unaligned_v(out + i, vec_op(aie::load_unaligned_v<V>(in + i)...));
            }
        }
    }

    // Tail of the vector loop, or the whole range when there is no vector path.
    for (int32_t i = vend; i < N; ++i) {
        out[i] = scalar_op(in[i]...);
    }

    event1();
}
