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
#include "ggml-aie.hpp"

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
 * @brief Whether the Python builder guaranteed every tile boundary is 512-bit aligned.
 *
 * Aligned loads/stores are only safe when the streamed tile is a whole number of vector
 * registers. A design that streams rows through double-buffered fifos whose per-row object
 * stride is not a multiple of the vector width would otherwise corrupt alternate (ping-pong)
 * rows, silently and with no diagnostic. The op builders define GGML_TILE_VECTOR_ALIGNED only
 * when the tile they chose satisfies that, and @ref transform_vector_n asserts on it whenever
 * Aligned is requested, so the guarantee cannot be assumed by a call site that does not have it.
 */
template <typename>
inline constexpr bool tile_vector_aligned_v =
#ifdef GGML_TILE_VECTOR_ALIGNED
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
 * @tparam Aligned  Use aligned loads/stores. False (the default) is always correct; true is
 *                  guarded by a static_assert on @ref tile_vector_aligned_v, so a call site can
 *                  only request it when its Python builder guaranteed the tile qualifies.
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
    static_assert(!Aligned || tile_vector_aligned_v<TOut>,
                  "Aligned=true needs every tile boundary 512-bit aligned, which only holds when "
                  "the tile is a whole number of vector registers: the op's Python builder must "
                  "define GGML_TILE_VECTOR_ALIGNED, and only does so when its tile qualifies");

    static_assert(vectorized_tiling_v<TOut>,
                  "this op has a vectorized body but was compiled without "
                  "GGML_VECTORIZED_TILING, so it streams one vector register per object-fifo "
                  "round trip: give it the large tile in the op's Python builder");

    event0();

    int32_t vend = 0;

    if constexpr (EnableVector && (std::is_same_v<TIn, TOut> && ...)) {
        constexpr int32_t V = 512 / (sizeof(TOut) * 8);
        vend = (N / V) * V; // division by constexpr V -> inline shift, once

        // Without the builder's guarantee every store goes through store_v_any, at the
        // alignment of out, which every store shares (they are a whole vector apart).
        const auto loop = [&](auto store) {
            const auto body = [&](int32_t i) {
                if constexpr (Aligned) {
                    aie::store_v(out + i, vec_op(aie::load_v<V>(in + i)...));
                } else {
                    store_v_any<decltype(store)::value>(
                        out + i, vec_op(aie::load_unaligned_v<V>(in + i)...));
                }
            };
#if __AIEARCH__ == 20 // aie2 only so far; aie2p keeps the single loop below, unmeasured there.
            // A tile narrower than V leaves vend 0, so the minimum trip count is versioned.
            GGML_AIE_LOOP_MIN_TRIPS(4, i, vend, V, body(i);)
#else
            // No AIE_LOOP_MIN_ITERATION_COUNT: a tile narrower than V leaves vend 0, and promising
            // >= 1 iteration would make the pipelined prologue run the body on too few elements.
            AIE_PREPARE_FOR_PIPELINING
            for (int32_t i = 0; i < vend; i += V) {
                body(i);
            }
#endif
        };
        using aligned_store = std::integral_constant<store_alignment, store_alignment::aligned>;
        using unaligned_store = std::integral_constant<store_alignment, store_alignment::unaligned>;
        if constexpr (Aligned) {
            loop(aligned_store{});
        } else if (store_v_any_aligned<V>(out)) {
            loop(aligned_store{});
        } else {
            loop(unaligned_store{});
        }
    }

    // Tail of the vector loop, or the whole range when there is no vector path.
    for (int32_t i = vend; i < N; ++i) {
        out[i] = scalar_op(in[i]...);
    }

    event1();
}

/**
 * @brief @ref transform_vector_n with walking pointers, and aligned accesses whenever the
 * pointers allow them.
 *
 * Same contract as @ref transform_vector_n (vector body only when every input shares the output
 * type, scalar tail), with two differences in the vector body:
 *
 * - Each operand is a cursor advanced one vector per iteration, instead of base + i. Peano does
 *   not strength-reduce base + i, and recomputing three addresses through one pointer register
 *   each iteration serialized the loop.
 * - Without the Python builder's guarantee (Aligned false), the pointers are checked at run time:
 *   when all of them are aligned for a vector load/store (aie::vector_ldst_align_v), the body uses
 *   aligned accesses, and unaligned ones otherwise. An unaligned store is a read-modify-write of
 *   the vectors it straddles, so it costs several times an aligned one.
 *
 * @tparam Aligned      As in @ref transform_vector_n: true only when the builder guaranteed it.
 * @tparam EnableVector As in @ref transform_vector_n.
 * @tparam VectorBits   Bits per vector step: 512, one register, or 1024 for an op whose vector
 *                      form costs the same at twice the lanes (the emulated f32 multiply on aie2p).
 *                      A 1024-bit loop ends with at most one 512-bit step before the scalar tail.
 */
template <bool Aligned = false,
          bool EnableVector = true,
          unsigned VectorBits = 512,
          typename TOut,
          typename VecOp,
          typename ScalarOp,
          typename... TIn>
void transform_vector_walk_n(TOut * __restrict out,
                             int32_t N,
                             VecOp vec_op,
                             ScalarOp scalar_op,
                             const TIn * __restrict... in) {
    static_assert(!Aligned || tile_vector_aligned_v<TOut>,
                  "Aligned=true needs every tile boundary 512-bit aligned: see transform_vector_n");
    static_assert(vectorized_tiling_v<TOut>,
                  "this op has a vectorized body but was compiled without GGML_VECTORIZED_TILING: "
                  "see transform_vector_n");

    event0();

    int32_t vend = 0;

    if constexpr (EnableVector && (std::is_same_v<TIn, TOut> && ...)) {
        static_assert(VectorBits == 512 || VectorBits == 1024, "a vector step is 512 or 1024 bits");
        constexpr int32_t V = VectorBits / (sizeof(TOut) * 8);
        constexpr int32_t H = VectorBits > 512 ? V / 2 : V; // lanes of the last vector step
        vend = (N / H) * H;
        // With H < V, the end of the steps of V lanes; one step of H lanes covers the rest of vend.
        [[maybe_unused]] const int32_t wend = (N / V) * V;

        // aligned: every pointer is aligned for a vector access. store: else, the alignment of
        // out for store_v_any, which every store shares (they are a whole vector apart).
        const auto loop = [&](auto aligned, auto store) {
            TOut * __restrict o = out;
            [&](auto... p) {
                const auto body = [&](int32_t) {
                    if constexpr (decltype(aligned)::value) {
                        aie::store_v(o, vec_op(aie::load_v<V>(p)...));
                    } else {
                        store_v_any<decltype(store)::value>(o,
                                                            vec_op(aie::load_unaligned_v<V>(p)...));
                    }
                    o += V;
                    ((p += V), ...);
                };
#if __AIEARCH__ == 20 // As in transform_vector_n: the promise is measured on aie2 only.
                if constexpr (H < V) {
                    GGML_AIE_LOOP_MIN_TRIPS(4, i, wend, V, body(i);)
                } else {
                    GGML_AIE_LOOP_MIN_TRIPS(4, i, vend, V, body(i);)
                }
#else
                if constexpr (H < V) {
                    // A wide step is one long latency-bound chain (the emulated f32 multiply):
                    // two in one body overlap.
                    AIE_LOOP_UNROLL(2)
                    for (int32_t i = 0; i < wend; i += V) {
                        body(i);
                    }
                } else {
                    for (int32_t i = 0; i < vend; i += V) {
                        body(i);
                    }
                }
#endif
                if constexpr (H < V) {
                    if (wend < vend) {
                        if constexpr (decltype(aligned)::value) {
                            aie::store_v(o, vec_op(aie::load_v<H>(p)...));
                        } else {
                            store_v_any<decltype(store)::value>(
                                o, vec_op(aie::load_unaligned_v<H>(p)...));
                        }
                    }
                }
            }(in...);
        };

        using aligned_store = std::integral_constant<store_alignment, store_alignment::aligned>;
        using unaligned_store = std::integral_constant<store_alignment, store_alignment::unaligned>;
        if constexpr (Aligned) {
            loop(std::true_type{}, aligned_store{});
        } else if (vend > 0) { // a row narrower than one vector step skips the check
            constexpr uintptr_t mask = aie::vector_ldst_align_v<TOut, V> - 1;
            if (((reinterpret_cast<uintptr_t>(out) | ... | reinterpret_cast<uintptr_t>(in)) &
                 mask) == 0) {
                loop(std::true_type{}, aligned_store{});
            } else if (store_v_any_aligned<V>(out)) {
                loop(std::false_type{}, aligned_store{});
            } else {
                loop(std::false_type{}, unaligned_store{});
            }
        }
    }

    // Tail of the vector loop, or the whole range when there is no vector path.
    for (int32_t i = vend; i < N; ++i) {
        out[i] = scalar_op(in[i]...);
    }

    event1();
}
