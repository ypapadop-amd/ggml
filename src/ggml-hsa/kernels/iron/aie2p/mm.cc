// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file mm.cc
 * @brief Matrix multiplication kernels for AIE2P architecture.
 *
 * This file provides scalar and vectorized matrix multiplication kernels
 * optimized for AIE2P. The vectorized kernels use the aie::mmul class with
 * 2x2 expansion and AIE2P-specific mmul shapes for optimal performance.
 */

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include "zero.cc"

#include <aie_api/aie.hpp>

/**
 * @brief Scalar matrix multiplication kernel for reference/verification.
 *
 * Computes C += A * B using scalar operations. Supports configurable memory
 * layouts for matrices B and C via template parameters.
 *
 * @tparam T_in      Input element type for matrices A and B.
 * @tparam T_out     Output element type for matrix C.
 * @tparam rowA      Number of rows in matrix A (and C).
 * @tparam colA      Number of columns in A (rows in B).
 * @tparam colB      Number of columns in B (and C).
 * @tparam b_row_maj If true, B is row-major; if false, column-major.
 * @tparam c_row_maj If true, C is row-major; if false, column-major.
 *
 * @param[in]     a Pointer to matrix A (rowA x colA, row-major).
 * @param[in]     b Pointer to matrix B (colA x colB, layout per b_row_maj).
 * @param[in,out] c Pointer to matrix C (rowA x colB, layout per c_row_maj).
 *                  Results are accumulated into C.
 */
template <typename T_in,
          typename T_out,
          int rowA,
          int colA,
          int colB,
          bool b_row_maj = true,
          bool c_row_maj = true>
static inline void matmul_scalar(T_in * a, T_in * b, T_out * c) {
    event0();
    for (int row = 0; row < rowA; row++) {
        for (int col = 0; col < colB; col++) {
            T_out running_sum = 0;
            for (int i = 0; i < colA; i++) {
                T_in a_val = a[row * colA + i];
                T_in b_val;
                if constexpr (b_row_maj) {
                    b_val = b[i * colB + col];
                } else {
                    b_val = b[i + col * colA];
                }
                running_sum += a_val * b_val;
            }
            T_out * c_ptr;
            if constexpr (c_row_maj) {
                c_ptr = &c[row * colB + col];
            } else {
                c_ptr = &c[row + col * rowA];
            }
            *c_ptr += running_sum;
        }
    }
    event1();
}

/**
 * @brief Vectorized matrix multiplication with 2x2 mmul expansion for AIE2P.
 *
 * Blocked MatMul kernel utilizing the aie::mmul class. Matrices are assumed
 * to be pre-tiled with shapes: A => rxs, B => sxt, C => rxt.
 *
 * This kernel expands the aie::mmul 2x in both A (m dimension) and B (n dimension),
 * resulting in a 2x2 expansion in output C (C00, C01, C10, C11). This expansion
 * maximizes accumulator register usage for high SIMD efficiency.
 *
 * Data layout: tiles are row-major, and data within tiles is row-major:
 * @verbatim
 *      <-s->
 *    _  ________________________
 *    r |  1 |  2 |  3 | ...
 *    _ |____|____|____|
 *      |  x | x+1| x+2| ...
 * @endverbatim
 *
 * @tparam T_in      Input element type.
 * @tparam T_out     Output element type.
 * @tparam rowA      Number of tile rows in A (in units of r).
 * @tparam colA      Number of tile columns in A / rows in B (in units of s).
 * @tparam colB      Number of tile columns in B (in units of t).
 * @tparam r         mmul M dimension.
 * @tparam s         mmul K dimension.
 * @tparam t         mmul N dimension.
 * @tparam b_row_maj If true, B tiles are row-major; if false, column-major.
 * @tparam c_row_maj If true, C tiles are row-major; if false, column-major.
 *
 * @param[in]     pA Pointer to pre-tiled matrix A.
 * @param[in]     pB Pointer to pre-tiled matrix B.
 * @param[in,out] pC Pointer to pre-tiled matrix C (accumulates results).
 *
 * @see https://xilinx.github.io/aie_api/group__group__mmul.html
 */
template <typename T_in,
          typename T_out,
          unsigned rowA,
          unsigned colA,
          unsigned colB,
          unsigned r,
          unsigned s,
          unsigned t,
          bool b_row_maj = true,
          bool c_row_maj = true>
static inline void matmul_vectorized_2x2_mmul(const T_in * __restrict pA,
                                              const T_in * __restrict pB,
                                              T_out * __restrict pC) {

    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

    event0();

    for (unsigned z = 0; z < rowA; z += 2)
        chess_prepare_for_pipelining chess_loop_range(4, ) {

            T_out * __restrict pC1;
            T_out * __restrict pC2;
            if constexpr (c_row_maj) {
                pC1 = pC + (z * colB) * MMUL::size_C;
                pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
            }

            for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
                chess_flatten_loop
#endif
                {

                    if constexpr (!c_row_maj) {
                        pC1 = pC + j * rowA * MMUL::size_C + z * MMUL::size_C;
                        pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
                    }
                    const T_in * __restrict pA1 = pA + (z * colA) * MMUL::size_A;
                    const T_in * __restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
                    const T_in * __restrict pB1;
                    const T_in * __restrict pB2;
                    if constexpr (b_row_maj) {
                        pB1 = pB + (j)*MMUL::size_B;
                        pB2 = pB + (j + 1) * MMUL::size_B;
                    } else {
                        pB1 = pB + (j * colA) * MMUL::size_B;
                        pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
                    }
                    aie::vector<T_in, MMUL::size_A> A0;
                    aie::vector<T_in, MMUL::size_A> A1;
                    aie::vector<T_in, MMUL::size_B> B0;
                    aie::vector<T_in, MMUL::size_B> B1;

                    // Load partial results from C buffer for accumulation in-place. The
                    // zero.cc function handles the zeroing of data when a new
                    // accumulation is needed (after the 'K' reduction dimension)
                    aie::vector<T_out, MMUL::size_C> acc_C00;
                    aie::vector<T_out, MMUL::size_C> acc_C01;
                    aie::vector<T_out, MMUL::size_C> acc_C10;
                    aie::vector<T_out, MMUL::size_C> acc_C11;
                    if constexpr (c_row_maj) {
                        acc_C00 = aie::load_v<MMUL::size_C>(pC1);
                        acc_C01 = aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C);
                        acc_C10 = aie::load_v<MMUL::size_C>(pC2);
                        acc_C11 = aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C);
                    } else {
                        acc_C00 = aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r);
                        acc_C01 = aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r);
                        acc_C10 =
                            aie::transpose(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r);
                        acc_C11 =
                            aie::transpose(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r);
                    }

                    MMUL C00(acc_C00);
                    MMUL C01(acc_C01);
                    MMUL C10(acc_C10);
                    MMUL C11(acc_C11);

                    for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
                        chess_flatten_loop
#endif
                        {
                            A0 = aie::load_v<MMUL::size_A>(pA1);
                            pA1 += MMUL::size_A;
                            A1 = aie::load_v<MMUL::size_A>(pA2);
                            pA2 += MMUL::size_A;
                            if constexpr (b_row_maj) {
                                B0 = aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;
                            } else {
                                B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
                                pB1 += MMUL::size_B;
                                B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
                                pB2 += MMUL::size_B;
                            }

                            C00.mac(A0, B0);
                            C01.mac(A0, B1);
                            C10.mac(A1, B0);
                            C11.mac(A1, B1);
                        }

                    // TODO make shift right here to keep most significat bits
                    // when lowering the output
                    // example below shows how to shift right 10 bits
                    // #define SHIFT 10
                    // aie::store_v(pC1, C00.template to_vector<T_out>(SHIFT));

                    if constexpr (c_row_maj) {
                        aie::store_v(pC1, C00.template to_vector<T_out>());
                        pC1 += MMUL::size_C;
                        aie::store_v(pC1, C01.template to_vector<T_out>());
                        pC1 += MMUL::size_C;
                        aie::store_v(pC2, C10.template to_vector<T_out>());
                        pC2 += MMUL::size_C;
                        aie::store_v(pC2, C11.template to_vector<T_out>());
                        pC2 += MMUL::size_C;
                    } else {
                        aie::store_v(pC1, aie::transpose(C00.template to_vector<T_out>(), r, t));
                        pC1 += MMUL::size_C;
                        aie::store_v(pC2, aie::transpose(C01.template to_vector<T_out>(), r, t));
                        pC2 += MMUL::size_C;
                        aie::store_v(pC1, aie::transpose(C10.template to_vector<T_out>(), r, t));
                        pC1 += MMUL::size_C;
                        aie::store_v(pC2, aie::transpose(C11.template to_vector<T_out>(), r, t));
                        pC2 += MMUL::size_C;
                    }
                }
        }

    event1();
}

#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

#ifdef C_COL_MAJ
constexpr bool is_c_row_maj = false;
#else
constexpr bool is_c_row_maj = true;
#endif

// The rounding mode can be set for bfloat16 mmul to improve accuracy
#ifdef ROUND_CONV_EVEN
constexpr aie::rounding_mode round_mode = aie::rounding_mode::conv_even;
#else
constexpr aie::rounding_mode round_mode = aie::rounding_mode::floor; // default
#endif

/**
 * @name AIE2P-Optimized MatMul Wrappers
 * @brief Type-specific matrix multiplication kernels optimized for AIE2P.
 *
 * These wrappers select optimal mmul shapes for each data type combination
 * on AIE2P architecture. All use 2x2 mmul expansion.
 *
 * Available shapes: https://xilinx.github.io/aie_api/group__group__mmul.html
 *
 * Each wrapper validates dimension divisibility via static_assert.
 * @{
 */

/**
 * @brief int16 -> int16 matrix multiply using 4x4x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 8).
 * @tparam k Tile K dimension (must be divisible by 4).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i16(const int16 * __restrict pA,
                                                   const int16 * __restrict pB,
                                                   int16 * __restrict pC) {
    constexpr int r = 4;
    constexpr int s = 4;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    return matmul_vectorized_2x2_mmul<int16, int16, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief int16 -> int32 matrix multiply using 4x4x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 8).
 * @tparam k Tile K dimension (must be divisible by 4).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x4x8_i16_i32(const int16 * __restrict pA,
                                                   const int16 * __restrict pB,
                                                   int32 * __restrict pC) {
    constexpr int r = 4;
    constexpr int s = 4;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    return matmul_vectorized_2x2_mmul<int16, int32, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief bfloat16 -> bfloat16 matrix multiply using 4x8x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 8).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_bf16_bf16(const bfloat16 * __restrict pA,
                                                     const bfloat16 * __restrict pB,
                                                     bfloat16 * __restrict pC) {
    constexpr int r = 4;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    ::aie::set_rounding(round_mode);

    return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief bfloat16 -> bfloat16 matrix multiply using 8x8x8 mmul shape with 2x2 expansion.
 *
 * @note This shape is only available when using bfp16 emulation
 *       (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16).
 *
 * @tparam m Tile M dimension (must be divisible by 16).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_bf16_bf16(const bfloat16 * __restrict pA,
                                                     const bfloat16 * __restrict pB,
                                                     bfloat16 * __restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    ::aie::set_rounding(round_mode);

    return matmul_vectorized_2x2_mmul<bfloat16, bfloat16, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief bfloat16 -> float32 matrix multiply using 4x8x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 8).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_4x8x8_bf16_f32(const bfloat16 * __restrict pA,
                                                    const bfloat16 * __restrict pB,
                                                    float * __restrict pC) {
    constexpr int r = 4;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    ::aie::set_rounding(round_mode);

    return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief bfloat16 -> float32 matrix multiply using 8x8x8 mmul shape with 2x2 expansion.
 *
 * @note This shape is only available when using bfp16 emulation
 *       (AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16).
 *
 * @tparam m Tile M dimension (must be divisible by 16).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_bf16_f32(const bfloat16 * __restrict pA,
                                                    const bfloat16 * __restrict pB,
                                                    float * __restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    ::aie::set_rounding(round_mode);

    return matmul_vectorized_2x2_mmul<bfloat16, float, (m / r), (k / s), (n / t), r, s, t,
                                      is_b_row_maj, is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief int8 -> int8 matrix multiply using 8x8x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 16).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i8(const int8 * __restrict pA,
                                                 const int8 * __restrict pB,
                                                 int8 * __restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    return matmul_vectorized_2x2_mmul<int8, int8, (m / r), (k / s), (n / t), r, s, t, is_b_row_maj,
                                      is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief int8 -> int16 matrix multiply using 8x8x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 16).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i16(const int8 * __restrict pA,
                                                  const int8 * __restrict pB,
                                                  int16 * __restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    return matmul_vectorized_2x2_mmul<int8, int16, (m / r), (k / s), (n / t), r, s, t, is_b_row_maj,
                                      is_c_row_maj>(pA, pB, pC);
}

/**
 * @brief int8 -> int32 matrix multiply using 8x8x8 mmul shape with 2x2 expansion.
 *
 * @tparam m Tile M dimension (must be divisible by 16).
 * @tparam k Tile K dimension (must be divisible by 8).
 * @tparam n Tile N dimension (must be divisible by 16).
 *
 * @param[in]     pA Input matrix A.
 * @param[in]     pB Input matrix B.
 * @param[in,out] pC Output matrix C (accumulated).
 */
template <unsigned m, unsigned k, unsigned n>
static inline void matmul_vectorized_8x8x8_i8_i32(const int8 * __restrict pA,
                                                  const int8 * __restrict pB,
                                                  int32 * __restrict pC) {
    constexpr int r = 8;
    constexpr int s = 8;
    constexpr int t = 8;

    static_assert(m % (2 * r) == 0);
    static_assert(k % s == 0);
    static_assert(n % (2 * t) == 0);

    return matmul_vectorized_2x2_mmul<int8, int32, (m / r), (k / s), (n / t), r, s, t, is_b_row_maj,
                                      is_c_row_maj>(pA, pB, pC);
}

extern "C" {

// If you want to compile microkernels with different inner tile sizes,
// define DIM_M, DIM_K and DIM_N at compile time using -DDIM_M 32 etc.
// These dimensions must be divisible by the r, s, t dimensions used in
// the kernels.

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef i8_i8_ONLY
#define combos(X) X(int8, i8, int8, i8, 8, 8, 8)
#endif

#ifdef i8_i16_ONLY
#define combos(X) X(int8, i8, int16, i16, 8, 8, 8)
#endif

#ifdef i8_i32_ONLY
#define combos(X) X(int8, i8, int32, i32, 8, 8, 8)
#endif

#ifdef i16_i16_ONLY
#define combos(X) X(int16, i16, int16, i16, 4, 4, 8)
#endif

#ifdef i16_i32_ONLY
#define combos(X) X(int16, i16, int32, i32, 4, 4, 8)
#endif

// The emulation of bf16 changes the available shapes for matrix multiplication
#ifdef bf16_bf16_ONLY
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)
#else
#define combos(X) X(bfloat16, bf16, bfloat16, bf16, 4, 8, 8)
#endif
#endif

#ifdef bf16_f32_ONLY
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X) X(bfloat16, bf16, float, f32, 8, 8, 8)
#else
#define combos(X) X(bfloat16, bf16, float, f32, 4, 8, 8)
#endif
#endif

#ifdef f32_f32_ONLY
// f32 input has no vectorized MAC support on AIE2p, use scalar only
#define combos(X) X(float, f32, float, f32, 1, 1, 1)
#define SCALAR_ONLY
#endif

#ifndef combos
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define combos(X)                                                                                  \
    X(int8, i8, int8, i8, 8, 8, 8)                                                                 \
    X(int16, i16, int16, i16, 4, 4, 8)                                                             \
    X(int16, i16, int32, i32, 4, 4, 8)                                                             \
    X(bfloat16, bf16, bfloat16, bf16, 8, 8, 8)                                                     \
    X(bfloat16, bf16, float, f32, 8, 8, 8)
#else
#define combos(X)                                                                                  \
    X(int8, i8, int8, i8, 8, 8, 8)                                                                 \
    X(int16, i16, int16, i16, 4, 4, 8)                                                             \
    X(int16, i16, int32, i32, 4, 4, 8)                                                             \
    X(bfloat16, bf16, bfloat16, bf16, 4, 8, 8)                                                     \
    X(bfloat16, bf16, float, f32, 4, 8, 8)
#endif
#endif

#define matmul_vectorized_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)        \
    void matmul_##mlir_type_in##_##mlir_type_out(ctype_in * a_in, ctype_in * b_in,                 \
                                                 ctype_out * c_out) {                              \
        matmul_vectorized_##r##x##s##x##t##_##mlir_type_in##_##mlir_type_out<DIM_M, DIM_K, DIM_N>( \
            a_in, b_in, c_out);                                                                    \
    }

#define matmul_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)            \
    void matmul_scalar_##mlir_type_in##_##mlir_type_out(ctype_in * a_in, ctype_in * b_in,          \
                                                        ctype_out * c_out) {                       \
        matmul_scalar<ctype_in, ctype_out, DIM_M, DIM_K, DIM_N, is_b_row_maj, is_c_row_maj>(       \
            a_in, b_in, c_out);                                                                    \
    }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)          \
    void zero_##mlir_type_out(ctype_out * c_out) {                                                 \
        zero_vectorized<ctype_out, DIM_M, DIM_N>(c_out);                                           \
    }

#define zero_scalar_c_func(ctype_in, mlir_type_in, ctype_out, mlir_type_out, r, s, t)              \
    void zero_scalar_##mlir_type_out(ctype_out * c_out) {                                          \
        zero_scalar<ctype_out, DIM_M, DIM_N>(c_out);                                               \
    }

combos(matmul_vectorized_c_func) combos(matmul_scalar_c_func) combos(zero_vectorized_c_func)
    combos(zero_scalar_c_func)

} // extern "C"