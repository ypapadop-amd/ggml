// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file zero.cc
 * @brief Zero-initialization kernels for AIE2 matrix buffers.
 *
 * Provides scalar and vectorized functions to zero-initialize output matrices
 * before matrix multiplication accumulation.
 */

#ifndef ZERO_CC
#define ZERO_CC

#include <aie_api/aie.hpp>

/**
 * @brief Scalar zero-initialization of a matrix buffer.
 *
 * Sets all M*N elements of the output buffer to zero using scalar stores.
 *
 * @tparam T Element type of the matrix.
 * @tparam M Number of rows.
 * @tparam N Number of columns.
 *
 * @param[out] c Output buffer of M*N elements to be zeroed.
 */
template <typename T, int M, int N>
void zero_scalar(T * __restrict c) {
    for (int i = 0; i < M * N; i++) {
        c[i] = 0;
    }
}

/**
 * @brief Vectorized zero-initialization of a matrix buffer.
 *
 * Sets all M*N elements of the output buffer to zero using 256-bit vector stores.
 * More efficient than scalar version for AIE2.
 *
 * @tparam T Element type of the matrix.
 * @tparam M Number of rows (M*N must be divisible by vector width).
 * @tparam N Number of columns (M*N must be divisible by vector width).
 *
 * @param[out] c Output buffer of M*N elements to be zeroed.
 */
template <typename T, int M, int N>
void zero_vectorized(T * __restrict c) {
    constexpr int r = 256 / (sizeof(T) * 8); // one 256 bit store unit
    static_assert((M * N) % r == 0);
    const aie::vector<T, r> zeros = aie::zeros<T, r>();
    const T * __restrict c_end = c + M * N;
    event0();
    for (; c < c_end; c += r) {
        aie::store_v(c, zeros);
    }
    event1();
}

#endif