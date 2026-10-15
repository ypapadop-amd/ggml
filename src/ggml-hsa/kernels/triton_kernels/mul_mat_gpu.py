# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernel for GGML_OP_MUL_MAT on AMD GPUs: C = A @ B^T in GGML layout."""

import triton
import triton.language as tl


@triton.jit
def mul_mat_gpu(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute one (BLOCK_SIZE_M, BLOCK_SIZE_N) tile of a GGML MUL_MAT.

    GGML stores A as [K, M] and B as [K, N] (innermost dimension first), i.e. row m of A and row
    n of B are each K contiguous elements, and C as [M, N], so C[n * M + m] = sum_k A[m * K + k] *
    B[n * K + k]. The grid is 1-D (the GPU dispatch only launches 1-D grids), so the program id
    is split into an M tile and an N tile. All accesses are masked, so M, N and K need not be
    multiples of the block sizes. Accumulation is in f32; the result is cast to C's dtype.

    Args:
        A: Pointer to the [K, M] first operand.
        B: Pointer to the [K, N] second operand.
        C: Pointer to the [M, N] result.
        M: Rows of A and the innermost dimension of C.
        N: Rows of B and the outer dimension of C.
        K: Shared (reduction) dimension.
        BLOCK_SIZE_M: Tile size along M.
        BLOCK_SIZE_N: Tile size along N.
        BLOCK_SIZE_K: Reduction step along K.
    """
    pid = tl.program_id(0)
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_SIZE_K):
        k = k0 + offs_k
        mask_k = k < K
        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K], b: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        a = tl.load(
            A + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            B + offs_n[None, :] * K + k[:, None],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        # Mixed operand dtypes (e.g. f16 weights x f32 activations) are not valid for tl.dot,
        # and "ieee" keeps f32 operands from being rounded to tf32.
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")

    c = acc.to(C.dtype.element_ty)
    tl.store(
        C + offs_n[None, :] * M + offs_m[:, None],
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )
