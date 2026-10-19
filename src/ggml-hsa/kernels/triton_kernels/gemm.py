# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernels for matrix multiplication.

``bare_matmul`` is the NPU (Triton-XDNA) kernel, a single-block C = A @ B ported verbatim from
Triton-XDNA examples/matmul_bf16_m64_n64_k64. All shape and stride parameters are compile-time
constants; the NPU tiling/packing lives in the paired matmul_<arch>.mlir transform scripts, not in
this kernel.

``gemm_kernel`` is the GPU kernel for GGML MUL_MAT (tiled, masked, portable ``tl.dot``).
The GPU kernels' only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


@triton.jit
def bare_matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute C = A @ B for a single (BLOCK_SIZE_M, BLOCK_SIZE_N) output block."""
    pid_m = tl.program_id(0)  # block row id
    pid_n = tl.program_id(1)  # block column id

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)


# ---------------------------------------------------------------------------
# Matrix multiply (GGML MUL_MAT): dst = src0 @ src1^T
#   src0 (A): [K, M] row-major -> logically (M, K)
#   src1 (B): [K, N] row-major -> logically (N, K)
#   dst     : ne0=M, ne1=N; dst(i0=im, i1=in) at offset in*M + im
#   dst(im, in) = sum_k A(im, k) * B(in, k)
# ---------------------------------------------------------------------------
@triton.jit
def gemm_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    NUM_PID_N: tl.constexpr,
):
    """Compute C = A @ B^T for GGML MUL_MAT (portable ``tl.dot``; masked edges)."""
    pid = tl.program_id(0)
    pid_m = pid // NUM_PID_N
    pid_n = pid % NUM_PID_N

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = B + (offs_n[:, None] * K + offs_k[None, :])

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        k_mask = (offs_k[None, :] + k0) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & k_mask, other=0.0)
        # f16/bf16 operands are widened to f32: tl.dot rejects mixed operand dtypes (e.g. f16
        # weights x f32 activations), and widening needs no native f16/bf16 dot support.
        acc += tl.dot(
            a.to(tl.float32), tl.trans(b).to(tl.float32), input_precision="ieee"
        )
        a_ptrs += BK
        b_ptrs += BK

    # dst(im, in) at offset in*M + im  -> store the transpose of the (BM, BN) accumulator
    c_ptrs = C + (offs_n[:, None] * M + offs_m[None, :])
    c_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(c_ptrs, tl.trans(acc).to(C.dtype.element_ty), mask=c_mask)
