# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU (gfx / HIP) kernels for the GGML HSA backend.

These kernels back the native Triton -> hsaco JIT path (see ``build_triton.py``). They follow the
kernarg ABI that ``gpu-kernel.cpp`` expects: the only *runtime* kernel arguments are the tensor
pointers (source tensors in order, then the destination); every dimension, stride, block size and
launch parameter is baked in as a ``tl.constexpr`` (the JIT specialises per shape, so the kernel
name already encodes the shapes and constexpr baking is free). The launch geometry (number of
programs, workgroup size) is emitted into the ``.hsaco.json`` sidecar by ``build_triton.py``.

Public kernels adapted from the upstream Triton tutorials:
  * ``gemm_kernel`` - matrix multiply (tutorial 03, portable ``tl.dot`` path; no MFMA, gfx1151 is
    RDNA3.5 / wave32).
  * ``softmax_kernel`` - fused softmax (tutorial 02).
  * ``relu_kernel`` / ``bcast_binary_kernel`` / ``argmax_kernel`` - trivial elementwise / reduction.
"""

import triton
import triton.language as tl


def next_pow2(n: int) -> int:
    """Return the smallest power of two that is >= ``n`` (>= 1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Elementwise unary: RELU (dst = max(src, 0))
# ---------------------------------------------------------------------------
@triton.jit
def relu_kernel(X, Y, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    """Compute Y = max(X, 0) over ``n_elements`` (masked, so any shape works)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(Y + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Elementwise binary with GGML broadcasting: C = A (op) B
#   A and C are the (contiguous) destination-shaped tensors; B broadcasts along
#   any dimension whose src1 extent is 1 (GGML semantics: index modulo src1 ne).
# ---------------------------------------------------------------------------
@triton.jit
def bcast_binary_kernel(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    NE0: tl.constexpr,
    NE1: tl.constexpr,
    NE2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute C = A (op) B with GGML broadcasting of B into A's (dst) shape.

    ``NE0..NE2`` are the destination extents of the three innermost dims (the outermost dim is
    implied by ``n_elements``); ``B0..B3`` are the src1 extents. The broadcast source index for
    each element is computed per GGML convention (index modulo the src1 extent per dim).
    ``OP`` selects the operation (0:add, 1:sub, 2:mul, 3:div).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # decode contiguous dst index -> (i0, i1, i2, i3)
    i0 = offs % NE0
    r = offs // NE0
    i1 = r % NE1
    r = r // NE1
    i2 = r % NE2
    i3 = r // NE2

    # broadcast into src1 (modulo its per-dim extent), then linearise with src1 strides
    j0 = i0 % B0
    j1 = i1 % B1
    j2 = i2 % B2
    j3 = i3 % B3
    boff = j0 + B0 * (j1 + B1 * (j2 + B2 * j3))

    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + boff, mask=mask)
    if OP == 0:
        c = a + b
    elif OP == 1:
        c = a - b
    elif OP == 2:
        c = a * b
    else:
        c = a / b
    tl.store(C + offs, c, mask=mask)


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
        acc += tl.dot(a, tl.trans(b), input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK

    # dst(im, in) at offset in*M + im  -> store the transpose of the (BM, BN) accumulator
    c_ptrs = C + (offs_n[:, None] * M + offs_m[None, :])
    c_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(c_ptrs, tl.trans(acc), mask=c_mask)


# ---------------------------------------------------------------------------
# Fused softmax over the innermost dim (GGML SOFT_MAX, no mask / no ALiBi)
#   one program per row; row length = n_cols (= ne0)
# ---------------------------------------------------------------------------
@triton.jit
def softmax_kernel(
    X,
    Y,
    n_cols: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute Y = softmax(X * SCALE) along each row of length ``n_cols``."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=float("-inf"))
    x = x * SCALE
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(Y + row * n_cols + offs, e / s, mask=mask)


# ---------------------------------------------------------------------------
# Argmax over the innermost dim (GGML ARGMAX): one program per row, i32 output
# ---------------------------------------------------------------------------
@triton.jit
def argmax_kernel(
    X,
    Y,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Store into Y[row] the index of the maximum element of row ``row`` of X."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=float("-inf"))
    idx = tl.argmax(x, axis=0)
    tl.store(Y + row, idx.to(tl.int32))
