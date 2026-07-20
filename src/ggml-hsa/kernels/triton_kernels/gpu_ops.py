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


# ---------------------------------------------------------------------------
# IM2COL (GGML_OP_IM2COL, is_2D): masked gather of input patches
#   src1 (image, contiguous): ne=[IW, IH, IC, N]
#   dst (contiguous): ne=[IC*KH*KW, OW, OH, N]
#   dst[(in*OH*OW + ioh*OW + iow)*ICKHKW + iic*KH*KW + ikh*KW + ikw]
#       = image(iiw, iih, iic, in), with iiw = iow*S0 + ikw*D0 - P0,
#         iih = ioh*S1 + ikh*D1 - P1 (0 if out of bounds).
# ``KERN`` is the (unused) kernel/src0 pointer, present only so the kernarg
# order matches the dispatch (src0, src1, dst).
# ---------------------------------------------------------------------------
@triton.jit
def im2col_kernel(
    KERN,
    SRC,
    DST,
    n_elements: tl.constexpr,
    ICKHKW: tl.constexpr,
    KW: tl.constexpr,
    KH: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    IW: tl.constexpr,
    IH: tl.constexpr,
    IWIH: tl.constexpr,
    IWIHIC: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GGML 2-D im2col as a per-destination-element masked gather."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    col = e % ICKHKW
    rest = e // ICKHKW
    iow = rest % OW
    rest2 = rest // OW
    ioh = rest2 % OH
    in_ = rest2 // OH

    ikw = col % KW
    c2 = col // KW
    ikh = c2 % KH
    iic = c2 // KH

    iiw = iow * S0 + ikw * D0 - P0
    iih = ioh * S1 + ikh * D1 - P1
    in_bounds = (iiw >= 0) & (iiw < IW) & (iih >= 0) & (iih < IH)

    src_off = in_ * IWIHIC + iic * IWIH + iih * IW + iiw
    v = tl.load(SRC + src_off, mask=m & in_bounds, other=0.0)
    tl.store(DST + e, v, mask=m)


# ---------------------------------------------------------------------------
# POOL_2D (GGML_OP_POOL_2D): max/avg pooling as a per-destination-element gather
#   src (contiguous): planes of [IW, IH]; dst: planes of [PX, PY]
#   OP: 0 = MAX (ggml GGML_OP_POOL_MAX), 1 = AVG (divides by k0*k1)
# ---------------------------------------------------------------------------
@triton.jit
def pool2d_kernel(
    SRC,
    DST,
    n_elements: tl.constexpr,
    PX: tl.constexpr,
    PY: tl.constexpr,
    IW: tl.constexpr,
    IH: tl.constexpr,
    IWIH: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GGML 2-D pooling as a per-destination-element gather over the k0 x k1 window."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    ox = e % PX
    r = e // PX
    oy = r % PY
    plane = r // PY

    ix = ox * S0 - P0
    iy = oy * S1 - P1
    plane_off = plane * IWIH

    if OP == 0:
        acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for ky in range(K1):
        yy = iy + ky
        y_ok = (yy >= 0) & (yy < IH)
        for kx in range(K0):
            xx = ix + kx
            ok = y_ok & (xx >= 0) & (xx < IW)
            off = plane_off + yy * IW + xx
            if OP == 0:
                v = tl.load(SRC + off, mask=m & ok, other=float("-inf"))
                acc = tl.maximum(acc, v.to(tl.float32))
            else:
                v = tl.load(SRC + off, mask=m & ok, other=0.0)
                acc = acc + v.to(tl.float32)

    if OP != 0:
        acc = acc / (K0 * K1)
    tl.store(DST + e, acc, mask=m)


# ---------------------------------------------------------------------------
# CONV_2D direct (GGML_OP_CONV_2D): per-output-element reduction
#   kernel src0 (contiguous): ne=[KW, KH, IC, OC]
#   src1 (image, contiguous): ne=[W, H, IC, N]
#   dst (contiguous): ne=[OW, OH, OC, N]
#   dst(ox, oy, oc, n) = sum_{ic,ky,kx} src(sx, sy, ic, n) * kernel(kx, ky, ic, oc)
#     with sx = ox*S0 + kx*D0 - P0, sy = oy*S1 + ky*D1 - P1 (0 if out of bounds).
# ---------------------------------------------------------------------------
@triton.jit
def conv2d_kernel(
    KERN,
    SRC,
    DST,
    n_elements: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    OC: tl.constexpr,
    KW: tl.constexpr,
    KH: tl.constexpr,
    IC: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    WH: tl.constexpr,
    WHC: tl.constexpr,
    KWKH: tl.constexpr,
    KNL_N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Direct 2-D convolution as a per-destination-element reduction."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    ox = e % OW
    r = e // OW
    oy = r % OH
    r2 = r // OH
    oc = r2 % OC
    batch = r2 // OC

    kern_base = oc * KNL_N
    src_batch = batch * WHC

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in range(IC):
        for ky in range(KH):
            sy = oy * S1 + ky * D1 - P1
            y_ok = (sy >= 0) & (sy < H)
            for kx in range(KW):
                sx = ox * S0 + kx * D0 - P0
                ok = y_ok & (sx >= 0) & (sx < W)
                soff = src_batch + ic * WH + sy * W + sx
                koff = kern_base + ic * KWKH + ky * KW + kx
                sv = tl.load(SRC + soff, mask=m & ok, other=0.0)
                kv = tl.load(KERN + koff, mask=m, other=0.0)
                acc += sv * kv
    tl.store(DST + e, acc, mask=m)


# ---------------------------------------------------------------------------
# CROSS_ENTROPY_LOSS (GGML_OP_CROSS_ENTROPY_LOSS): scalar mean cross-entropy
#   src0 = logits [NC, NR], src1 = labels [NC, NR] (contiguous)
#   dst = -(1/NR) * sum_{i,j} labels[i,j] * log_softmax(logits[i])[j]
#   Single program loops over the NR rows (NR baked as constexpr).
# ---------------------------------------------------------------------------
@triton.jit
def cross_entropy_loss_kernel(
    S0,
    S1,
    DST,
    NR: tl.constexpr,
    NC: tl.constexpr,
    INV_NR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute the scalar mean cross-entropy loss over NR rows of NC classes."""
    offs = tl.arange(0, BLOCK)
    mask = offs < NC
    total = 0.0
    for i in range(NR):
        base = i * NC
        x = tl.load(S0 + base + offs, mask=mask, other=float("-inf"))
        lbl = tl.load(S1 + base + offs, mask=mask, other=0.0)
        mx = tl.max(x, axis=0)
        z = x - mx
        s = tl.sum(tl.exp(z), axis=0)
        log_sm = z - tl.log(s)
        contrib = tl.where(mask, lbl * log_sm, 0.0)
        total += tl.sum(contrib, axis=0)
    tl.store(DST, -total * INV_NR)
