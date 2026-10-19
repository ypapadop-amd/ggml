# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernels for GGML binary operations.

``vecadd`` is the unmasked NPU (Triton-XDNA) ADD kernel; ``bcast_binary_kernel`` is the
masked GPU kernel for ADD/SUB/MUL/DIV with GGML broadcasting of the second source.
The GPU kernels' only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


@triton.jit
def vecadd(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute C = A + B, BLOCK_SIZE_N elements per block.

    n_elements is unused: callers must size the grid so that BLOCK_SIZE_N evenly
    covers the vectors, since no bounds mask is applied against it here.

    Args:
        A: Pointer to the first input vector.
        B: Pointer to the second input vector.
        C: Pointer to the output vector.
        n_elements: Unused.
        BLOCK_SIZE_N: Number of elements processed per block.
    """
    pid = tl.program_id(0)  # block row id
    block_start = pid * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)

    # mask = offsets < n_elements    #AMK - in triton example, do we need?

    a_block = tl.load(A + offsets[:])
    b_block = tl.load(B + offsets[:])

    c_block = a_block + b_block

    tl.store(C + offsets[:], c_block)


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
