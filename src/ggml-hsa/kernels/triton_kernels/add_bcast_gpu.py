# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernel for GGML_OP_ADD on AMD GPUs with GGML broadcasting of the second source."""

import triton
import triton.language as tl


@triton.jit
def add_bcast_gpu(
    A,
    B,
    C,
    NE0: tl.constexpr,
    NE1: tl.constexpr,
    NE2: tl.constexpr,
    NE3: tl.constexpr,
    NB0: tl.constexpr,
    NB1: tl.constexpr,
    NB2: tl.constexpr,
    NB3: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute C = A + B, repeating B over C's shape, BLOCK_SIZE_N elements per block.

    A and C are contiguous with shape (NE0, NE1, NE2, NE3); B is contiguous with shape
    (NB0, NB1, NB2, NB3), where each NBi divides NEi (ggml_can_repeat). Element (i0, i1, i2, i3)
    of C reads element (i0 % NB0, i1 % NB1, i2 % NB2, i3 % NB3) of B. Accesses are masked, so the
    element count need not be a multiple of BLOCK_SIZE_N.

    Args:
        A: Pointer to the first source, shaped like C.
        B: Pointer to the broadcast second source.
        C: Pointer to the destination.
        NE0: Innermost dimension of A and C.
        NE1: Second dimension of A and C.
        NE2: Third dimension of A and C.
        NE3: Outermost dimension of A and C.
        NB0: Innermost dimension of B.
        NB1: Second dimension of B.
        NB2: Third dimension of B.
        NB3: Outermost dimension of B.
        BLOCK_SIZE_N: Number of elements processed per block.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < NE0 * NE1 * NE2 * NE3

    i0 = offsets % NE0
    rest = offsets // NE0
    i1 = rest % NE1
    rest = rest // NE1
    i2 = rest % NE2
    i3 = rest // NE2
    b_offsets = (((i3 % NB3) * NB2 + (i2 % NB2)) * NB1 + (i1 % NB1)) * NB0 + (i0 % NB0)

    a = tl.load(A + offsets, mask=mask)
    b = tl.load(B + b_offsets, mask=mask)
    tl.store(C + offsets, (a + b).to(C.dtype.element_ty), mask=mask)
