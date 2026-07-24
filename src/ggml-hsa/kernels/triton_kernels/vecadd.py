# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernel for vector addition: C = A + B."""

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
