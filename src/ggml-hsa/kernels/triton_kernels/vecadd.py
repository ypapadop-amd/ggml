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
    """Triton kernel for vector addition: C = A + B.

    Parameters:
        A: Pointer to input vector A
        B: Pointer to input vector B
        C: Pointer to output vector C
        n_elements: Total number of elements in the vectors
        BLOCK_SIZE_N: Number of elements processed by each block
    """
    pid = tl.program_id(0)  # block row id
    block_start = pid * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)

    # mask = offsets < n_elements    #AMK - in triton example, do we need?

    a_block = tl.load(A + offsets[:])
    b_block = tl.load(B + offsets[:])

    c_block = a_block + b_block

    tl.store(C + offsets[:], c_block)
