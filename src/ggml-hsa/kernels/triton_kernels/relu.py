# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernel for the RELU activation: Y = max(X, 0)."""

import triton
import triton.language as tl


@triton.jit
def relu(
    X,
    Y,
    n_elements: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Compute Y = max(X, 0), BLOCK_SIZE_N elements per block.

    n_elements is unused: callers must size the grid so that BLOCK_SIZE_N evenly
    covers the vector, since no bounds mask is applied against it here.

    Args:
        X: Pointer to the input vector.
        Y: Pointer to the output vector.
        n_elements: Unused.
        BLOCK_SIZE_N: Number of elements processed per block.
    """
    pid = tl.program_id(0)  # block row id
    block_start = pid * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)

    x_block = tl.load(X + offsets[:])
    y_block = tl.maximum(x_block, 0.0)
    tl.store(Y + offsets[:], y_block)
