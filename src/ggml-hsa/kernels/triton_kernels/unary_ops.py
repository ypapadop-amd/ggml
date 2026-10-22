# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernels for GGML unary operations.

``relu`` is the unmasked NPU (Triton-XDNA) RELU kernel; ``relu_kernel`` is the masked GPU
RELU kernel. The GPU kernels' only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

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

    n_elements is unused: this kernel applies no bounds mask, so callers must
    size the grid so BLOCK_SIZE_N covers the vector exactly. A mask is not an
    option -- Triton-XDNA cannot compile one (see
    spec_utils.elementwise_block_size), so the caller rejects other sizes.

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
