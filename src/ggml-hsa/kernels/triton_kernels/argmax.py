# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_ARGMAX.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


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
