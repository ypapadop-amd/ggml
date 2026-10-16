# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_SOFT_MAX.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


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
