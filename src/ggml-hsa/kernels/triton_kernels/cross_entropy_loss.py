# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_CROSS_ENTROPY_LOSS.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


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
