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
    BLOCK_R: tl.constexpr,
):
    """Compute the scalar mean cross-entropy loss over NR rows of NC classes.

    One program processes BLOCK_R rows per iteration as a (BLOCK_R, BLOCK) tile, so the rows of a
    tile are reduced in parallel rather than one after another.
    """
    cols = tl.arange(0, BLOCK)
    col_ok = cols < NC
    total = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for r0 in range(0, NR, BLOCK_R):
        rows = r0 + tl.arange(0, BLOCK_R)
        row_ok = rows < NR
        ok = row_ok[:, None] & col_ok[None, :]
        offs = rows[:, None] * NC + cols[None, :]
        x = tl.load(S0 + offs, mask=ok, other=float("-inf"))
        # A row past NR would be all -inf, and its max - max a NaN; give it finite values instead.
        x = tl.where(row_ok[:, None], x, 0.0)
        lbl = tl.load(S1 + offs, mask=ok, other=0.0)
        z = x - tl.max(x, axis=1)[:, None]
        log_sm = z - tl.log(tl.sum(tl.exp(z), axis=1))[:, None]
        total += tl.sum(tl.where(ok, lbl * log_sm, 0.0), axis=1)
    tl.store(DST, -tl.sum(total, axis=0) * INV_NR)
