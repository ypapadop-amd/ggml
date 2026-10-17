# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_COUNT_EQUAL.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# COUNT_EQUAL (GGML_OP_COUNT_EQUAL): number of elementwise-equal entries
#   dst (I64 scalar) = sum_i (src0[i] == src1[i]) over the N contiguous I32
#   elements. A single program loops over ceil(N / BLOCK) tiles (N and the
#   tile count baked as constexprs); masked lanes never contribute.
# ---------------------------------------------------------------------------
@triton.jit
def count_equal_kernel(
    S0,
    S1,
    DST,
    N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Count elementwise-equal entries between two N-length I32 tensors -> I64 scalar."""
    # Reduce in int32 (N is small; the equal-count fits) and widen only the final
    # scalar to int64 for the store, mirroring the fp32 reductions used by the
    # other GPU kernels. Masked-out lanes load mismatched sentinels so they never
    # count as equal.
    total = 0
    for t in range(NUM_TILES):
        offs = t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        a = tl.load(S0 + offs, mask=mask, other=0)
        b = tl.load(S1 + offs, mask=mask, other=-1)
        eq = tl.where(a == b, 1, 0)
        total += tl.sum(eq, axis=0)
    tl.store(DST, total.to(tl.int64))
