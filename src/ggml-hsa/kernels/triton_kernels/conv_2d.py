# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_CONV_2D.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# CONV_2D direct (GGML_OP_CONV_2D): per-output-element reduction
#   kernel src0 (contiguous): ne=[KW, KH, IC, OC]
#   src1 (image, contiguous): ne=[W, H, IC, N]
#   dst (contiguous): ne=[OW, OH, OC, N]
#   dst(ox, oy, oc, n) = sum_{ic,ky,kx} src(sx, sy, ic, n) * kernel(kx, ky, ic, oc)
#     with sx = ox*S0 + kx*D0 - P0, sy = oy*S1 + ky*D1 - P1 (0 if out of bounds).
# ---------------------------------------------------------------------------
@triton.jit
def conv2d_kernel(
    KERN,
    SRC,
    DST,
    n_elements: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    OC: tl.constexpr,
    KW: tl.constexpr,
    KH: tl.constexpr,
    IC: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    WH: tl.constexpr,
    WHC: tl.constexpr,
    KWKH: tl.constexpr,
    KNL_N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Direct 2-D convolution as a per-destination-element reduction."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    ox = e % OW
    r = e // OW
    oy = r % OH
    r2 = r // OH
    oc = r2 % OC
    batch = r2 // OC

    kern_base = oc * KNL_N
    src_batch = batch * WHC

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for ic in range(IC):
        for ky in range(KH):
            sy = oy * S1 + ky * D1 - P1
            y_ok = (sy >= 0) & (sy < H)
            for kx in range(KW):
                sx = ox * S0 + kx * D0 - P0
                ok = y_ok & (sx >= 0) & (sx < W)
                soff = src_batch + ic * WH + sy * W + sx
                koff = kern_base + ic * KWKH + ky * KW + kx
                sv = tl.load(SRC + soff, mask=m & ok, other=0.0)
                kv = tl.load(KERN + koff, mask=m, other=0.0)
                acc += sv * kv
    tl.store(DST + e, acc, mask=m)
