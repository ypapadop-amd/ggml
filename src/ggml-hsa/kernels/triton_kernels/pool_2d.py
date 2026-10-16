# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_POOL_2D.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# POOL_2D (GGML_OP_POOL_2D): max/avg pooling as a per-destination-element gather
#   src (contiguous): planes of [IW, IH]; dst: planes of [PX, PY]
#   OP: 0 = MAX (ggml GGML_OP_POOL_MAX), 1 = AVG (divides by k0*k1)
# ---------------------------------------------------------------------------
@triton.jit
def pool2d_kernel(
    SRC,
    DST,
    n_elements: tl.constexpr,
    PX: tl.constexpr,
    PY: tl.constexpr,
    IW: tl.constexpr,
    IH: tl.constexpr,
    IWIH: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GGML 2-D pooling as a per-destination-element gather over the k0 x k1 window."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    ox = e % PX
    r = e // PX
    oy = r % PY
    plane = r // PY

    ix = ox * S0 - P0
    iy = oy * S1 - P1
    plane_off = plane * IWIH

    if OP == 0:
        acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for ky in range(K1):
        yy = iy + ky
        y_ok = (yy >= 0) & (yy < IH)
        for kx in range(K0):
            xx = ix + kx
            ok = y_ok & (xx >= 0) & (xx < IW)
            off = plane_off + yy * IW + xx
            if OP == 0:
                v = tl.load(SRC + off, mask=m & ok, other=float("-inf"))
                acc = tl.maximum(acc, v.to(tl.float32))
            else:
                v = tl.load(SRC + off, mask=m & ok, other=0.0)
                acc = acc + v.to(tl.float32)

    if OP != 0:
        acc = acc / (K0 * K1)
    tl.store(DST + e, acc, mask=m)
