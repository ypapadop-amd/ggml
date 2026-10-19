# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton GPU kernel for GGML_OP_IM2COL.

The kernel's only runtime arguments are the tensor pointers (sources, then the
destination); every dimension, stride and block size is a ``tl.constexpr``, and the
launch geometry goes in the ``.hsaco.json`` sidecar (see ``build_triton.py``).
"""

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# IM2COL (GGML_OP_IM2COL, is_2D): masked gather of input patches
#   src1 (image, contiguous): ne=[IW, IH, IC, N]
#   dst (contiguous): ne=[IC*KH*KW, OW, OH, N]
#   dst[(in*OH*OW + ioh*OW + iow)*ICKHKW + iic*KH*KW + ikh*KW + ikw]
#       = image(iiw, iih, iic, in), with iiw = iow*S0 + ikw*D0 - P0,
#         iih = ioh*S1 + ikh*D1 - P1 (0 if out of bounds).
# ``KERN`` is the (unused) kernel/src0 pointer, present only so the kernarg
# order matches the dispatch (src0, src1, dst).
# ---------------------------------------------------------------------------
@triton.jit
def im2col_kernel(
    KERN,
    SRC,
    DST,
    n_elements: tl.constexpr,
    ICKHKW: tl.constexpr,
    KW: tl.constexpr,
    KH: tl.constexpr,
    OW: tl.constexpr,
    OH: tl.constexpr,
    IW: tl.constexpr,
    IH: tl.constexpr,
    IWIH: tl.constexpr,
    IWIHIC: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """GGML 2-D im2col as a per-destination-element masked gather."""
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    m = e < n_elements

    col = e % ICKHKW
    rest = e // ICKHKW
    iow = rest % OW
    rest2 = rest // OW
    ioh = rest2 % OH
    in_ = rest2 // OH

    ikw = col % KW
    c2 = col // KW
    ikh = c2 % KH
    iic = c2 // KH

    iiw = iow * S0 + ikw * D0 - P0
    iih = ioh * S1 + ikh * D1 - P1
    in_bounds = (iiw >= 0) & (iiw < IW) & (iih >= 0) & (iih < IH)

    src_off = in_ * IWIHIC + iic * IWIH + iih * IW + iiw
    v = tl.load(SRC + src_off, mask=m & in_bounds, other=0.0)
    tl.store(DST + e, v, mask=m)
