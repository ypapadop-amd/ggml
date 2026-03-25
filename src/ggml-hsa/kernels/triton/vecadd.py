# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl


@triton.jit
def vecadd(
    A,
    B,
    C,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)  # block row id
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    a_block = tl.load(A + offsets[:])
    b_block = tl.load(B + offsets[:])

    c_block = a_block + b_block

    tl.store(C + offsets[:], c_block)
