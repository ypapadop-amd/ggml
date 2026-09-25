# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Helpers for building Triton KernelSpecs.

Deliberately separate from ``triton_kernels.utils``: the helpers here run at
dispatch time, before a backend has been selected, so they must be importable
in an IRON-only environment. ``utils`` imports ``torch`` at module scope and is
therefore only safe to import from inside a KernelSpec's compile function.
"""

from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent


def transform_script(stem: str, arch: str, dtype=None) -> str:
    """Return the path to the Triton-XDNA transform script for an op.

    Scripts are named ``{stem}_{arch}.mlir``, with an ``_f32`` variant where one
    exists: the bf16 scripts pad with a bf16 zero, which aircc rejects for f32
    tensors. Pass ``dtype`` to select that variant; omit it for ops that have no
    f32 script (MUL_MAT).

    Args:
        stem: Op script stem, e.g. "vecadd", "relu", "matmul".
        arch: Target architecture, e.g. "aie2", "aie2p".
        dtype: Output dtype used to pick the padding variant, or None to always
            use the base script.

    Returns:
        Absolute path to the transform script, as a string.
    """
    suffix = "_f32" if dtype is not None and np.dtype(dtype) == np.float32 else ""
    return str(_SCRIPT_DIR / f"{stem}_{arch}{suffix}.mlir")


def elementwise_block_size(n_elements: int, max_block: int = 1024) -> int:
    """Round n_elements up to a power of two, capped at max_block.

    The elementwise Triton kernels (vecadd, relu) apply no bounds mask: every
    lane of every program loads and stores unconditionally. The launch grid is
    ``cdiv(n_elements, block)``, so unless the block divides n_elements exactly
    the last program runs off the end of the tensor -- e.g. n_elements=1000
    gives one 1024-lane program that touches 24 elements past the allocation.

    Masking is not an option: Triton-XDNA cannot compile a masked elementwise
    kernel. Adding ``mask=offsets < n_elements`` to the load and store crashes
    aircc ("Assertion 'this->_M_is_engaged()' failed" inside the AIR pipeline,
    no diagnostic), and it does so even when n_elements is an exact multiple of
    the block and the mask is a semantic no-op -- so it is the construct, not
    the shape. Upstream agrees: examples/relu/relu.py and examples/gelu/gelu.py
    are both unmasked with an unused n_elements constexpr, and their benchmarks
    only ever sweep exact multiples of BLOCK_SIZE.

    Rejecting the shape is therefore the only safe response. The caller runs
    inside a KernelSpec compile function, so the ValueError is caught by
    build.py and dispatch falls back to IRON, which has no such restriction.

    Args:
        n_elements: Number of elements the kernel covers.
        max_block: Largest permitted block size.

    Returns:
        A power of two in [1, max_block] that divides n_elements exactly.

    Raises:
        ValueError: If no such block size exists, i.e. n_elements is not a
            multiple of the selected power-of-two block.
    """
    block = 1 << (min(max_block, n_elements) - 1).bit_length()
    if n_elements % block != 0:
        msg = (
            f"n_elements={n_elements} is not a multiple of the {block}-element "
            f"block; the Triton elementwise kernels are unmasked and would "
            f"access {block - n_elements % block} elements out of bounds."
        )
        raise ValueError(msg)
    return block
