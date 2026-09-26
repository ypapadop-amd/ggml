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

    The elementwise Triton kernels are unmasked, and the grid is
    ``cdiv(n_elements, block)``, so a block that does not divide n_elements
    exactly leaves the last program running off the end of the tensor. Masking
    is not an alternative -- Triton-XDNA cannot compile a masked elementwise
    kernel, failing even when the mask is a semantic no-op, and upstream's own
    examples are unmasked over exact multiples of the block. So reject the
    shape: the ValueError is caught by build.py, which falls back to IRON.

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
