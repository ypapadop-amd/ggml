# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Smoke tests for the ported Triton matmul kernel source files."""

from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
TRITON_DIR = KERNELS_DIR / "triton_kernels"


def test_transform_scripts_present():
    assert (TRITON_DIR / "matmul_aie2.mlir").is_file()
    assert (TRITON_DIR / "matmul_aie2p.mlir").is_file()


def test_bare_matmul_is_jit():
    import sys

    sys.path.insert(0, str(KERNELS_DIR))
    triton = pytest.importorskip("triton")
    from triton_kernels.matmul import bare_matmul

    assert isinstance(bare_matmul, triton.runtime.jit.JITFunction)
