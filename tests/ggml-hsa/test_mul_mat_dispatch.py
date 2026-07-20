# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Dispatch-selection tests for GGML_OP_MUL_MAT (Triton vs IRON)."""

import importlib
import sys
import types
from pathlib import Path

# mul_mat.py uses package-relative imports (from .kernel import ...), so it must
# be loaded with package context. Map a synthetic package onto the kernels dir
# and import through it; this avoids pulling in the heavy kernels/__init__.py
# (which imports build.py) while keeping Backend identity consistent.
KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
_PKG = "_ggml_hsa_kernels"
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [str(KERNELS_DIR)]
_pkg.__package__ = _PKG
sys.modules[_PKG] = _pkg

Backend = importlib.import_module(f"{_PKG}.kernel").Backend
ggml_op_mul_mat = importlib.import_module(f"{_PKG}.mul_mat").ggml_op_mul_mat
TensorDesc = importlib.import_module(f"{_PKG}.tensor_desc").TensorDesc


def _td(dtype, dim=256):
    return TensorDesc(dtype=dtype, shape=(dim, dim, 1, 1))


def test_triton_first_for_256_bf16():
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16"), _td("bf16")], _td("f32"), bytearray()
    )
    assert isinstance(specs, list)
    assert [s.backend for s in specs] == [Backend.TRITON, Backend.IRON]
    assert specs[0].config["transform_script"].endswith("matmul_aie2.mlir")


def test_iron_only_for_wrong_dtype():
    specs = ggml_op_mul_mat(
        "aie2", [_td("f32"), _td("f32")], _td("f32"), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]


def test_iron_only_for_wrong_shape():
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16", 128), _td("bf16", 128)], _td("f32", 128), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]
