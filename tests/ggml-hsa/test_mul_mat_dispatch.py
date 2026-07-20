# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Dispatch-selection tests for GGML_OP_MUL_MAT (Triton vs IRON)."""

import pytest


@pytest.fixture(scope="module")
def dispatch(import_kernel_module):
    """Load the dispatch entry point and the types the assertions need."""
    Backend = import_kernel_module("kernel").Backend
    ggml_op_mul_mat = import_kernel_module("mul_mat").ggml_op_mul_mat
    TensorDesc = import_kernel_module("tensor_desc").TensorDesc

    def _td(dtype, dim=256):
        return TensorDesc(dtype=dtype, shape=(dim, dim, 1, 1))

    return Backend, ggml_op_mul_mat, _td


def test_triton_first_for_256_bf16(dispatch):
    Backend, ggml_op_mul_mat, _td = dispatch
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16"), _td("bf16")], _td("f32"), bytearray()
    )
    assert isinstance(specs, list)
    assert [s.backend for s in specs] == [Backend.TRITON, Backend.IRON]
    assert specs[0].config["transform_script"].endswith("matmul_aie2.mlir")


def test_iron_only_for_wrong_dtype(dispatch):
    Backend, ggml_op_mul_mat, _td = dispatch
    specs = ggml_op_mul_mat(
        "aie2", [_td("f32"), _td("f32")], _td("f32"), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]


def test_iron_only_for_wrong_shape(dispatch):
    Backend, ggml_op_mul_mat, _td = dispatch
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16", 128), _td("bf16", 128)], _td("f32", 128), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]
