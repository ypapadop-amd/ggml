#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""
Top-level entry point for the GGML matrix multiplication operation (GGML_OP_MUL_MAT).
"""

from .iron.gemm import gemm
from .kernel import Backend, KernelSpec


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """
    GGML_OP_MUL_MAT implementation.

    Parameters:
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        input_tensors (list): List of two input tensors (A and B).
        output_tensor (TensorDesc): Output tensor (C).
        op_params (bytearray): Operation-specific parameters.

    Returns:
        KernelSpec for the MUL_MAT operation.
    """
    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=gemm,
    )
