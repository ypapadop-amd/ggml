#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

from .iron.gemm import gemm


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_MUL_MAT implemetation.

    Args:
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        input_tensors (list): List of two input tensors (A and B).
        output_tensor: Output tensor (C).
        op_params (bytearray): Operation-specific parameters as a bytearray.
    """

    return gemm(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )
