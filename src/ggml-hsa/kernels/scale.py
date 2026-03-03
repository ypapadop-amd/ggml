#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry point for the GGML scale operation (GGML_OP_SCALE).
"""

from .iron.scale import scale


def ggml_op_scale(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SCALE implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params (op_params): Operation parameters.
    """

    return scale(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )
