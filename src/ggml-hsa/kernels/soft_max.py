#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from .iron.softmax import softmax


def ggml_op_soft_max(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_OP_SOFT_MAX implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of 1-3 input tensors:
            - input_tensors[0]: Input tensor (required)
            - input_tensors[1]: Mask tensor (optional)
            - input_tensors[2]: Sink tensor (optional)
        output_tensor: Output tensor.
        op_params (bytearray): Operation parameters (scale, max_bias).
    """

    return softmax(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )
