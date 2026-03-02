#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

from .iron.binary_ops import ggml_op_binary


def ggml_op_add(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_ADD implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_binary(
        arch=arch,
        op_name="GGML_OP_ADD",
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_op_sub(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SUB implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_binary(
        arch=arch,
        op_name="GGML_OP_SUB",
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_op_mul(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_MUL implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_binary(
        arch=arch,
        op_name="GGML_OP_MUL",
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_op_div(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_DIV implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_binary(
        arch=arch,
        op_name="GGML_OP_DIV",
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )
