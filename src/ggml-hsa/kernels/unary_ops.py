#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry points for GGML unary operations.
"""

from .iron.unary_ops import unary_op


def ggml_op_sqr(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SQR implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_OP_SQR",
    )


def ggml_op_sqrt(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SQRT implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_log(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_LOG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_sin(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SIN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_cos(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_COS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_abs(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ABS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_ABS",
    )


def ggml_unary_op_sgn(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SGN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_SGN",
    )


def ggml_unary_op_neg(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_NEG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_NEG",
    )


def ggml_unary_op_step(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_STEP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_STEP",
    )


def ggml_unary_op_tanh(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_TANH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_elu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_relu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_RELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_RELU",
    )


def ggml_unary_op_sigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_quick(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU_QUICK implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_silu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SILU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_hardswish(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_HARDSWISH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_HARDSWISH",
    )


def ggml_unary_op_hardsigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_HARDSIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_HARDSIGMOID",
    )


def ggml_unary_op_exp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_EXP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_erf(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU_ERF implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_xielu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_XIELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_floor(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_FLOOR implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_FLOOR",
    )


def ggml_unary_op_ceil(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_CEIL implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_CEIL",
    )


def ggml_unary_op_round(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ROUND implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_ROUND",
    )


def ggml_unary_op_trunc(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_TRUNC implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_TRUNC",
    )
