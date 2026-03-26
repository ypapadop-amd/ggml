#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML unary operations."""

from functools import partial

from .iron.unary_ops import unary_op
from .kernel import Backend, KernelSpec


def _iron_unary_kernel(
    op_name: str,
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
):
    """Return rapper for IRON unary operations matching the KernelFunction protocol.

    Parameters:
        op_name: Name of the unary operation.
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for unary ops).

    Returns:
        MLIR module for the unary operation.

    """
    return unary_op(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name=op_name,
    )


def _make_unary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
    op_name: str,
) -> KernelSpec:
    """Create a KernelSpec for a unary operation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
        op_name: Name of the unary operation.

    Returns:
        KernelSpec configured for IRON backend.

    """
    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(_iron_unary_kernel, op_name=op_name),
    )


def ggml_op_sqr(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SQR implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SQR operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_SQR"
    )


def ggml_op_sqrt(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SQRT implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SQRT operation.

    """
    raise NotImplementedError


def ggml_op_log(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_LOG implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the LOG operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_LOG"
    )


def ggml_op_sin(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SIN implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SIN operation.

    """
    raise NotImplementedError


def ggml_op_cos(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_COS implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the COS operation.

    """
    raise NotImplementedError


def ggml_unary_op_abs(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_ABS implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the ABS operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_ABS"
    )


def ggml_unary_op_sgn(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_SGN implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SGN operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_SGN"
    )


def ggml_unary_op_neg(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_NEG implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the NEG operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_NEG"
    )


def ggml_unary_op_step(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_STEP implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the STEP operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_STEP"
    )


def ggml_unary_op_tanh(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_TANH implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the TANH operation.

    """
    raise NotImplementedError


def ggml_unary_op_elu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_ELU implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the ELU operation.

    """
    raise NotImplementedError


def ggml_unary_op_relu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_RELU implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the RELU operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_RELU"
    )


def ggml_unary_op_sigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_SIGMOID implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SIGMOID operation.

    """
    raise NotImplementedError


def ggml_unary_op_gelu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_GELU implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the GELU operation.

    """
    raise NotImplementedError


def ggml_unary_op_gelu_quick(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_GELU_QUICK implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the GELU_QUICK operation.

    """
    raise NotImplementedError


def ggml_unary_op_silu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_SILU implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SILU operation.

    """
    raise NotImplementedError


def ggml_unary_op_hardswish(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_HARDSWISH implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the HARDSWISH operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_HARDSWISH"
    )


def ggml_unary_op_hardsigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_HARDSIGMOID implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the HARDSIGMOID operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_HARDSIGMOID"
    )


def ggml_unary_op_exp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_EXP implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor : Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the EXP operation.

    """
    raise NotImplementedError


def ggml_unary_op_gelu_erf(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_GELU_ERF implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the GELU_ERF operation.

    """
    raise NotImplementedError


def ggml_unary_op_xielu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_XIELU implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the XIELU operation.

    """
    raise NotImplementedError


def ggml_unary_op_floor(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_FLOOR implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the FLOOR operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_FLOOR"
    )


def ggml_unary_op_ceil(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_CEIL implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the CEIL operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_CEIL"
    )


def ggml_unary_op_round(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_ROUND implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the ROUND operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_ROUND"
    )


def ggml_unary_op_trunc(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_UNARY_OP_TRUNC implementation.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the TRUNC operation.

    """
    return _make_unary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_UNARY_OP_TRUNC"
    )
