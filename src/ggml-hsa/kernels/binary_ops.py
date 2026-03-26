#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML binary operations (GGML_OP_ADD, GGML_OP_SUB,
GGML_OP_MUL, GGML_OP_DIV).
"""

from functools import partial

from .kernel import Backend, KernelSpec


def _iron_binary_kernel(
    op_name: str,
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
):
    """Wrapper for IRON binary operations matching the KernelFunction protocol.

    Parameters
    ----------
        op_name: Name of the binary operation.
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for binary ops).

    Returns
    -------
        MLIR module for the binary operation.

    """
    from .iron.binary_ops import binary_op

    return binary_op(
        arch=arch,
        op_name=op_name,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def _make_binary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
    op_name: str,
) -> KernelSpec:
    """Create a KernelSpec for a binary operation.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
        op_name: Name of the operation.

    Returns
    -------
        KernelSpec configured for IRON backend.

    Raises
    ------
        ValueError: If input_tensors does not contain exactly two tensors.

    """
    if len(input_tensors) != 2:
        raise ValueError("Operation requires exactly two input tensors.")

    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(_iron_binary_kernel, op_name=op_name),
    )


def ggml_op_add(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_ADD implementation.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns
    -------
        KernelSpec for the ADD operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_ADD"
    )


def ggml_op_sub(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SUB implementation.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns
    -------
        KernelSpec for the SUB operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_SUB"
    )


def ggml_op_mul(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_MUL implementation.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns
    -------
        KernelSpec for the MUL operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_MUL"
    )


def ggml_op_div(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_DIV implementation.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns
    -------
        KernelSpec for the DIV operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_DIV"
    )
