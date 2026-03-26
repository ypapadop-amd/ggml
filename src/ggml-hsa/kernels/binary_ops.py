#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML binary operations."""

from functools import partial

from .kernel import Backend, KernelSpec


def _iron_binary_kernel(
    op_name: str,
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
):
    """Return wrapper for IRON binary operations matching the KernelFunction protocol.

    Parameters:
        op_name: Name of the binary operation.
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for binary ops).

    Returns:
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

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
        op_name: Name of the operation.

    Returns:
        KernelSpec configured for IRON backend.

    Raises:
        ValueError: If input_tensors does not contain exactly two tensors.

    """
    if len(input_tensors) != 2:
        msg = "Operation requires exactly two input tensors."
        raise ValueError(msg)

    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(_iron_binary_kernel, op_name=op_name),
    )


def _create_triton_kernel_config(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
):
    """
    Generate Triton vecadd kernel configuration.

    Parameters:
        arch (str): Target architecture (aie2, aie2p).
        input_tensors (list): Two input tensors.
        output_tensor (TensorDesc): Output tensor.
        op_params (bytearray): Operation parameters (unused).

    Returns:
        Tuple of (kernel_function, config_dict).
    """
    # Calculate total elements from output tensor
    n_elements = output_tensor.numel()

    # Choose block size based on architecture
    if arch == "aie2":
        block_size = min(256, n_elements)
    if arch == "aie2p":
        block_size = min(1024, n_elements)
    else:
        raise ValueError(f"Unsupported architecture for Triton kernel: {arch}")

    # Ensure block size divides n_elements evenly
    if n_elements % block_size != 0:
        for candidate in [512, 256, 128, 64, 32, 16]:
            if n_elements % candidate == 0:
                block_size = candidate
                break

    # Return constexpr parameters
    config = {
        "n_elements": n_elements,
        "block_size": block_size,
    }

    return config


def _make_triton_add_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
) -> KernelSpec:
    """
    Create a KernelSpec for Triton ADD operation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Two input tensors.
        output_tensor (TensorDesc): Output tensor.
        op_params (bytearray): Operation parameters.

    Returns:
        KernelSpec configured for TRITON backend.

    Raises:
        ValueError: If input_tensors does not contain exactly two tensors.
    """
    from .triton.vecadd import vecadd

    if len(input_tensors) != 2:
        raise ValueError("Operation requires exactly two input tensors.")

    config = _create_triton_kernel_config(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=vecadd,
        config=config,
    )


def ggml_op_add(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_ADD implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the ADD operation.

    """
    return _make_triton_add_kernel_spec(arch, input_tensors, output_tensor, op_params)
    # return _make_binary_kernel_spec(
    #    arch, input_tensors, output_tensor, op_params, "GGML_OP_ADD"
    # )


def ggml_op_sub(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SUB implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the SUB operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_SUB"
    )


def ggml_op_mul(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_MUL implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the MUL operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_MUL"
    )


def ggml_op_div(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_DIV implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the DIV operation.

    """
    return _make_binary_kernel_spec(
        arch, input_tensors, output_tensor, op_params, "GGML_OP_DIV"
    )
