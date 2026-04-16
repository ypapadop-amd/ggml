#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML binary operations."""

from pathlib import Path

from .kernel import Backend, KernelSpec


def _make_iron_binary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_name: str,
) -> KernelSpec:
    """Create a KernelSpec for a binary operation targeting the IRON backend.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_name: Name of the operation.

    Returns:
        KernelSpec configured for IRON backend.

    Raises:
        ValueError: If input_tensors does not contain exactly two tensors.

    """
    from functools import partial

    from .iron.binary_ops import binary_op

    if len(input_tensors) != 2:
        msg = "Operation requires exactly two input tensors."
        raise ValueError(msg)

    return KernelSpec(
        backend=Backend.IRON,
        op_name=op_name,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            binary_op,
            op_name=op_name,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )


def _make_triton_add_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
) -> KernelSpec:
    """Create a KernelSpec for ADD operation targeting the TRITON backend.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Two input tensors.
        output_tensor (TensorDesc): Output tensor.

    Returns:
        KernelSpec configured for TRITON backend.

    Raises:
        ValueError: If input_tensors does not contain exactly two tensors.
    """
    from functools import partial

    import torch
    import triton

    from .triton.utils import numpy_dtype_to_torch
    from .triton.vecadd import vecadd

    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)

    n_elements = output_tensor.numel()

    # Choose block size based on architecture
    if arch in ["aie2", "aie2p"]:
        block_size = min(1024, n_elements)
    else:
        msg = f"Unsupported architecture for Triton kernel: {arch}"
        raise ValueError(msg)

    # Ensure block size divides n_elements evenly
    if n_elements % block_size != 0:
        for candidate in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
            if n_elements % candidate == 0:
                block_size = candidate
                break

    device = "cpu"
    grid = (triton.cdiv(n_elements, block_size),)
    a = torch.randn(
        n_elements,
        device=device,
        dtype=numpy_dtype_to_torch(input_tensors[0].dtype),
    )
    b = torch.randn(
        n_elements,
        device=device,
        dtype=numpy_dtype_to_torch(input_tensors[1].dtype),
    )
    c = torch.empty(
        n_elements,
        device=device,
        dtype=numpy_dtype_to_torch(output_tensor.dtype),
    )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            vecadd[grid], A=a, B=b, C=c, n_elements=n_elements, BLOCK_SIZE_N=block_size
        ),
        config={
            "transform_script": str(
                Path(__file__).parent / "triton" / f"vecadd_{arch}.mlir"
            ),
        },
    )


def ggml_op_add(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """GGML_OP_ADD implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for ADD, but required
            by the dispatch interface).

    Returns:
        KernelSpec for the ADD operation.

    """
    return [
        _make_iron_binary_kernel_spec(
            arch, input_tensors, output_tensor, "GGML_OP_ADD"
        ),
        _make_triton_add_kernel_spec(arch, input_tensors, output_tensor),
    ]


def ggml_op_sub(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_SUB implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for SUB, but required
            by the dispatch interface).

    Returns:
        KernelSpec for the SUB operation.

    """
    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_SUB"
    )


def ggml_op_mul(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_MUL implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for MUL, but required
            by the dispatch interface).

    Returns:
        KernelSpec for the MUL operation.

    """
    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_MUL"
    )


def ggml_op_div(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_DIV implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for DIV, but required
            by the dispatch interface).

    Returns:
        KernelSpec for the DIV operation.

    """
    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_DIV"
    )
