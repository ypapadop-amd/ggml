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

    """
    from functools import partial

    from .iron.binary_ops import binary_op

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
        ValueError: If the tensors require broadcasting or are non-contiguous
            (raised lazily when the returned compile function is invoked).
    """
    n_elements = output_tensor.numel()

    def _compile(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        n_elements=n_elements,
    ):
        # All imports, grid specialisation, and tensor creation are deferred into
        # _compile so that any failure is caught by the try/except in build.py,
        # allowing the IRON fallback to be reached.

        import torch
        import triton

        from .triton.utils import numpy_dtype_to_torch
        from .triton.vecadd import vecadd

        broadcast = input_tensors[0].shape != input_tensors[1].shape
        if broadcast or any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Broadcasting or non-contiguous tensors detected."
            raise ValueError(msg)

        block_size = 1 << (min(1024, n_elements) - 1).bit_length()
        block_size = min(block_size, 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        device = "cpu"
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
        return vecadd[grid](
            A=a, B=b, C=c, n_elements=n_elements, BLOCK_SIZE_N=block_size
        )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
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
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)

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
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)

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
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)

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
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_DIV"
    )
