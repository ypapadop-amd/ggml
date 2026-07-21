#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry points for GGML binary operations."""

from pathlib import Path

from .kernel import Backend, KernelSpec, order_kernel_specs


def _validate_binary_inputs(input_tensors: list) -> None:
    """Validate that a binary operation has exactly two input tensors.

    Args:
        input_tensors: List of input tensors to validate.

    Raises:
        ValueError: If the number of input tensors is not exactly two.
    """
    if len(input_tensors) != 2:
        msg = f"Operation requires exactly two input tensors, got {len(input_tensors)}."
        raise ValueError(msg)


def _make_iron_binary_kernel_spec(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_name: str,
) -> KernelSpec:
    """Create an IRON-backend KernelSpec for a binary operation.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_name: Name of the GGML operation.

    Returns:
        KernelSpec configured for the IRON backend.
    """
    from functools import partial

    from .iron_kernels.binary_ops import binary_op

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
    """Create a TRITON-backend KernelSpec for ADD.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.

    Returns:
        KernelSpec configured for the TRITON backend.

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
        # _compile so that any failure is caught by the try/except in build.py.
        # IRON is the primary backend (tried first); this Triton spec is the
        # fallback, reached only if IRON compilation fails.

        import torch
        import triton

        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device
        from .triton_kernels.vecadd import vecadd

        broadcast = input_tensors[0].shape != input_tensors[1].shape
        if broadcast or any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Broadcasting or non-contiguous tensors detected."
            raise ValueError(msg)

        block_size = 1 << (min(1024, n_elements) - 1).bit_length()
        block_size = min(block_size, 1024)
        grid = (triton.cdiv(n_elements, block_size),)
        device = triton_device(arch)
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

    # The bf16 transform script (vecadd_{arch}.mlir) pads with a bf16 zero, which
    # aircc rejects for f32 tensors. Select an f32-padding variant for f32 inputs.
    import numpy as np

    script_stem = f"vecadd_{arch}"
    if np.dtype(output_tensor.dtype) == np.float32:
        script_stem += "_f32"

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ADD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
        config={
            "transform_script": str(
                Path(__file__).parent / "triton_kernels" / f"{script_stem}.mlir"
            ),
        },
    )


def ggml_op_add(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_OP_ADD (IRON primary, Triton fallback).

    IRON is tried first; the Triton spec is the fallback, reached only if IRON
    compilation fails. Setting ``GGML_HSA_PREFER_TRITON=1`` flips the order so
    Triton is tried first.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        List of KernelSpecs for the ADD operation.
    """
    _validate_binary_inputs(input_tensors)

    return order_kernel_specs(
        [
            _make_iron_binary_kernel_spec(
                arch, input_tensors, output_tensor, "GGML_OP_ADD"
            ),
            _make_triton_add_kernel_spec(arch, input_tensors, output_tensor),
        ]
    )


def ggml_op_sub(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SUB.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the SUB operation.
    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_SUB"
    )


def ggml_op_mul(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_MUL.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the MUL operation.
    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_MUL"
    )


def ggml_op_div(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_DIV.

    Args:
        arch: Target architecture.
        input_tensors: List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters (unused for elementwise ops but required
            by the dispatch interface).

    Returns:
        KernelSpec for the DIV operation.
    """
    _validate_binary_inputs(input_tensors)

    return _make_iron_binary_kernel_spec(
        arch, input_tensors, output_tensor, "GGML_OP_DIV"
    )
