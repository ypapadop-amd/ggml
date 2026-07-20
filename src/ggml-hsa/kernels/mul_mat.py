#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

from functools import partial
from pathlib import Path

from .kernel import Backend, KernelSpec


def _make_iron_matmul_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create the IRON-backend KernelSpec for MUL_MAT (the general path).

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.

    Returns:
        KernelSpec configured for the IRON backend.
    """
    from .iron_kernels.gemm import gemm

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            gemm, arch=arch, input_tensors=input_tensors, output_tensor=output_tensor
        ),
    )


def _make_triton_matmul_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create the TRITON-backend KernelSpec for MUL_MAT.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.

    Returns:
        KernelSpec configured for the TRITON backend.

    Raises:
        ValueError: If the tensors are non-contiguous (raised lazily when the
            returned compile function is invoked).
    """

    def _compile(arch=arch, input_tensors=input_tensors, output_tensor=output_tensor):
        # Imports and tensor creation are deferred so any failure is caught by
        # the try/except fallback in build.py, mirroring the ADD Triton spec.
        # IRON is the primary backend (tried first); this Triton spec is the
        # fallback, reached only if IRON compilation fails.
        import torch

        from .triton_kernels.matmul import bare_matmul
        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device

        if any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Non-contiguous tensors detected."
            raise ValueError(msg)

        # GGML shape convention (innermost first): A is [K, M], B is [K, N],
        # C is [M, N]. The matmul reads A as [M, K] and B as [K, N] row-major.
        m = input_tensors[0].shape[1]
        k = input_tensors[0].shape[0]
        n = input_tensors[1].shape[1]

        device = triton_device(arch)
        a = torch.randn(
            (m, k), device=device, dtype=numpy_dtype_to_torch(input_tensors[0].dtype)
        )
        b = torch.randn(
            (k, n), device=device, dtype=numpy_dtype_to_torch(input_tensors[1].dtype)
        )
        c = torch.empty(
            (m, n), device=device, dtype=numpy_dtype_to_torch(output_tensor.dtype)
        )
        # bare_matmul is single-block: one block covers the whole [M, N] output.
        grid = (1, 1)
        return bare_matmul[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=m,
            BLOCK_SIZE_N=n,
            BLOCK_SIZE_K=k,
        )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
        config={
            "transform_script": str(
                Path(__file__).parent / "triton_kernels" / f"matmul_{arch}.mlir"
            ),
        },
    )


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_OP_MUL_MAT (IRON primary, Triton fallback).

    IRON is the general path and is tried first; the Triton spec is appended so
    the build system falls back to it only if IRON compilation fails, mirroring
    the ADD path. The Triton kernel derives its M/N/K from the tensor shapes and
    validates them lazily at compile time.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        List of KernelSpecs: IRON first, then Triton as a fallback.
    """
    return [
        _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor),
        _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
    ]
