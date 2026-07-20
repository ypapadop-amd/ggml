#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

from functools import partial
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from .kernel import Backend, KernelSpec

# The ported Triton matmul is specialised to a single square problem size,
# matching the verbatim transform scripts (see the design doc). Only nodes of
# exactly this shape/dtype are routed to Triton; everything else uses IRON.
_TRITON_MATMUL_DIM = 256


def _matches_triton_matmul_profile(input_tensors: list, output_tensor) -> bool:
    """Return True if the node is the exact profile the Triton matmul supports.

    The profile is: two bf16 inputs, one f32 output, all operands square with
    leading two dims equal to _TRITON_MATMUL_DIM, higher dims trivial (== 1),
    and all operands contiguous.

    Args:
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.

    Returns:
        True if the node matches the fixed Triton profile, False otherwise.
    """
    if len(input_tensors) != 2:
        return False
    tensors = [*input_tensors, output_tensor]
    if not all(getattr(t, "contiguous", True) for t in tensors):
        return False
    if any(np.dtype(t.dtype) != np.dtype(bfloat16) for t in input_tensors):
        return False
    if np.dtype(output_tensor.dtype) != np.dtype(np.float32):
        return False
    d = _TRITON_MATMUL_DIM
    for t in tensors:
        shape = tuple(t.shape)
        if shape[0] != d or shape[1] != d:
            return False
        if any(s != 1 for s in shape[2:]):
            return False
    return True


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
    """Create the TRITON-backend KernelSpec for the fixed 256x256x256 bf16 matmul.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.

    Returns:
        KernelSpec configured for the TRITON backend.

    Raises:
        ValueError: If the tensors do not match the fixed 256x256x256 bf16->f32
            profile (raised lazily when the returned compile function is invoked).
    """
    dim = _TRITON_MATMUL_DIM

    def _compile(arch=arch, input_tensors=input_tensors, output_tensor=output_tensor):
        # Imports and tensor creation are deferred so any failure is caught by
        # the try/except fallback in build.py, mirroring the ADD Triton spec.
        import torch
        import triton

        from .triton_kernels.matmul import bare_matmul
        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device

        if not _matches_triton_matmul_profile(input_tensors, output_tensor):
            msg = "Triton matmul supports only 256x256x256 bf16->f32 contiguous nodes."
            raise ValueError(msg)

        m = n = k = dim
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
        grid = (triton.cdiv(m, dim), triton.cdiv(n, dim))
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
            BLOCK_SIZE_M=dim,
            BLOCK_SIZE_N=dim,
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
    """Return KernelSpecs for GGML_OP_MUL_MAT.

    IRON is always available (the general path). For nodes matching the fixed
    Triton profile (256x256x256 bf16->f32), the Triton spec is returned first so
    the build system tries it before falling back to IRON.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        List of KernelSpecs; Triton first iff the profile matches, else IRON only.
    """
    iron = _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor)
    if _matches_triton_matmul_profile(input_tensors, output_tensor):
        return [
            _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
            iron,
        ]
    return [iron]
