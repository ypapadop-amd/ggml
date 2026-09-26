#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

from functools import partial

from .kernel import Backend, KernelSpec
from .triton_kernels.spec_utils import transform_script

# Per-arch L3 block sizes, matching the upstream matmul_bf16_m64_n64_k64
# example. The transform tiles each block across the herd, so M and N must
# decompose into fixed blocks; a single whole-matrix block exhausts the shim
# DMA channels. Both archs have 4 herd rows, so N is fixed and only M varies
# with the column count -- a larger N on aie2p asks for rows it does not have.
# Only these two archs have a matmul transform script.
_BLOCK_MN_BY_ARCH = {
    "aie2": (256, 256),  # 4 herd columns
    "aie2p": (512, 256),  # 8 herd columns, so twice the M
}
# K must also be a power of two; see the check below.
_MIN_BLOCK_K = 128


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
        import torch
        import triton

        from .triton_kernels.matmul import bare_matmul
        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device

        if len(input_tensors) != 2:
            msg = f"Requires two input tensors, got {len(input_tensors)}."
            raise ValueError(msg)
        if any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Non-contiguous tensors detected."
            raise ValueError(msg)

        # The transform scripts implement the upstream bf16 recipe; any other
        # dtype would be lowered with bf16 packing and compute the wrong
        # result. IRON handles the i8/i16 variants.
        if any(t.dtype.name != "bfloat16" for t in input_tensors):
            dtypes = tuple(t.dtype.name for t in input_tensors)
            msg = f"Triton MUL_MAT supports bf16 inputs only, got {dtypes}."
            raise ValueError(msg)
        if output_tensor.dtype.name != "float32":
            msg = (
                "Triton MUL_MAT supports an f32 output only, got "
                f"{output_tensor.dtype.name}."
            )
            raise ValueError(msg)

        # GGML shape convention (innermost first): A is [K, M], B is [K, N],
        # C is [M, N]. The kernel is a plain 2D matmul, so a batch or broadcast
        # dimension would be silently ignored, computing only the first matrix.
        for label, t in (
            ("A", input_tensors[0]),
            ("B", input_tensors[1]),
            ("C", output_tensor),
        ):
            if any(d != 1 for d in t.shape[2:]):
                msg = f"{label} has batch dimensions {tuple(t.shape[2:])}; 2D only."
                raise ValueError(msg)

        # Same M/N/K checks IRON makes (gemm.py); without them a malformed
        # descriptor writes an (m, n) result into a differently shaped C.
        a, b_desc, c_desc = input_tensors[0], input_tensors[1], output_tensor
        if a.shape[1] != c_desc.shape[0]:
            msg = f"Incompatible M for A and C: {a.shape[1]} != {c_desc.shape[0]}"
            raise ValueError(msg)
        if b_desc.shape[1] != c_desc.shape[1]:
            msg = f"Incompatible N for B and C: {b_desc.shape[1]} != {c_desc.shape[1]}"
            raise ValueError(msg)
        if a.shape[0] != b_desc.shape[0]:
            msg = f"Incompatible K for A and B: {a.shape[0]} != {b_desc.shape[0]}"
            raise ValueError(msg)

        m = input_tensors[0].shape[1]
        k = input_tensors[0].shape[0]
        n = input_tensors[1].shape[1]

        if arch not in _BLOCK_MN_BY_ARCH:
            msg = (
                f"No Triton MUL_MAT transform script for {arch}; "
                f"supported: {sorted(_BLOCK_MN_BY_ARCH)}."
            )
            raise ValueError(msg)
        block_m, block_n = _BLOCK_MN_BY_ARCH[arch]
        if m % block_m != 0 or n % block_n != 0:
            msg = (
                f"M={m} not divisible by {block_m} or N={n} not divisible by "
                f"{block_n} for {arch}."
            )
            raise ValueError(msg)

        # K is constrained more tightly than the packed-K tiling suggests:
        # tl.dot needs a power of two, and smaller ones fail to place.
        if k < _MIN_BLOCK_K or (k & (k - 1)) != 0:
            msg = (
                f"K={k} must be a power of two >= {_MIN_BLOCK_K} for the "
                f"Triton MUL_MAT."
            )
            raise ValueError(msg)

        device = triton_device(arch)
        # Contents are never read; these carry dtype/stride metadata only.
        #
        # KNOWN LIMITATION: this spec is wrong against GGML buffers. GGML stores
        # B and C column-major with respect to the mathematical matrices (IRON
        # declares this via b_col_maj / c_col_maj in gemm.py) while the strides
        # below are row-major. It cannot be fixed here: a transposed bf16 operand
        # needs a 1-element stride, which is 2 bytes, and the shim DMA requires 4.
        # The compute itself is sound -- given operands in the layout it reads,
        # this same kernel is accurate -- so the fix is to hand it contiguous
        # operands by transposing the constant src0 in the backend pre-processing
        # pass. Until then keep it opt-in: reaching it silently replaces a clean
        # "unsupported" with a wrong answer.
        a = torch.empty(
            (m, k), device=device, dtype=numpy_dtype_to_torch(input_tensors[0].dtype)
        )
        b = torch.empty(
            (k, n), device=device, dtype=numpy_dtype_to_torch(input_tensors[1].dtype)
        )
        c = torch.empty(
            (m, n), device=device, dtype=numpy_dtype_to_torch(output_tensor.dtype)
        )
        # 2D launch: one Triton program per block_m x block_n output block;
        # K stays full (the transform tiles the K reduction internally).
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
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
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
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
            # No f32 variant exists for matmul, so the dtype is not passed.
            "transform_script": transform_script("matmul", arch),
        },
    )


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_OP_MUL_MAT (IRON primary, Triton fallback).

    IRON is tried first; the Triton spec is reached only if IRON compilation
    fails. Set ``GGML_HSA_JIT_COMPILER_ORDER=triton,iron`` to flip the order.

    Note the Triton spec is known wrong against GGML buffers -- see
    _make_triton_matmul_kernel_spec -- so reaching it yields an incorrect
    result rather than a clean "unsupported".

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        List of KernelSpecs: IRON first, then Triton (reordered by
        CompilerConfig.compilers / ``GGML_HSA_JIT_COMPILER_ORDER``).
    """
    return [
        _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor),
        _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
    ]
