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

# L3 block sizes for the Triton MUL_MAT, per arch. The transform script tiles
# each L3 block into 64x64 per-core L1 tiles across the AIE herd, so M/N must be
# decomposed into fixed blocks: a single whole-matrix block (grid=(1,1),
# BLOCK=m/n/k) exhausts the shim DMA channels and herd placement fails.
#
# The herd forall is tiled [16,16,0] on aie2 (pack=[4,4,8]) and [8,8,0] on aie2p
# (pack=[8,8,8]); dim0 maps to AIE columns and dim1 to rows. Both archs have 4
# rows, so N is pinned at 256 (256/pack/tile = 4) and only the column count
# differs: a 4x4 herd on aie2 (M=256) and an 8-column x 4-row herd on aie2p
# (M=512). N=512 on aie2p would ask for 8 rows and aircc rejects it ("row index
# (6) must be less than the number of rows in the device (6)"). This matches the
# upstream example, which raises only BLOCK_SIZE_M to 512 for aie2p:
# Triton-XDNA/examples/matmul_bf16_m64_n64_k64/matmul_bf16_m64_n64_k64.py
_DEFAULT_BLOCK_M = 256
_BLOCK_M_BY_ARCH = {"aie2p": 512}
_BLOCK_N = 256


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

        if any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Non-contiguous tensors detected."
            raise ValueError(msg)

        # GGML shape convention (innermost first): A is [K, M], B is [K, N],
        # C is [M, N]. The kernel is a plain 2D M x N matmul, so any batch or
        # broadcast dimension would be silently ignored, computing only the
        # first matrix and leaving the rest of C untouched. Reject those and let
        # dispatch fall back to IRON.
        for label, t in (
            ("A", input_tensors[0]),
            ("B", input_tensors[1]),
            ("C", output_tensor),
        ):
            if any(d != 1 for d in t.shape[2:]):
                msg = f"{label} has batch dimensions {tuple(t.shape[2:])}; 2D only."
                raise ValueError(msg)

        m = input_tensors[0].shape[1]
        k = input_tensors[0].shape[0]
        n = input_tensors[1].shape[1]

        block_m = _BLOCK_M_BY_ARCH.get(arch, _DEFAULT_BLOCK_M)
        block_n = _BLOCK_N
        if m % block_m != 0 or n % block_n != 0:
            msg = (
                f"M={m} not divisible by {block_m} or N={n} not divisible by "
                f"{block_n} for {arch}."
            )
            raise ValueError(msg)

        device = triton_device(arch)
        # Contents are never read: the kernel is compiled, not launched, so
        # these exist only to carry dtype/stride metadata to Triton.
        #
        # KNOWN LIMITATION (operand layout). GGML tensors are innermost-first
        # contiguous, so B with ne == (K, N) is N rows of K, i.e. the
        # mathematical K x N matrix stored COLUMN-major; likewise C. The IRON
        # path declares exactly that via b_col_maj=True / c_col_maj=True in
        # iron_kernels/gemm.py. The row-major strides baked in below therefore
        # do NOT describe the GGML buffers this kernel would run against.
        #
        # The obvious fix -- torch.empty_strided((k, n), (1, k)) and
        # ((m, n), (1, m)) -- does not compile: the shim DMA requires strides
        # divisible by 4 bytes, and a column-major bf16 operand has an innermost
        # stride of one element = 2 bytes, so aircc rejects the design with
        # "'aie.dma_bd' op Stride 1 is 1 elements * 2 bytes = 2 bytes, which is
        # not divisible by 4" (measured on aie2 and aie2p, all shapes).
        # Expressing the GGML layout needs the transpose handled inside the
        # transform script, not via operand strides.
        #
        # MEASURED ON DEVICE (aie2/NPU1): this kernel does NOT produce correct
        # results. bf16 256x256x256, same test and build tree, only the
        # dispatch order changed:
        #   GGML_HSA_JIT_COMPILER_ORDER=iron,triton -> 0/65536 elements off,
        #                                              worst rel 5.8e-07
        #   GGML_HSA_JIT_COMPILER_ORDER=triton,iron -> 35086/65536 off,
        #                                              worst rel 0.47
        # So the surrounding machinery is fine and this kernel is the fault.
        # The output does not match a simple transpose either, so the layout
        # above is necessary but not sufficient to explain it. Treat this spec
        # as non-functional: when it is reached it turns a clean "unsupported"
        # into a silently wrong answer.
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

    IRON is the general path and is tried first; the Triton spec is appended so
    the build system falls back to it only if IRON compilation fails, mirroring
    the ADD path. The Triton kernel derives its M/N/K from the tensor shapes and
    validates them lazily at compile time.

    Set ``GGML_HSA_JIT_COMPILER_ORDER=triton,iron`` to flip the order so Triton is
    tried first and IRON becomes the fallback (used to benchmark the Triton path).

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        List of KernelSpecs: IRON first, then Triton as a fallback (reordered by
        CompilerConfig.compilers / ``GGML_HSA_JIT_COMPILER_ORDER``).
    """
    return [
        _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor),
        _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
    ]
