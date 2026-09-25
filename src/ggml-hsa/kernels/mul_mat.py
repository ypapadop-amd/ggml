#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

import os
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
# Smallest K the transform tiling accepts; K must also be a power of two.
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

        if any(not t.contiguous for t in (*input_tensors, output_tensor)):
            msg = "Non-contiguous tensors detected."
            raise ValueError(msg)

        # The transform scripts are the bf16 matmul recipe from the upstream
        # matmul_bf16_m64_n64_k64 example (pack=[4,4,8] / [8,8,8], accum=f32).
        # Any other input dtype would be lowered with bf16 packing and compute
        # the wrong result, so restrict to what the script actually implements.
        # IRON handles the i8/i16 variants.
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

        # Same M/N/K compatibility checks the IRON path makes (gemm.py). Without
        # them a malformed descriptor compiles a kernel that writes an (m, n)
        # result into a differently shaped C.
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

        block_m = _BLOCK_M_BY_ARCH.get(arch, _DEFAULT_BLOCK_M)
        block_n = _BLOCK_N
        if m % block_m != 0 or n % block_n != 0:
            msg = (
                f"M={m} not divisible by {block_m} or N={n} not divisible by "
                f"{block_n} for {arch}."
            )
            raise ValueError(msg)

        # K is constrained too: the transform tiles the packed K dim by 8
        # (= 64 raw elements) and the kernel loads exactly k unmasked. The
        # usable set is narrower than "divisible by 64" -- swept on aie2 at
        # M=N=256, K in {128, 256, 512} compiles, while 32 and 64 fail in aircc
        # and 96, 192, 320 and 384 fail in Triton itself (tl.dot needs a
        # power-of-two K). So require a power of two, at least 128.
        if k < _MIN_BLOCK_K or (k & (k - 1)) != 0:
            msg = (
                f"K={k} must be a power of two >= {_MIN_BLOCK_K} for the "
                f"Triton MUL_MAT."
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
        # MEASURED ON DEVICE (aie2/NPU1), bf16 256x256x256. Against GGML
        # buffers this spec is wrong -- 35086/65536 elements off, worst rel
        # 0.47, where IRON on the same test and build tree gives 0/65536 off --
        # but the arithmetic is sound. Fed operands laid out the way the kernel
        # actually reads them (A as [M,K], B as [K,N], C as [M,N], all
        # row-major), the same PDI returns 0/65536 off at worst rel 5.6e-07.
        # So at that shape and dtype the operand layout accounts for the whole
        # error, and the compute is sound. That is one shape on one arch, not a
        # proof for all of them: if a layout fix does not make some other shape
        # correct, do not assume layout was the only thing wrong there.
        #
        # It cannot be fixed from here. The GGML layout needs a transposed
        # operand, i.e. a dimension of stride 1 element, and for bf16 that is
        # 2 bytes while the shim DMA requires strides divisible by 4. Verified
        # by bisection: the identical design with a 2-element (4-byte) stride
        # compiles, the 1-element (2-byte) one does not. Upstream's transposed
        # example avoids this only by being f32, where 1 element is 4 bytes.
        #
        # The fix is to hand the kernel contiguous operands. Reassociating as
        # C_stored[n][m] = sum_k B_stored[n][k] * A_stored[m][k] leaves src1 and
        # dst contiguous and needs only src0 transposed -- and src0 is the
        # weight matrix, which is constant, so it can be transposed once in the
        # backend's pre-processing pass rather than per dispatch. That is a
        # backend change, not a kernel one.
        #
        # Until then treat this spec as non-functional: when it is reached it
        # turns a clean "unsupported" into a silently wrong answer.
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
    """Return KernelSpecs for GGML_OP_MUL_MAT (IRON only unless Triton opted in).

    Only the IRON spec is returned by default. The Triton MUL_MAT is measured
    wrong on device (see _make_triton_matmul_kernel_spec), so offering it as an
    automatic fallback would turn an IRON compile failure -- the exact case that
    selects it -- into a silently incorrect result instead of a clean
    "unsupported". Set ``GGML_HSA_ENABLE_TRITON_MUL_MAT=1`` to append it for
    benchmarking or for work on the kernel itself.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        The IRON KernelSpec, plus the Triton one when it is explicitly enabled.
        Order within the list is still subject to CompilerConfig.compilers /
        ``GGML_HSA_JIT_COMPILER_ORDER``.
    """
    iron_spec = _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor)
    if os.environ.get("GGML_HSA_ENABLE_TRITON_MUL_MAT", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return [iron_spec]

    return [
        iron_spec,
        _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
    ]
