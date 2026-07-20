#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

import numpy as np

from .kernel import Backend, KernelSpec


def _make_triton_gpu_mul_mat_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for MUL_MAT.

    GGML MUL_MAT computes dst = src0 @ src1^T with
        src0 (A): ne=[K, M] (row-major -> logically (M, K))
        src1 (B): ne=[K, N] (row-major -> logically (N, K))
        dst     : ne0=M, ne1=N, dst(im, in) = sum_k A(im, k) * B(in, k)

    Only the plain 2-D FP32 case (no batching / broadcasting) is handled on the GPU; other cases
    raise ValueError so the op is reported unsupported. All dims/block sizes are baked as Triton
    constexprs; the only runtime kernel args are the src0, src1 and dst pointers.
    """
    import triton

    from .triton_kernels.gpu_ops import gemm_kernel

    src0, src1 = input_tensors[0], input_tensors[1]

    # only contiguous, non-batched FP32 matmul is supported on the Triton GPU path
    f32 = np.dtype(np.float32)
    if src0.dtype != f32 or src1.dtype != f32 or output_tensor.dtype != f32:
        msg = "Triton GPU MUL_MAT supports FP32 only."
        raise ValueError(msg)
    if any(not t.contiguous for t in (src0, src1, output_tensor)):
        msg = "Triton GPU MUL_MAT requires contiguous tensors."
        raise ValueError(msg)
    if src0.shape[2] != 1 or src0.shape[3] != 1 or src1.shape[2] != 1 or src1.shape[3] != 1:
        msg = "Triton GPU MUL_MAT supports 2-D (non-batched) matmul only."
        raise ValueError(msg)

    k = int(src0.shape[0])
    m = int(src0.shape[1])
    n = int(src1.shape[1])
    if int(src1.shape[0]) != k:
        msg = "Triton GPU MUL_MAT: src0/src1 inner dimension mismatch."
        raise ValueError(msg)

    block_m = 64
    block_n = 64
    block_k = 32
    num_pid_m = triton.cdiv(m, block_m)
    num_pid_n = triton.cdiv(n, block_n)
    num_programs = num_pid_m * num_pid_n

    signature = {
        "A": "*fp32",
        "B": "*fp32",
        "C": "*fp32",
        "M": "constexpr",
        "N": "constexpr",
        "K": "constexpr",
        "BM": "constexpr",
        "BN": "constexpr",
        "BK": "constexpr",
        "NUM_PID_N": "constexpr",
    }
    config = {
        "triton_fn": gemm_kernel,
        "signature": signature,
        "constexprs": {
            "M": m,
            "N": n,
            "K": k,
            "BM": block_m,
            "BN": block_n,
            "BK": block_k,
            "NUM_PID_N": num_pid_n,
        },
        "num_warps": 4,
        "num_programs": num_programs,
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=gemm_kernel,
        config=config,
    )


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_MUL_MAT.

    Parameters:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters.

    Returns:
        KernelSpec for the MUL_MAT operation.

    """
    from functools import partial

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_mul_mat_kernel_spec(arch, input_tensors, output_tensor)

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
