#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML im2col operation (GGML_OP_IM2COL)."""

from .kernel import Backend, KernelSpec


def ggml_op_im2col(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_IM2COL.

    Args:
        arch: Target architecture.
        input_tensors: [kernel src0, image src1].
        output_tensor: Output tensor.
        op_params: {s0, s1, p0, p1, d0, d1, is_2D} as 7 x int32.

    Returns:
        KernelSpec for the IM2COL operation.

    """
    from functools import partial

    from .iron_kernels.im2col import im2col

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_IM2COL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            im2col,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
