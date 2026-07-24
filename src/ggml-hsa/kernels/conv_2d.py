#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML direct 2D convolution operation (GGML_OP_CONV_2D)."""

from .kernel import Backend, KernelSpec


def ggml_op_conv_2d(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_CONV_2D.

    Args:
        arch: Target architecture.
        input_tensors: [kernel src0 (KW, KH, IC, OC), image src1 (IW, IH, IC, N)].
        output_tensor: Output tensor, shape (OW, OH, OC, N).
        op_params: {s0, s1, p0, p1, d0, d1} as 6 x int32.

    Returns:
        KernelSpec for the CONV_2D operation.

    """
    from functools import partial

    from .iron_kernels.conv_2d import conv_2d

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_CONV_2D",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            conv_2d,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
