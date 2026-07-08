#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML 2D pooling operation (GGML_OP_POOL_2D)."""

from .kernel import Backend, KernelSpec


def ggml_op_pool_2d(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_POOL_2D.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: {op, k0, k1, s0, s1, p0, p1} as 7 x int32.

    Returns:
        KernelSpec for the POOL_2D operation.

    """
    from functools import partial

    from .iron_kernels.pool_2d import pool_2d

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_POOL_2D",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            pool_2d,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
