#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML clamp operation (GGML_OP_CLAMP)."""

from .kernel import Backend, KernelSpec


def ggml_op_clamp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_CLAMP.

    Clamps each element to [min_val, max_val].

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Min and max values.

    Returns:
        KernelSpec for the CLAMP operation.

    """
    from functools import partial

    from .iron_kernels.clamp import clamp

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_CLAMP",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            clamp,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
