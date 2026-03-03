#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry point for the GGML clamp operation (GGML_OP_CLAMP).
"""

from .iron.clamp import clamp
from .kernel import Backend, KernelSpec


def ggml_op_clamp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """
    GGML_OP_CLAMP implementation.

    Clamps each element of the input tensor to the range [min_val, max_val].
    output[i] = max(min_val, min(input[i], max_val))

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters containing min and max values.

    Returns:
        KernelSpec for the CLAMP operation.
    """
    return KernelSpec(
        backend=Backend.IRON,
        function=clamp,
    )
