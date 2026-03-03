#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry point for the GGML softmax operation (GGML_OP_SOFT_MAX).

Returns a KernelSpec specifying the compilation backend and kernel function.
"""

from .iron.softmax import softmax
from .kernel import Backend, KernelSpec


def ggml_op_soft_max(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """
    GGML_OP_SOFT_MAX implementation.

    Parameters:
        arch: Target architecture.
        input_tensors: List of 1-3 input tensors:
            - input_tensors[0]: Input tensor (required)
            - input_tensors[1]: Mask tensor (optional)
            - input_tensors[2]: Sink tensor (optional)
        output_tensor: Output tensor.
        op_params: Operation parameters (scale, max_bias).

    Returns:
        KernelSpec for the SOFT_MAX operation.
    """
    return KernelSpec(
        backend=Backend.IRON,
        function=softmax,
    )
