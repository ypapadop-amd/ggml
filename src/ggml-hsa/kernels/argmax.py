#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML argmax operation (GGML_OP_ARGMAX)."""

from .kernel import Backend, KernelSpec


def ggml_op_argmax(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_ARGMAX.

    Argmax over ne0 for input [ne0, ne1, ne2, ne3], producing I32 output [ne1, ne2, ne3].

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: I32 output tensor.
        op_params: Unused.

    Returns:
        KernelSpec for the ARGMAX operation.

    """
    from functools import partial

    from .iron_kernels.argmax import argmax_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_ARGMAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            argmax_op,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )
