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
    """GGML_OP_ARGMAX dispatch function.

    Finds the index of the maximum value along the first dimension (ne0) of each row.
    For a tensor with shape [ne0, ne1, ne2, ne3], computes argmax over ne0 for each
    of the ne1 * ne2 * ne3 rows, producing an I32 output tensor with shape [ne1, ne2, ne3].

    Parameters:
        arch: Target architecture.
        input_tensors: List containing exactly one input tensor.
        output_tensor: Output tensor of type I32. Shape is
            the input shape with the first dimension removed.
        op_params: Operation parameters (unused for ARGMAX, but required
            by the dispatch interface).

    Returns:
        KernelSpec: Kernel specification for the ARGMAX operation.

    """
    from .iron.argmax import argmax_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_ARGMAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=argmax_op,
    )
