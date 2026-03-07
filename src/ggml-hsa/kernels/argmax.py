#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry point for the GGML argmax operation (GGML_OP_ARGMAX).

Returns a KernelSpec specifying the compilation backend and kernel function.
"""

from .iron.argmax import argmax_op
from .kernel import Backend, KernelSpec


def ggml_op_argmax(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """
    GGML_OP_ARGMAX dispatch function.

    Finds the index of the maximum value along the first dimension (ne0) of each row.
    For a tensor with shape [ne0, ne1, ne2, ne3], computes argmax over ne0 for each
    of the ne1 * ne2 * ne3 rows, producing an I32 output tensor with shape [ne1, ne2, ne3].

    Parameters:
        arch (str): Target architecture (e.g., "aie2" for Phoenix/Hawk Point,
            "aie2p" for Strix Halo/Krackan).
        input_tensors (list[TensorDesc]): List containing exactly one input tensor
            descriptor. The tensor must be F32 type and contiguous in memory.
        output_tensor (TensorDesc): Output tensor descriptor of type I32. Shape is
            the input shape with the first dimension removed.
        op_params (bytearray): Operation parameters as a 64-byte buffer (unused
            for ARGMAX, but required by the dispatch interface).

    Returns:
        KernelSpec: Kernel specification with backend=IRON and the argmax_op function
            for generating the MLIR module.
    """
    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_ARGMAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=argmax_op,
    )
