#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML count equal operation (GGML_OP_COUNT_EQUAL)."""

from .kernel import Backend, KernelSpec


def ggml_op_count_equal(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """GGML_OP_COUNT_EQUAL dispatch function.

    Counts the number of elements that are equal between two input tensors.
    Both input tensors must have the same shape and be of type I32.
    The output is a single I64 scalar containing the count.

    Parameters:
        arch: Target architecture.
        input_tensors: List containing exactly two input tensors. Both tensors must be I32 type and contiguous in memory.
        output_tensor: Output tensor of type I64 with
            shape [1, 1, 1, 1] containing the count of equal elements.
        op_params: Operation parameters as a 64-byte buffer (unused
            for COUNT_EQUAL, but required by the dispatch interface).

    Returns:
        KernelSpec: Kernel specification with backend=IRON and the count_equal_op
            function for generating the MLIR module.

    """
    from .iron.count_equal import count_equal_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_COUNT_EQUAL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=count_equal_op,
    )
