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
    """Return the KernelSpec for GGML_OP_COUNT_EQUAL.

    Counts elementwise-equal entries between two same-shaped I32 tensors.

    Parameters:
        arch: Target architecture.
        input_tensors: List of two contiguous I32 tensors.
        output_tensor: I64 scalar [1, 1, 1, 1] holding the count.
        op_params: Unused.

    Returns:
        KernelSpec for the COUNT_EQUAL operation.

    """
    from functools import partial

    from .iron_kernels.count_equal import count_equal_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_COUNT_EQUAL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            count_equal_op,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )
