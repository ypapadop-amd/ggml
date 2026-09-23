#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the element-wise dtype-conversion (CPY cast) kernel."""

from .kernel import Backend, KernelSpec


def ggml_hsa_convert(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for an element-wise dtype conversion.

    Args:
        arch: Target architecture.
        input_tensors: [src] dense contiguous tensor.
        output_tensor: dense contiguous tensor, same element count, different dtype.
        op_params: unused.

    Returns:
        KernelSpec for the convert operation.

    """
    from functools import partial

    from .iron_kernels.convert import convert

    return KernelSpec(
        backend=Backend.IRON,
        op_name="HSA_CONVERT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            convert,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
