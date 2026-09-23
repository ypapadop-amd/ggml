#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the MUL_MAT convert+pad pre-amble kernel."""

from .kernel import Backend, KernelSpec


def ggml_hsa_convert_pad(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for the convert+pad pre-amble.

    Args:
        arch: Target architecture.
        input_tensors: [src] dense f32 or bf16 tensor of logical shape [d0, d1].
        output_tensor: padded bf16 tensor of shape [d0pad, d1pad].
        op_params: unused.

    Returns:
        KernelSpec for the convert+pad operation.

    """
    from functools import partial

    from .iron_kernels.convert_pad import convert_pad

    return KernelSpec(
        backend=Backend.IRON,
        op_name="HSA_CONVERT_PAD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            convert_pad,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
