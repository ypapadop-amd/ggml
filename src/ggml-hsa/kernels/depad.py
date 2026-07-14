#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the MUL_MAT de-pad post-amble kernel."""

from .kernel import Backend, KernelSpec


def ggml_hsa_depad(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for the de-pad post-amble.

    Parameters:
        arch: Target architecture.
        input_tensors: [src] padded f32 buffer of shape [d0pad, d1pad].
        output_tensor: dense f32 tensor of logical shape [d0, d1].
        op_params: unused.

    Returns:
        KernelSpec for the de-pad operation.

    """
    from functools import partial

    from .iron_kernels.depad import depad

    return KernelSpec(
        backend=Backend.IRON,
        op_name="HSA_DEPAD",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            depad,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
