#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML softmax operation (GGML_OP_SOFT_MAX)."""

from .kernel import Backend, KernelSpec


def ggml_op_soft_max(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_SOFT_MAX.

    Parameters:
        arch: Target architecture.
        input_tensors: [input, mask (optional), sink (optional)].
        output_tensor: Output tensor.
        op_params: scale, max_bias.

    Returns:
        KernelSpec for the SOFT_MAX operation.

    """
    from functools import partial

    from .iron_kernels.softmax import softmax

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_SOFT_MAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            softmax,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
