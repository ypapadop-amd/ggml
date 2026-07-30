#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML norm operation (GGML_OP_NORM)."""

from .kernel import Backend, KernelSpec


def ggml_op_norm(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_NORM.

    Layer normalization over dim 0: each row of an [nc, nr, nz] tensor is
    normalized to zero mean and unit variance as y = (x - mean) / sqrt(var + eps).

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: eps as a single float32.

    Returns:
        KernelSpec for the NORM operation.

    """
    from functools import partial

    from .iron_kernels.norm import norm

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_NORM",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            norm,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
