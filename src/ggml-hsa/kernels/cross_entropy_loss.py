#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML cross entropy loss operation (GGML_OP_CROSS_ENTROPY_LOSS)."""

from .kernel import Backend, KernelSpec


def ggml_op_cross_entropy_loss(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_CROSS_ENTROPY_LOSS.

    Args:
        arch: Target architecture.
        input_tensors: [logits, labels].
        output_tensor: Scalar loss value.
        op_params: Unused.

    Returns:
        KernelSpec for the CROSS_ENTROPY_LOSS operation.

    """
    from functools import partial

    from .iron_kernels.cross_entropy_loss import cross_entropy_loss

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_CROSS_ENTROPY_LOSS",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            cross_entropy_loss,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )
