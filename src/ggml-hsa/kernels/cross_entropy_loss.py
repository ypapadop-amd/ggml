#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""
Top-level entry point for the GGML cross entropy loss operation (GGML_OP_CROSS_ENTROPY_LOSS).
"""

from .kernel import Backend, KernelSpec


def ggml_op_cross_entropy_loss(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """
    GGML_OP_CROSS_ENTROPY_LOSS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of 2 input tensors:
            - input_tensors[0]: Logits tensor (predictions before softmax)
            - input_tensors[1]: Labels tensor (ground truth, often one-hot encoded)
        output_tensor: Output scalar tensor containing the loss value.
        op_params (bytearray): Operation parameters (currently unused).

    Returns:
        KernelSpec for the CROSS_ENTROPY_LOSS operation.
    """
    from .iron.cross_entropy_loss import cross_entropy_loss

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_CROSS_ENTROPY_LOSS",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=cross_entropy_loss,
    )
