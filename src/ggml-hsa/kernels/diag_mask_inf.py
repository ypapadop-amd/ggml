#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML diag_mask_inf operation (GGML_OP_DIAG_MASK_INF)."""

from .kernel import Backend, KernelSpec


def ggml_op_diag_mask_inf(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_DIAG_MASK_INF.

    Causal masking over dim 0: for row j of an [nc, nr, nz] tensor, columns
    i > n_past + j are set to -inf and the rest are copied from the input.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: n_past as a single int32.

    Returns:
        KernelSpec for the DIAG_MASK_INF operation.

    """
    from functools import partial

    from .iron_kernels.diag_mask_inf import diag_mask_inf

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_DIAG_MASK_INF",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            diag_mask_inf,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
