#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML argmax operation (GGML_OP_ARGMAX)."""

from .kernel import Backend, KernelSpec
from .triton_kernels.spec_utils import ArchKind, arch_kind


def _make_triton_gpu_argmax_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for ARGMAX.

    Argmax over the innermost dim (ne0) of the FP32 input, one Triton program per row, producing
    an I32 index per row. All dims/block sizes are baked as constexprs; the only runtime kernel
    args are the input and (I32) output pointers.
    """
    import numpy as np

    from .triton_kernels.argmax import argmax_kernel
    from .triton_kernels.spec_utils import next_pow2

    src = input_tensors[0]
    if src.dtype != np.dtype(np.float32):
        msg = "Triton GPU ARGMAX supports FP32 input only."
        raise ValueError(msg)
    if not src.contiguous:
        msg = "Triton GPU ARGMAX requires a contiguous input."
        raise ValueError(msg)

    n_cols = int(src.shape[0])
    n_rows = src.numel() // n_cols
    block = next_pow2(n_cols)

    signature = {
        "X": "*fp32",
        "Y": "*i32",
        "n_cols": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": argmax_kernel,
        "signature": signature,
        "constexprs": {"n_cols": n_cols, "BLOCK": block},
        "num_warps": 4,
        "num_programs": int(n_rows),
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_ARGMAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=argmax_kernel,
        config=config,
    )


def ggml_op_argmax(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec | list[KernelSpec]:
    """Return the KernelSpec(s) for GGML_OP_ARGMAX.

    Argmax over ne0 for input [ne0, ne1, ne2, ne3], producing I32 output [ne1, ne2, ne3].

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: I32 output tensor.
        op_params: Unused.

    Returns:
        KernelSpec for the ARGMAX operation (IRON) on NPU targets; on GPU (gfx) targets a list
        holding the Triton -> hsaco spec, or an empty list if the tensors are unsupported.
    """
    if arch_kind(arch) is ArchKind.GPU:
        try:
            return [
                _make_triton_gpu_argmax_kernel_spec(arch, input_tensors, output_tensor)
            ]
        except ValueError:
            return []

    from functools import partial

    from .iron_kernels.argmax import argmax_op

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_ARGMAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            argmax_op,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        ),
    )
