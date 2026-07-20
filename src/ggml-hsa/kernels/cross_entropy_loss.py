#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML cross entropy loss operation (GGML_OP_CROSS_ENTROPY_LOSS)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_cross_entropy_loss_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for CROSS_ENTROPY_LOSS.

    Computes the scalar mean cross-entropy loss with a single Triton program that loops over the
    rows (row count baked as a constexpr). Only the contiguous FP32 case is supported; other cases
    raise ValueError so the op is reported unsupported.
    """
    import numpy as np

    from .triton_kernels.gpu_ops import cross_entropy_loss_kernel, next_pow2

    logits, labels = input_tensors[0], input_tensors[1]
    f32 = np.dtype(np.float32)
    if logits.dtype != f32 or labels.dtype != f32:
        msg = "Triton GPU CROSS_ENTROPY_LOSS supports FP32 only."
        raise ValueError(msg)
    if not logits.contiguous or not labels.contiguous:
        msg = "Triton GPU CROSS_ENTROPY_LOSS requires contiguous inputs."
        raise ValueError(msg)

    nc = int(logits.shape[0])
    nr = logits.numel() // nc
    block = next_pow2(nc)

    signature = {
        "S0": "*fp32",
        "S1": "*fp32",
        "DST": "*fp32",
        "NR": "constexpr",
        "NC": "constexpr",
        "INV_NR": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": cross_entropy_loss_kernel,
        "signature": signature,
        "constexprs": {
            "NR": int(nr),
            "NC": nc,
            "INV_NR": 1.0 / float(nr),
            "BLOCK": block,
        },
        "num_warps": 4,
        "num_programs": 1,
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_CROSS_ENTROPY_LOSS",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=cross_entropy_loss_kernel,
        config=config,
    )


def ggml_op_cross_entropy_loss(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_CROSS_ENTROPY_LOSS.

    Parameters:
        arch: Target architecture.
        input_tensors: [logits, labels].
        output_tensor: Scalar loss value.
        op_params: Unused.

    Returns:
        KernelSpec for the CROSS_ENTROPY_LOSS operation.

    """
    from functools import partial

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_cross_entropy_loss_kernel_spec(
            arch, input_tensors, output_tensor
        )

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
