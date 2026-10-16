#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML cross entropy loss operation (GGML_OP_CROSS_ENTROPY_LOSS)."""

from .kernel import Backend, KernelSpec
from .triton_kernels.spec_utils import ArchKind, arch_kind

# Elements per (rows, classes) tile of the GPU kernel: as many rows as fit, so a narrow class
# dimension (e.g. MNIST's 10) still keeps the workgroup busy.
_GPU_TILE_ELEMENTS = 4096


def _make_triton_gpu_cross_entropy_loss_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for CROSS_ENTROPY_LOSS.

    Computes the scalar mean cross-entropy loss with a single Triton program that loops over the
    rows in tiles of up to _GPU_TILE_ELEMENTS elements, reducing the rows of a tile in parallel.
    Only the contiguous FP32 case is supported; other cases raise ValueError so the op is reported
    unsupported.
    """
    import numpy as np

    from .triton_kernels.cross_entropy_loss import cross_entropy_loss_kernel
    from .triton_kernels.spec_utils import next_pow2

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
    block_r = min(next_pow2(nr), max(1, _GPU_TILE_ELEMENTS // block))

    signature = {
        "S0": "*fp32",
        "S1": "*fp32",
        "DST": "*fp32",
        "NR": "constexpr",
        "NC": "constexpr",
        "INV_NR": "constexpr",
        "BLOCK": "constexpr",
        "BLOCK_R": "constexpr",
    }
    config = {
        "triton_fn": cross_entropy_loss_kernel,
        "signature": signature,
        "constexprs": {
            "NR": int(nr),
            "NC": nc,
            "INV_NR": 1.0 / float(nr),
            "BLOCK": block,
            "BLOCK_R": block_r,
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
) -> KernelSpec | list[KernelSpec]:
    """Return the KernelSpec(s) for GGML_OP_CROSS_ENTROPY_LOSS.

    Args:
        arch: Target architecture.
        input_tensors: [logits, labels].
        output_tensor: Scalar loss value.
        op_params: Unused.

    Returns:
        KernelSpec for the CROSS_ENTROPY_LOSS operation (IRON) on NPU targets; on GPU (gfx) targets a list
        holding the Triton -> hsaco spec, or an empty list if the tensors are unsupported.

    """
    if arch_kind(arch) is ArchKind.GPU:
        try:
            return [
                _make_triton_gpu_cross_entropy_loss_kernel_spec(
                    arch, input_tensors, output_tensor
                )
            ]
        except ValueError:
            return []

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
