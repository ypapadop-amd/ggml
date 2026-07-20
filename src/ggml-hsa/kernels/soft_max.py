#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML softmax operation (GGML_OP_SOFT_MAX)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_soft_max_kernel_spec(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for SOFT_MAX.

    Softmax is computed over the innermost dimension (ne0), one Triton program per row. Only the
    plain case (no attention mask, no ALiBi ``max_bias``) is handled on the GPU; the ``scale`` from
    op_params is baked into the kernel as a constexpr. Other cases raise ValueError so the op is
    reported unsupported.
    """
    import struct

    import triton

    from .triton_kernels.gpu_ops import next_pow2, softmax_kernel

    # op_params: float scale @ 0, float max_bias @ 4
    scale = 1.0
    max_bias = 0.0
    if op_params is not None and len(op_params) >= 8:
        scale, max_bias = struct.unpack_from("<ff", bytes(op_params), 0)

    # src[0] = input; src[1] = mask (optional); GPU path only handles the mask-free, no-ALiBi case
    if len(input_tensors) > 1 and input_tensors[1] is not None:
        msg = "Triton GPU SOFT_MAX does not support an attention mask."
        raise ValueError(msg)
    if max_bias != 0.0:
        msg = "Triton GPU SOFT_MAX does not support max_bias (ALiBi)."
        raise ValueError(msg)
    if not output_tensor.contiguous or not input_tensors[0].contiguous:
        msg = "Triton GPU SOFT_MAX requires contiguous tensors."
        raise ValueError(msg)

    n_cols = int(output_tensor.shape[0])
    n_rows = output_tensor.numel() // n_cols
    block = next_pow2(n_cols)

    signature = {
        "X": "*fp32",
        "Y": "*fp32",
        "n_cols": "constexpr",
        "SCALE": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": softmax_kernel,
        "signature": signature,
        "constexprs": {"n_cols": n_cols, "SCALE": float(scale), "BLOCK": block},
        "num_warps": 4,
        "num_programs": int(n_rows),
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_SOFT_MAX",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=softmax_kernel,
        config=config,
    )


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

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_soft_max_kernel_spec(
            arch, input_tensors, output_tensor, op_params
        )

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
