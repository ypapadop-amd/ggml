#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML 2D pooling operation (GGML_OP_POOL_2D)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_pool_2d_kernel_spec(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for POOL_2D (max/avg).

    Handled as a per-destination-element gather over the k0 x k1 window; all geometry is baked as
    Triton constexprs. Supports FP32 or FP16 input (dst is FP32). Non-contiguous inputs raise
    ValueError so the op is reported unsupported.
    """
    import struct

    import numpy as np
    import triton

    from .triton_kernels.gpu_ops import next_pow2, pool2d_kernel
    from .triton_kernels.utils import numpy_dtype_to_triton_elt

    op, k0, k1, s0, s1, p0, p1 = struct.unpack_from("<7i", bytes(op_params), 0)

    src = input_tensors[0]
    if src.dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        msg = "Triton GPU POOL_2D supports FP32/FP16 input only."
        raise ValueError(msg)
    if output_tensor.dtype != np.dtype(np.float32):
        msg = "Triton GPU POOL_2D produces FP32 output."
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "Triton GPU POOL_2D requires contiguous tensors."
        raise ValueError(msg)

    iw = int(src.shape[0])
    ih = int(src.shape[1])
    px = int(output_tensor.shape[0])
    py = int(output_tensor.shape[1])
    n_elements = output_tensor.numel()

    block = min(1024, next_pow2(n_elements))
    num_programs = triton.cdiv(n_elements, block)

    elt = numpy_dtype_to_triton_elt(src.dtype)
    signature = {
        "SRC": f"*{elt}",
        "DST": "*fp32",
        "n_elements": "constexpr",
        "PX": "constexpr",
        "PY": "constexpr",
        "IW": "constexpr",
        "IH": "constexpr",
        "IWIH": "constexpr",
        "K0": "constexpr",
        "K1": "constexpr",
        "S0": "constexpr",
        "S1": "constexpr",
        "P0": "constexpr",
        "P1": "constexpr",
        "OP": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": pool2d_kernel,
        "signature": signature,
        "constexprs": {
            "n_elements": n_elements,
            "PX": px,
            "PY": py,
            "IW": iw,
            "IH": ih,
            "IWIH": iw * ih,
            "K0": k0,
            "K1": k1,
            "S0": s0,
            "S1": s1,
            "P0": p0,
            "P1": p1,
            "OP": int(op),
            "BLOCK": block,
        },
        "num_warps": 4,
        "num_programs": num_programs,
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_POOL_2D",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=pool2d_kernel,
        config=config,
    )


def ggml_op_pool_2d(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_POOL_2D.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: {op, k0, k1, s0, s1, p0, p1} as 7 x int32.

    Returns:
        KernelSpec for the POOL_2D operation.

    """
    from functools import partial

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_pool_2d_kernel_spec(
            arch, input_tensors, output_tensor, op_params
        )

    from .iron_kernels.pool_2d import pool_2d

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_POOL_2D",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            pool_2d,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
