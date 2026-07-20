#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML im2col operation (GGML_OP_IM2COL)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_im2col_kernel_spec(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for 2-D IM2COL.

    Handled as a per-destination-element masked gather; every geometry parameter is baked as a
    Triton constexpr. Only the contiguous FP32, is_2D case is supported; other cases raise
    ValueError so the op is reported unsupported.
    """
    import struct

    import numpy as np
    import triton

    from .triton_kernels.gpu_ops import im2col_kernel, next_pow2

    s0, s1, p0, p1, d0, d1, is_2d = struct.unpack_from("<7i", bytes(op_params), 0)
    if is_2d != 1:
        msg = "Triton GPU IM2COL supports only the 2-D case."
        raise ValueError(msg)

    kern, image = input_tensors[0], input_tensors[1]
    if (
        image.dtype != np.dtype(np.float32)
        or output_tensor.dtype != np.dtype(np.float32)
    ):
        msg = "Triton GPU IM2COL supports FP32 only."
        raise ValueError(msg)
    if not image.contiguous or not output_tensor.contiguous:
        msg = "Triton GPU IM2COL requires contiguous image and output."
        raise ValueError(msg)

    # kernel src0 ne=[KW, KH, IC, OC]; image src1 ne=[IW, IH, IC, N]
    kw = int(kern.shape[0])
    kh = int(kern.shape[1])
    ic = int(kern.shape[2])
    iw = int(image.shape[0])
    ih = int(image.shape[1])

    # dst ne=[IC*KH*KW, OW, OH, N]
    ickhkw = int(output_tensor.shape[0])
    ow = int(output_tensor.shape[1])
    oh = int(output_tensor.shape[2])
    n_elements = output_tensor.numel()

    block = min(1024, next_pow2(n_elements))
    num_programs = triton.cdiv(n_elements, block)

    signature = {
        "KERN": "*fp32",
        "SRC": "*fp32",
        "DST": "*fp32",
        "n_elements": "constexpr",
        "ICKHKW": "constexpr",
        "KW": "constexpr",
        "KH": "constexpr",
        "OW": "constexpr",
        "OH": "constexpr",
        "IW": "constexpr",
        "IH": "constexpr",
        "IWIH": "constexpr",
        "IWIHIC": "constexpr",
        "S0": "constexpr",
        "S1": "constexpr",
        "P0": "constexpr",
        "P1": "constexpr",
        "D0": "constexpr",
        "D1": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": im2col_kernel,
        "signature": signature,
        "constexprs": {
            "n_elements": n_elements,
            "ICKHKW": ickhkw,
            "KW": kw,
            "KH": kh,
            "OW": ow,
            "OH": oh,
            "IW": iw,
            "IH": ih,
            "IWIH": iw * ih,
            "IWIHIC": iw * ih * ic,
            "S0": s0,
            "S1": s1,
            "P0": p0,
            "P1": p1,
            "D0": d0,
            "D1": d1,
            "BLOCK": block,
        },
        "num_warps": 4,
        "num_programs": num_programs,
    }
    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_IM2COL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=im2col_kernel,
        config=config,
    )


def ggml_op_im2col(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_IM2COL.

    Parameters:
        arch: Target architecture.
        input_tensors: [kernel src0, image src1].
        output_tensor: Output tensor.
        op_params: {s0, s1, p0, p1, d0, d1, is_2D} as 7 x int32.

    Returns:
        KernelSpec for the IM2COL operation.

    """
    from functools import partial

    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_im2col_kernel_spec(
            arch, input_tensors, output_tensor, op_params
        )

    from .iron_kernels.im2col import im2col

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_IM2COL",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=partial(
            im2col,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
