#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Top-level entry point for the GGML direct 2-D convolution (GGML_OP_CONV_2D)."""

from .kernel import Backend, KernelSpec


def _make_triton_gpu_conv_2d_kernel_spec(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Create a TRITON-backend (GPU AOT) KernelSpec for direct CONV_2D.

    Handled as a per-destination-element reduction over the (IC, KH, KW) window; all geometry is
    baked as Triton constexprs. Only the contiguous FP32 case is supported; other cases raise
    ValueError so the op is reported unsupported.
    """
    import struct

    import numpy as np
    import triton

    from .triton_kernels.gpu_ops import conv2d_kernel, next_pow2

    s0, s1, p0, p1, d0, d1 = struct.unpack_from("<6i", bytes(op_params), 0)

    kern, src = input_tensors[0], input_tensors[1]
    f32 = np.dtype(np.float32)
    if kern.dtype != f32 or src.dtype != f32 or output_tensor.dtype != f32:
        msg = "Triton GPU CONV_2D supports FP32 only."
        raise ValueError(msg)
    if any(not t.contiguous for t in (kern, src, output_tensor)):
        msg = "Triton GPU CONV_2D requires contiguous tensors."
        raise ValueError(msg)

    # kernel src0 ne=[KW, KH, IC, OC]; image src1 ne=[W, H, IC, N]; dst ne=[OW, OH, OC, N]
    kw = int(kern.shape[0])
    kh = int(kern.shape[1])
    ic = int(kern.shape[2])
    w = int(src.shape[0])
    h = int(src.shape[1])
    ow = int(output_tensor.shape[0])
    oh = int(output_tensor.shape[1])
    oc = int(output_tensor.shape[2])
    if int(src.shape[2]) != ic:
        msg = "Triton GPU CONV_2D: kernel/src input-channel mismatch."
        raise ValueError(msg)

    n_elements = output_tensor.numel()
    block = min(1024, next_pow2(n_elements))
    num_programs = triton.cdiv(n_elements, block)

    signature = {
        "KERN": "*fp32",
        "SRC": "*fp32",
        "DST": "*fp32",
        "n_elements": "constexpr",
        "OW": "constexpr",
        "OH": "constexpr",
        "OC": "constexpr",
        "KW": "constexpr",
        "KH": "constexpr",
        "IC": "constexpr",
        "W": "constexpr",
        "H": "constexpr",
        "WH": "constexpr",
        "WHC": "constexpr",
        "KWKH": "constexpr",
        "KNL_N": "constexpr",
        "S0": "constexpr",
        "S1": "constexpr",
        "P0": "constexpr",
        "P1": "constexpr",
        "D0": "constexpr",
        "D1": "constexpr",
        "BLOCK": "constexpr",
    }
    config = {
        "triton_fn": conv2d_kernel,
        "signature": signature,
        "constexprs": {
            "n_elements": n_elements,
            "OW": ow,
            "OH": oh,
            "OC": oc,
            "KW": kw,
            "KH": kh,
            "IC": ic,
            "W": w,
            "H": h,
            "WH": w * h,
            "WHC": w * h * ic,
            "KWKH": kw * kh,
            "KNL_N": kw * kh * ic,
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
        op_name="GGML_OP_CONV_2D",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=conv2d_kernel,
        config=config,
    )


def ggml_op_conv_2d(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> KernelSpec:
    """Return the KernelSpec for GGML_OP_CONV_2D.

    Parameters:
        arch: Target architecture.
        input_tensors: [kernel src0, image src1].
        output_tensor: Output tensor [OW, OH, OC, N].
        op_params: {stride_x, stride_y, pad_x, pad_y, dilation_x, dilation_y} as 6 x int32.

    Returns:
        KernelSpec for the CONV_2D operation.

    """
    from .triton_kernels.utils import is_gpu_arch

    if is_gpu_arch(arch):
        return _make_triton_gpu_conv_2d_kernel_spec(
            arch, input_tensors, output_tensor, op_params
        )

    msg = "CONV_2D is only implemented for the Triton GPU path."
    raise NotImplementedError(msg)
