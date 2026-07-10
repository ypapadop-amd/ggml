#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_IM2COL (2D image-to-column layout transform).

im2col gathers sliding kernel windows into columns so a convolution can run as a
matmul. The full input image for one batch element [IW, IH, IC] is loaded into
L1 once, then one output row [OW, IC*KH*KW] is emitted per worker iteration for
each of the OH output rows. Output columns pack taps channel-major
(IC*KH*KW), applying zero-fill padding and dilation. Out-of-bounds taps are zero.

The convolution kernel (src0) is only used for its KW/KH shape; it carries no
data, so it is not moved onto the AIE array. The image (src1) is float32; the
output is float32 or bfloat16 (ggml_conv_2d requests the kernel's dtype).

Scope: 2D mode only (is_2D == 1).
"""

import struct
from pathlib import Path

import numpy as np
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

from .utils import arch_to_device


def im2col(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the im2col IRON program.

    Parameters:
        arch: Target architecture.
        input_tensors: [kernel src0 (KW, KH, IC, OC), image src1 (IW, IH, IC, N)].
        output_tensor: Output tensor, shape (IC*KH*KW, OW, OH, N).
        op_params: {s0, s1, p0, p1, d0, d1, is_2D} as 7 x int32.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, op_params, or
            unsupported mode.

    """
    if len(input_tensors) != 2:
        msg = "Operation requires exactly two input tensors (kernel and image)."
        raise ValueError(msg)

    kernel_tensor = input_tensors[0]
    image_tensor = input_tensors[1]

    if image_tensor.dtype != np.float32:
        msg = f"IM2COL only supports float32 image input; got {image_tensor.dtype}."
        raise ValueError(msg)

    if output_tensor.dtype not in (np.float32, bfloat16):
        msg = (
            f"IM2COL only supports float32/bfloat16 output; got {output_tensor.dtype}."
        )
        raise ValueError(msg)

    if not image_tensor.contiguous or not output_tensor.contiguous:
        msg = "Image and output tensors must be contiguous in memory."
        raise ValueError(msg)

    # op_params: {s0, s1, p0, p1, d0, d1, is_2D} as 7 x int32.
    _IM2COL_PARAMS_SIZE = 7 * 4
    if len(op_params) < _IM2COL_PARAMS_SIZE:
        msg = (
            f"op_params too short: expected at least {_IM2COL_PARAMS_SIZE} bytes, "
            f"got {len(op_params)}."
        )
        raise ValueError(msg)
    s0, s1, p0, p1, d0, d1, is_2d = struct.unpack_from("7i", op_params, 0)

    if is_2d != 1:
        msg = "IM2COL only supports 2D mode (is_2D == 1)."
        raise ValueError(msg)

    if s0 <= 0 or s1 <= 0:
        msg = f"Strides must be positive; got s0={s0}, s1={s1}."
        raise ValueError(msg)

    if p0 < 0 or p1 < 0:
        msg = f"Padding must be non-negative; got p0={p0}, p1={p1}."
        raise ValueError(msg)

    if d0 <= 0 or d1 <= 0:
        msg = f"Dilation must be positive; got d0={d0}, d1={d1}."
        raise ValueError(msg)

    # Kernel (src0) provides only the window shape.
    kw, kh = kernel_tensor.shape[0], kernel_tensor.shape[1]
    # Image (src1): [IW, IH, IC, N].
    iw, ih, ic, n = image_tensor.shape
    # Output: [IC*KH*KW, OW, OH, N].
    col_stride, ow, oh, out_n = output_tensor.shape

    if col_stride != ic * kh * kw:
        msg = (
            f"Output column stride {col_stride} does not match IC*KH*KW="
            f"{ic * kh * kw} (IC={ic}, KH={kh}, KW={kw})."
        )
        raise ValueError(msg)

    if out_n != n:
        msg = f"Batch mismatch: image N={n} vs output N={out_n}."
        raise ValueError(msg)

    image_size = ic * ih * iw
    row_size = ow * col_stride

    function = _create_external_function(
        image_tensor=image_tensor,
        output_tensor=output_tensor,
        image_size=image_size,
        row_size=row_size,
    )

    # One image per outer iteration; one output row per inner iteration.
    image_tile_ty = np.ndarray[(image_size,), np.dtype[image_tensor.dtype]]
    row_tile_ty = np.ndarray[(row_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(image_tile_ty, name="in")
    of_out = ObjectFifo(row_tile_ty, name="out")

    # The outer loop over batch elements is emitted as an AIE loop (range_). The
    # inner loop over output rows uses a plain Python range so it unrolls at
    # build time: nested range_ loops are miscompiled (mlir-aie issue #1547), and
    # OH is small. oh_idx is then a compile-time constant passed straight in.
    def ext_core_fn(of_in, of_out, function):
        for _ in range_(n):
            img = of_in.acquire(1)
            for oh_idx in range(oh):
                row = of_out.acquire(1)
                function(
                    img, row, oh_idx, iw, ih, ic, kw, kh, ow, s0, s1, p0, p1, d0, d1
                )
                of_out.release(1)
            of_in.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    # The runtime sequence binds one buffer per ggml tensor, positionally:
    # src0 (kernel), src1 (image), then dst. The kernel carries no data (only its
    # shape is used) so it is declared but never moved onto the array.
    rt = Runtime()
    kernel_numel = kernel_tensor.numel()
    kernel_tensor_ty = np.ndarray[(kernel_numel,), np.dtype[kernel_tensor.dtype]]
    image_tensor_ty = np.ndarray[(image_size * n,), np.dtype[image_tensor.dtype]]
    output_tensor_ty = np.ndarray[(row_size * oh * n,), np.dtype[output_tensor.dtype]]
    with rt.sequence(kernel_tensor_ty, image_tensor_ty, output_tensor_ty) as (
        _a_kernel,
        a_in,
        b_out,
    ):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    image_tensor,
    output_tensor,
    image_size: int,
    row_size: int,
) -> ExternalFunction:
    """Create the ExternalFunction for the im2col core function.

    Parameters:
        image_tensor: Image input tensor (src1).
        output_tensor: Output tensor.
        image_size: Elements in one input image (IC * IH * IW).
        row_size: Elements in one output row (OW * IC * KH * KW).

    Returns:
        The configured ExternalFunction.

    """
    op_name = "GGML_OP_IM2COL"
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "im2col.cc"),
        arg_types=[
            np.ndarray[(image_size,), np.dtype[image_tensor.dtype]],
            np.ndarray[(row_size,), np.dtype[output_tensor.dtype]],
            np.int32,  # oh
            np.int32,  # iw
            np.int32,  # ih
            np.int32,  # ic
            np.int32,  # kw
            np.int32,  # kh
            np.int32,  # ow
            np.int32,  # s0
            np.int32,  # s1
            np.int32,  # p0
            np.int32,  # p1
            np.int32,  # d0
            np.int32,  # d1
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(image_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
