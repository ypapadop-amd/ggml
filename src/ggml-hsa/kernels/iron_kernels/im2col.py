#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_IM2COL (2D image-to-column layout transform)."""

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

from .utils import arch_to_device, batch_slice_tap, partition_units

# Cap on data-parallel workers (compute tiles). Beyond this the per-worker shim/
# mem-tile DMA channels exhaust the array's routing budget on NPU1 (aie2).
_MAX_WORKERS = 8


def im2col(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the im2col IRON program.

    Args:
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

    if n <= 0:
        msg = f"Batch size N must be positive; got N={n}."
        raise ValueError(msg)

    image_size = ic * ih * iw
    row_size = ow * col_stride

    function = _create_external_function(
        image_tensor=image_tensor,
        output_tensor=output_tensor,
        image_size=image_size,
        row_size=row_size,
    )

    # The batch dimension is embarrassingly parallel: distribute the N images
    # across compute tiles. Each worker owns an independent input/output fifo
    # pair (one worker per compute tile) and streams its slice of the batch one
    # image at a time. Independent per-worker fifos fed by per-worker DMA taps
    # are used rather than a single split/join fifo: the split-then-stream-many-
    # objects access pattern fails aiecc DMA lowering, whereas per-worker taps
    # lower cleanly. Workers are capped at _MAX_WORKERS to stay within the shim/
    # mem-tile DMA channel budget (16 workers exhausts it on NPU1).
    num_workers = min(_MAX_WORKERS, n)
    images_per_worker, image_starts = partition_units(num_workers, n)
    out_per_image = row_size * oh

    image_tile_ty = np.ndarray[(image_size,), np.dtype[image_tensor.dtype]]
    row_tile_ty = np.ndarray[(row_size,), np.dtype[output_tensor.dtype]]

    of_ins = [ObjectFifo(image_tile_ty, name=f"in{w}") for w in range(num_workers)]
    of_outs = [ObjectFifo(row_tile_ty, name=f"out{w}") for w in range(num_workers)]

    # The outer loop over this worker's images is emitted as an AIE loop
    # (range_). The inner loop over output rows uses a plain Python range so it
    # unrolls at build time: nested range_ loops are miscompiled (mlir-aie issue
    # #1547), and OH is small. oh_idx is then a compile-time constant.
    def make_core_fn(image_count):
        def ext_core_fn(of_in, of_out, function):
            for _ in range_(image_count):
                img = of_in.acquire(1)
                for oh_idx in range(oh):
                    row = of_out.acquire(1)
                    function(
                        img, row, oh_idx, iw, ih, ic, kw, kh, ow, s0, s1, p0, p1, d0, d1
                    )
                    of_out.release(1)
                of_in.release(1)

        return ext_core_fn

    workers = [
        Worker(
            make_core_fn(images_per_worker[w]),
            fn_args=[of_ins[w].cons(), of_outs[w].prod(), function],
        )
        for w in range(num_workers)
    ]

    # Per-worker DMA taps select each worker's contiguous slice of the batch from
    # the whole tensors: the input viewed as [N, image_size], the output as
    # [N, out_per_image]. worker w reads images [start, start+count).

    # The runtime sequence binds one buffer per ggml tensor, positionally:
    # src0 (kernel), src1 (image), then dst. The kernel carries no data (only its
    # shape is used) so it is declared but never moved onto the array.
    kernel_numel = kernel_tensor.numel()
    kernel_tensor_ty = np.ndarray[(kernel_numel,), np.dtype[kernel_tensor.dtype]]
    image_tensor_ty = np.ndarray[(image_size * n,), np.dtype[image_tensor.dtype]]
    output_tensor_ty = np.ndarray[(out_per_image * n,), np.dtype[output_tensor.dtype]]

    def sequence(_a_kernel, a_in, b_out, in_prods, out_conses):
        for w in range(num_workers):
            count = images_per_worker[w]
            in_tap = batch_slice_tap(n, image_size, image_starts[w], count)
            in_prods[w].fill(a_in, in_tap)
        for w in range(num_workers):
            count = images_per_worker[w]
            out_tap = batch_slice_tap(n, out_per_image, image_starts[w], count)
            out_conses[w].drain(b_out, out_tap, wait=True)

    rt = Runtime(
        sequence,
        [
            kernel_tensor_ty,
            image_tensor_ty,
            output_tensor_ty,
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )

    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _create_external_function(
    image_tensor,
    output_tensor,
    image_size: int,
    row_size: int,
) -> ExternalFunction:
    """Create the ExternalFunction for the im2col core function.

    Args:
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
