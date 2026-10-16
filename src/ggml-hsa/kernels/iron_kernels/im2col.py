#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_IM2COL (2D image-to-column layout transform)."""

import struct
from math import gcd
from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
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

from .utils import (
    _arch_params,
    arch_to_device,
    batch_slice_tap,
    partition_units,
    shape_defines,
)

# Lanes of the packed path's f32 vectors (im2col.cc V).
_VECTOR = 16
# A core tile's DMA buffer descriptor (aie2): 8-bit sizes, 13-bit strides.
_CORE_BD_MAX_SIZE = 255
_CORE_BD_MAX_STRIDE = 8192
# How far past the image the packed kernel's unaligned loads may read (three 32-byte blocks).
_OVERREAD_BYTES = 96
# Vector chunks the packed kernel unrolls (one period of slot groups; program memory).
_PACKED_MAX_CHUNKS = 32
# Words (two 16-bit values) per chunk of the pairs kernel (im2col.cc W), and the chunks it unrolls.
_PAIR_WORDS = 8
_PAIRS_MAX_CHUNKS = 32

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
    row_bytes = row_size * output_tensor.dtype.itemsize
    image_bytes = image_size * image_tensor.dtype.itemsize

    # Slot-major objects (see im2col.cc) for a contiguous f32 copy whose DMA walk fits a core
    # tile's buffer descriptor: several output rows per object, a whole number of vectors.
    # Otherwise, but for the pairs objects below, one row per object, written in order.
    packed = None
    if (
        arch in ("aie2", "aie2p")
        and output_tensor.dtype == np.float32
        and s0 == 1
        and _packed_walk_fits(ow, kw, col_stride)
    ):
        packed = _rows_per_object(
            arch, oh, row_bytes, image_bytes, multiple=_VECTOR // gcd(row_size, _VECTOR)
        )
    # Pixel-pair word objects (see im2col.cc), on aie2p, for a contiguous copy to a 16-bit
    # output, whose DMA cannot transpose 16-bit elements: the object is moved as 32-bit words.
    pairs = None
    if (
        arch == "aie2p"
        and output_tensor.dtype == bfloat16
        and s0 == 1
        and ow % 2 == 0
        and _pairs_walk_fits(ow // 2, col_stride)
    ):
        pairs = _rows_per_object(
            arch,
            oh,
            row_bytes,
            image_bytes,
            multiple=_pairs_period(ow // 2) // gcd(_pairs_period(ow // 2), col_stride),
            reserved_bytes=_pairs_overread_bytes(iw, ow, kw, p0, d0)
            + _pairs_table_bytes(ow // 2, col_stride),
        )
    rows = packed or pairs or 1
    obj_size = rows * row_size

    function = _create_external_function(
        image_tensor=image_tensor,
        output_tensor=output_tensor,
        image_size=image_size,
        row_size=obj_size,
        shape={
            "IW": iw,
            "IH": ih,
            "IC": ic,
            "KW": kw,
            "KH": kh,
            "OW": ow,
            "S0": s0,
            "S1": s1,
            "P0": p0,
            "P1": p1,
            "D0": d0,
            "D1": d1,
            "ROWS": rows,
            **({"PACKED": 1} if packed else {}),
            **({"PAIRS": 1} if pairs else {}),
        },
        # The pairs kernel writes 32-bit words.
        object_dtype=np.dtype(np.uint32) if pairs else None,
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
    row_tile_ty = np.ndarray[(obj_size,), np.dtype[output_tensor.dtype]]
    # A packed object holds [rows][col_stride][ow]; its DMA streams it as [rows][ow][col_stride].
    to_stream = (
        TensorAccessPattern(
            (obj_size,), 0, [rows, ow, col_stride], [col_stride * ow, 1, ow]
        )
        if packed
        else None
    )
    # A pairs object holds [rows][col_stride][ow / 2] words; its DMA streams it as
    # [rows][ow / 2][col_stride] words. The output is moved as words throughout.
    out_dtype = output_tensor.dtype
    if pairs:
        q = ow // 2
        out_dtype = np.uint32
        obj_size //= 2
        out_per_image //= 2
        row_tile_ty = np.ndarray[(obj_size,), np.dtype[np.uint32]]
        to_stream = TensorAccessPattern(
            (obj_size,), 0, [rows, q, col_stride], [col_stride * q, 1, q]
        )

    of_ins = [ObjectFifo(image_tile_ty, name=f"in{w}") for w in range(num_workers)]
    of_outs = [
        ObjectFifo(row_tile_ty, name=f"out{w}", to_stream=to_stream)
        for w in range(num_workers)
    ]

    # The outer loop over this worker's images is emitted as an AIE loop
    # (range_). The inner loop over output rows uses a plain Python range so it
    # unrolls at build time: nested range_ loops are miscompiled (mlir-aie issue
    # #1547), and OH is small. oh_idx is then a compile-time constant.
    def make_core_fn(image_count):
        def ext_core_fn(of_in, of_out, function):
            for _ in range_(image_count):
                img = of_in.acquire(1)
                for oh_idx in range(0, oh, rows):
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
    output_tensor_ty = np.ndarray[(out_per_image * n,), np.dtype[out_dtype]]

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


def _packed_walk_fits(ow: int, kw: int, col_stride: int) -> bool:
    """Whether the packed path's [rows][OW][IC*KH*KW] walk fits and its kernel stays small.

    Args:
        ow: Output width (OW).
        kw: Kernel width (KW).
        col_stride: Column slots per output pixel (IC * KH * KW).

    Returns:
        True if the walk's sizes and strides fit a core tile's buffer descriptor and the
        kernel unrolls at most _PACKED_MAX_CHUNKS vector chunks (one period of groups of KW
        runs, after which a run again starts on a vector boundary).
    """
    period = _VECTOR // gcd(kw * ow, _VECTOR)
    chunks = period * kw * -(-ow // _VECTOR)
    return (
        ow <= _CORE_BD_MAX_SIZE
        and col_stride <= _CORE_BD_MAX_SIZE
        and col_stride * ow <= _CORE_BD_MAX_STRIDE
        and chunks <= _PACKED_MAX_CHUNKS
    )


def _pairs_walk_fits(q: int, col_stride: int) -> bool:
    """Whether the pairs path's [rows][OW / 2][IC*KH*KW] walk fits and its kernel stays small.

    Args:
        q: Pixel pairs per output row (OW / 2), the words in one run.
        col_stride: Column slots per output pixel (IC * KH * KW), the runs per row.

    Returns:
        True if the walk's sizes and strides (in words) fit a core tile's buffer descriptor
        and the kernel unrolls at most _PAIRS_MAX_CHUNKS chunks (one period of runs).
    """
    chunks = _pairs_period(q) * -(-q // _PAIR_WORDS)
    return (
        q <= _CORE_BD_MAX_SIZE
        and col_stride <= _CORE_BD_MAX_SIZE
        and col_stride * q <= _CORE_BD_MAX_STRIDE
        and chunks <= _PAIRS_MAX_CHUNKS
    )


def _pairs_period(q: int) -> int:
    """Runs of the pairs kernel after which a run again starts on a vector boundary.

    Args:
        q: Words per run (OW / 2).

    Returns:
        The period, in runs (im2col.cc period).
    """
    return _PAIR_WORDS // gcd(q, _PAIR_WORDS)


def _pairs_overread_bytes(iw: int, ow: int, kw: int, p0: int, d0: int) -> int:
    """How far past an input row the pairs kernel's loads may read, in bytes.

    Chunk k of a run loads 2 * _PAIR_WORDS f32 values from column 2 * _PAIR_WORDS * k plus at
    most 1 + (KW - 1) * D0 - P0 (or from the row's start, when that is negative).

    Args:
        iw: Input width.
        ow: Output width.
        kw: Window width.
        p0: Column padding.
        d0: Column dilation.

    Returns:
        The bytes read past the end of the last input row, at least 0.
    """
    lanes = 2 * _PAIR_WORDS
    last = lanes * (-(-(ow // 2) // _PAIR_WORDS) - 1) + 1 + (kw - 1) * d0 - p0
    return 4 * max(0, max(0, last) + lanes - iw)


def _pairs_table_bytes(q: int, col_stride: int) -> int:
    """Bytes of the pairs kernel's per-slot table (im2col.cc pair_table).

    Args:
        q: Words per run (OW / 2).
        col_stride: Column slots per output pixel (IC * KH * KW), the table's entries.

    Returns:
        The table's size in bytes.
    """
    chunks = -(-q // _PAIR_WORDS)
    half = -(-(6 + 8 * chunks) // 4) * 4  # pair_half, 4-byte aligned
    return col_stride * -(-(2 * half + 2 * chunks) // 4) * 4


def _rows_per_object(
    arch: str,
    oh: int,
    row_bytes: int,
    image_bytes: int,
    multiple: int,
    reserved_bytes: int = _OVERREAD_BYTES,
) -> int | None:
    """Output rows per packed or pairs object: the largest divisor of OH that fits.

    The input image and the output objects, each double-buffered, must fit half the core
    data memory, less reserved_bytes: how far the kernel's unaligned loads may read past the
    image (_OVERREAD_BYTES for the packed kernel), as they must not run off the core's memory
    (in the MNIST designs the input buffers start at 32 and 48 KB), and the pairs kernel's
    table. The row count is also a DMA walk size.

    Args:
        arch: Target architecture.
        oh: Output rows per image (OH).
        row_bytes: Bytes in one output row.
        image_bytes: Bytes in one input image.
        multiple: The row count must be a multiple of this.
        reserved_bytes: Memory the buffers must leave: how far past the image the kernel's
            loads may read, and its tables.

    Returns:
        The rows per object, a divisor of oh and a multiple of `multiple`, or None if no
        such count fits.
    """
    budget = (
        _arch_params(arch)["core_data_mem_bytes"] // 2
        - 2 * image_bytes
        - reserved_bytes
    )
    cap = min(oh, _CORE_BD_MAX_SIZE, max(0, budget) // (2 * row_bytes))
    return next(
        (r for r in range(cap, 0, -1) if oh % r == 0 and r % multiple == 0), None
    )


def _create_external_function(
    image_tensor,
    output_tensor,
    image_size: int,
    row_size: int,
    shape: dict[str, int],
    object_dtype: np.dtype | None = None,
) -> ExternalFunction:
    """Create the ExternalFunction for the im2col core function.

    Args:
        image_tensor: Image input tensor (src1).
        output_tensor: Output tensor.
        image_size: Elements in one input image (IC * IH * IW).
        row_size: Elements in one output row (OW * IC * KH * KW).
        shape: Image and window extents, strides, padding and dilation, passed as
            -DGGML_IM2COL_<key>=<value> as well as at run time, so the kernel can be
            specialized for them (im2col.cc).
        object_dtype: The output object's element type, if not the output's: the same bytes
            as row_size output elements.

    Returns:
        The configured ExternalFunction.
    """
    op_name = "GGML_OP_IM2COL"
    current_dir = Path(__file__).resolve().parent
    out_ty = np.ndarray[(row_size,), np.dtype[output_tensor.dtype]]
    if object_dtype is not None:
        words = row_size * output_tensor.dtype.itemsize // object_dtype.itemsize
        out_ty = np.ndarray[(words,), np.dtype[object_dtype.type]]
    return ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "im2col.cc"),
        arg_types=[
            np.ndarray[(image_size,), np.dtype[image_tensor.dtype]],
            out_ty,
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
            *shape_defines("GGML_IM2COL", shape),
            # The packed kernel without the pre-RA machine scheduler: it hoisted every load
            # of the unrolled period ahead of its use and spilled them (up to ~2 KB of stack,
            # past the core's 1 KB, on 128-wide rows).
            *(["-mllvm", "-enable-misched=false"] if "PACKED" in shape else []),
        ],
    )
