#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the convert_pad operation."""

import math
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

from .utils import arch_to_device, fan_out_worker_count

# Fewest valid elements worth a worker of its own. The convert is cheap per element, so a worker
# pays off well below the shared default: fanning the MNIST FC operands ([784, 500], [500, 500])
# out over 3-4 workers halved their time on aie2, while ops of a few thousand elements stay at the
# dispatch floor either way.
_MIN_ELEMS_PER_WORKER = 64 * 1024

# Most bytes (input plus output) one ObjectFifo object of rows may hold. Each object costs a DMA
# buffer descriptor and lock handshake at both ends and a core-function call, which dominate rows
# of a few dozen bytes; batching rows amortizes that. Objects are double-buffered.
_MAX_OBJECT_BYTES = 4096


def _rows_per_object(
    d0: int, d1: int, d0pad: int, d1pad: int, in_size: int, out_size: int
) -> int:
    """Rows to stream per ObjectFifo object.

    The largest count that divides both the source rows and the padding rows (every object is
    whole) and keeps an object within _MAX_OBJECT_BYTES. Rows are only batched when an output
    row's byte size is a multiple of 16, so every row of an object starts 16-byte aligned for the
    kernel's aligned stores.

    Args:
        d0: Valid elements per row.
        d1: Source rows.
        d0pad: Padded row width.
        d1pad: Padded row count.
        in_size: Bytes per input element.
        out_size: Bytes per output element.

    Returns:
        The rows per object (>= 1).
    """
    if (d0pad * out_size) % 16 != 0:
        return 1
    common = math.gcd(d1, d1pad - d1)
    cap = max(1, _MAX_OBJECT_BYTES // (d0 * in_size + d0pad * out_size))
    return next(r for r in range(min(cap, common), 0, -1) if common % r == 0)


def convert_pad(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
    max_workers: int | None = None,
):
    """Build the convert_pad IRON program: zero-pad a tensor, with an optional f32 -> bf16 convert.

    MUL_MAT pre-amble. Writes the whole [d0pad, d1pad] destination: each of the first d1 rows is
    widened from d0 to d0pad elements (the compute kernel zero-fills the tail), and the trailing
    rows [d1, d1pad), which have no source row, are zero-filled by the cores. The caller need not
    pre-zero the destination.

    Rows travel in ObjectFifo objects of several rows when they are short (see
    ``_rows_per_object``), and the objects are fanned out across compute tiles, each worker with
    an independent in/out ObjectFifo (its own shim-DMA path), until the DMA bandwidth saturates.
    The worker count scales with the work (see ``_MIN_ELEMS_PER_WORKER``) and is capped at the
    array column count, so ops of a few thousand elements stay single-worker.

    Args:
        arch: Target architecture.
        input_tensors: [src] input tensor.
        output_tensor: padded output tensor: bf16, or the source dtype (bf16 or f32) to pad
            without converting.
        op_params: unused (kept for the dispatch ABI).
        max_workers: Hard cap on compute tiles to fan the row loop across. Defaults to the array's
            column count (one worker per shim-DMA path); the actual count also scales with the work
            and is capped at the number of objects.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, shape, or non-positive max_workers.
    """
    del op_params  # placement is derived from shapes, not op_params

    if max_workers is not None and max_workers < 1:
        msg = f"convert_pad max_workers must be >= 1; got {max_workers}."
        raise ValueError(msg)

    if len(input_tensors) != 1:
        msg = "convert_pad requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    # Two modes: f32 -> bf16 (convert + pad), or pad only when the operand already has the
    # destination dtype (bf16 -> bf16, or f32 -> f32 for a GEMM that converts f32 on the core).
    pad_only = src.dtype == output_tensor.dtype
    if pad_only:
        if src.dtype not in (bfloat16, np.float32):
            msg = f"convert_pad pad-only dtype must be bfloat16 or float32; got {src.dtype}."
            raise ValueError(msg)
    elif src.dtype != np.float32 or output_tensor.dtype != bfloat16:
        msg = (
            "convert_pad converts float32 -> bfloat16 only; got "
            f"{src.dtype} -> {output_tensor.dtype}."
        )
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "convert_pad tensors must be contiguous in memory."
        raise ValueError(msg)

    # The tensors are addressed as 2D [d0, d1]; higher dimensions are never streamed, so a
    # 3D/4D operand would silently have all but its first slice dropped.
    for name, t in (("source", src), ("destination", output_tensor)):
        if t.shape[2] != 1 or t.shape[3] != 1:
            msg = (
                f"convert_pad {name} must be 2D; got shape {t.shape} "
                f"(ne[2]={t.shape[2]}, ne[3]={t.shape[3]})."
            )
            raise ValueError(msg)

    # GGML convention: shape[0] is innermost/contiguous.
    d0, d1 = src.shape[0], src.shape[1]
    d0pad, d1pad = output_tensor.shape[0], output_tensor.shape[1]

    if d0pad < d0 or d1pad < d1:
        msg = (
            f"convert_pad destination [{d0pad}, {d1pad}] must be >= source "
            f"[{d0}, {d1}] in both dimensions."
        )
        raise ValueError(msg)

    # Rows travel in objects of `rows` rows each, for source and padding rows alike.
    rows = _rows_per_object(
        d0,
        d1,
        d0pad,
        d1pad,
        np.dtype(src.dtype).itemsize,
        np.dtype(output_tensor.dtype).itemsize,
    )

    function, zero_function = _create_external_functions(
        src=src, output_tensor=output_tensor, d0=d0, d0pad=d0pad, rows=rows
    )

    row_in_ty = np.ndarray[(rows * d0,), np.dtype[src.dtype]]
    row_out_ty = np.ndarray[(rows * d0pad,), np.dtype[output_tensor.dtype]]

    # Fan the d1 / rows independent objects across compute tiles, scaled to the work (tiny ops
    # stay single-worker). Each worker gets a contiguous band of objects; the first
    # (n_objects % n_workers) workers take one extra so the bands cover d1 exactly.
    n_objects = d1 // rows
    n_workers = fan_out_worker_count(
        arch, d0 * d1, n_objects, max_workers, _MIN_ELEMS_PER_WORKER
    )
    base, rem = divmod(n_objects, n_workers)
    rows_per_worker = [rows * (base + (1 if w < rem else 0)) for w in range(n_workers)]
    # The d1pad - d1 trailing padding rows have no source row; each worker also zero-fills a
    # contiguous band of them. The last workers take the extra objects, balancing the first
    # workers' extra source objects.
    zbase, zrem = divmod((d1pad - d1) // rows, n_workers)
    zero_rows_per_worker = [
        rows * (zbase + (1 if w >= n_workers - zrem else 0)) for w in range(n_workers)
    ]

    of_ins = [ObjectFifo(row_in_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(row_out_ty, name=f"out{w}") for w in range(n_workers)]

    def core_fn(of_in, of_out, function, zero_function, n_objects, n_zero_objects):
        for _ in range_(n_objects):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            function(row_in, row_out, d0, d0pad)
            of_in.release(1)
            of_out.release(1)
        # Then the worker's band of padding rows: output only, no input.
        if n_zero_objects > 0:
            for _ in range_(n_zero_objects):
                row_out = of_out.acquire(1)
                zero_function(row_out)
                of_out.release(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[
                of_ins[w].cons(),
                of_outs[w].prod(),
                function,
                zero_function,
                rows_per_worker[w] // rows,
                zero_rows_per_worker[w] // rows,
            ],
        )
        for w in range(n_workers)
    ]

    # Per-worker fill/drain: worker w reads its band of rows from the contiguous src buffer and
    # writes them to the matching band of the dst buffer, then writes its band of zero rows to the
    # matching band of the trailing [d0pad, d1pad - d1] region. Each band is a contiguous 1-D slice
    # (offset, size) of the flat buffer, so every worker drives its own shim DMA path. The fills
    # cover the d1 source rows; the drains cover all d1pad destination rows.
    src_ty = np.ndarray[(d0 * d1,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(d0pad * d1pad,), np.dtype[output_tensor.dtype]]

    def sequence(a_in, b_out, in_prods, out_conses):
        # Issue every worker's fill up front so their DMAs run concurrently, then wait on every
        # drain: the workers finish out of order (uneven row bands), so waiting only on the
        # last-issued drain can signal completion while a slower worker is still writing, and an
        # on-queue consumer (the MUL_MAT) would then read partial data.
        #
        # Each worker's output stream is its source rows then its zero rows, so its drains are
        # issued in that order (DMA tasks on one channel run in issue order).

        def out_tap(offset, length):
            return TensorAccessPattern(
                (d0pad * d1pad,),
                offset=offset,
                sizes=[1, 1, 1, length],
                strides=[0, 0, 0, 1],
            )

        in_off = 0
        out_off = 0
        zero_off = d1 * d0pad
        out_taps = []
        for w in range(n_workers):
            n_rows = rows_per_worker[w]
            in_len = n_rows * d0
            out_len = n_rows * d0pad
            zero_len = zero_rows_per_worker[w] * d0pad
            in_tap = TensorAccessPattern(
                (d0 * d1,), offset=in_off, sizes=[1, 1, 1, in_len], strides=[0, 0, 0, 1]
            )
            taps = [out_tap(out_off, out_len)]
            if zero_len > 0:
                taps.append(out_tap(zero_off, zero_len))
            out_taps.append(taps)
            in_prods[w].fill(a_in, in_tap)
            in_off += in_len
            out_off += out_len
            zero_off += zero_len
        for w in range(n_workers):
            for tap in out_taps[w]:
                out_conses[w].drain(b_out, tap, wait=True)

    rt = Runtime(
        sequence,
        [
            src_ty,
            dst_ty,
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )

    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _create_external_functions(
    src, output_tensor, d0: int, d0pad: int, rows: int
) -> tuple[ExternalFunction, ExternalFunction]:
    """Create the ExternalFunctions for the convert_pad core functions.

    Args:
        src: Source tensor.
        output_tensor: Destination tensor.
        d0: Number of valid elements in one logical row.
        d0pad: Padded row width.
        rows: Rows per call (per ObjectFifo object).

    Returns:
        The row convert/pad function and the padding-row zero-fill function, both built into one
        object file.
    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Row shape is fixed per kernel instance (each shape JITs its own .o), so pass it
        # as compile-time constants: lets Peano fold the trip count and pipeline the hot loop.
        f"-DCONVERT_PAD_D0={d0}",
        f"-DCONVERT_PAD_D0PAD={d0pad}",
        f"-DCONVERT_PAD_ROWS={rows}",
    ]
    # A same-dtype pair selects the pad-only kernel body (no dtype conversion); f32 -> bf16 keeps
    # the default convert+pad body.
    if src.dtype == output_tensor.dtype:
        compile_flags.append("-DCONVERT_PAD_PAD_ONLY=1")

    object_file_name = "ggml_hsa_convert_pad_core_function.o"
    source_file = str(current_dir / "convert_pad.cc")
    row_out_ty = np.ndarray[(rows * d0pad,), np.dtype[output_tensor.dtype]]
    function = ExternalFunction(
        name="ggml_hsa_convert_pad",
        object_file_name=object_file_name,
        source_file=source_file,
        arg_types=[
            np.ndarray[(rows * d0,), np.dtype[src.dtype]],
            row_out_ty,
            np.int32,  # d0
            np.int32,  # d0pad
        ],
        compile_flags=compile_flags,
    )
    zero_function = ExternalFunction(
        name="ggml_hsa_convert_pad_zero_row",
        object_file_name=object_file_name,
        source_file=source_file,
        arg_types=[row_out_ty],
        compile_flags=compile_flags,
    )
    return function, zero_function
