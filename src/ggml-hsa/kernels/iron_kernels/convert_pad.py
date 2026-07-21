#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the convert_pad operation."""

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


def convert_pad(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
    max_workers: int | None = None,
):
    """Build the convert_pad IRON program: zero-pad a tensor, with an optional f32 -> bf16 convert.

    MUL_MAT pre-amble. Widens each of the first d1 rows from d0 to d0pad elements
    (compute kernel zero-fills the tail); trailing rows [d1, d1pad) are left as
    the pre-zeroed destination buffer contents.

    The d1 independent rows are fanned out across compute tiles: the convert is compute-bound per
    core (full RNE f32 -> bf16 emulation), so distributing rows scales throughput until the
    shim-DMA bandwidth saturates. Each worker owns an independent in/out ObjectFifo (its own
    shim-DMA path). The worker count scales with the work (see ``fan_out_worker_count``) and is
    capped at the array column count, so small ops (e.g. the MNIST FC operands) stay single-worker
    and avoid per-worker DMA/fence overhead, while large ops fan out.

    Args:
        arch: Target architecture.
        input_tensors: [src] input tensor.
        output_tensor: padded bf16 output tensor.
        op_params: unused (kept for the dispatch ABI).
        max_workers: Hard cap on compute tiles to fan the row loop across. Defaults to the array's
            column count (one worker per shim-DMA path); the actual count also scales with the work
            and is capped at d1.

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

    # Two modes: f32 -> bf16 (convert + pad) or bf16 -> bf16 (pad only, operand already bf16).
    if output_tensor.dtype != bfloat16:
        msg = f"convert_pad destination must be bfloat16; got {output_tensor.dtype}."
        raise ValueError(msg)
    pad_only = src.dtype == bfloat16
    if src.dtype != np.float32 and not pad_only:
        msg = f"convert_pad source must be float32 or bfloat16; got {src.dtype}."
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "convert_pad tensors must be contiguous in memory."
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

    function = _create_external_function(
        src=src, output_tensor=output_tensor, d0=d0, d0pad=d0pad
    )

    row_in_ty = np.ndarray[(d0,), np.dtype[src.dtype]]
    row_out_ty = np.ndarray[(d0pad,), np.dtype[output_tensor.dtype]]

    # Fan the d1 independent rows across compute tiles, scaled to the work (tiny ops stay
    # single-worker). Each worker gets a contiguous band of rows; the first (d1 % n_workers)
    # workers take one extra row so the bands cover d1 exactly.
    n_workers = fan_out_worker_count(arch, d0 * d1, d1, max_workers)
    base, rem = divmod(d1, n_workers)
    rows_per_worker = [base + (1 if w < rem else 0) for w in range(n_workers)]

    of_ins = [ObjectFifo(row_in_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(row_out_ty, name=f"out{w}") for w in range(n_workers)]

    def core_fn(of_in, of_out, function, n_rows):
        for _ in range_(n_rows):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            function(row_in, row_out, d0, d0pad)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[of_ins[w].cons(), of_outs[w].prod(), function, rows_per_worker[w]],
        )
        for w in range(n_workers)
    ]

    rt = Runtime()
    # Per-worker fill/drain: worker w reads its band of rows from the contiguous src buffer and
    # writes them to the matching band of the (pre-zeroed) dst buffer. Each band is a contiguous
    # 1-D slice (offset, size) of the flat buffer, so every worker drives its own shim DMA path.
    src_ty = np.ndarray[(d0 * d1,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(d0pad * d1,), np.dtype[output_tensor.dtype]]
    with rt.sequence(src_ty, dst_ty) as (a_in, b_out):
        for worker in workers:
            rt.start(worker)
        # Issue every worker's fill up front so their DMAs run concurrently, then wait on every
        # drain: the workers finish out of order (uneven row bands), so waiting only on the
        # last-issued drain can signal completion while a slower worker is still writing, and an
        # on-queue consumer (the MUL_MAT) would then read partial data.
        in_off = 0
        out_off = 0
        out_taps = []
        for w in range(n_workers):
            n_rows = rows_per_worker[w]
            in_len = n_rows * d0
            out_len = n_rows * d0pad
            in_tap = TensorAccessPattern(
                (d0 * d1,), offset=in_off, sizes=[1, 1, 1, in_len], strides=[0, 0, 0, 1]
            )
            out_taps.append(
                TensorAccessPattern(
                    (d0pad * d1,), offset=out_off, sizes=[1, 1, 1, out_len], strides=[0, 0, 0, 1]
                )
            )
            rt.fill(of_ins[w].prod(), a_in, tap=in_tap)
            in_off += in_len
            out_off += out_len
        for w in range(n_workers):
            rt.drain(of_outs[w].cons(), b_out, tap=out_taps[w], wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    src, output_tensor, d0: int, d0pad: int
) -> ExternalFunction:
    """Create the ExternalFunction for the convert_pad core function.

    Args:
        src: Source tensor.
        output_tensor: Destination tensor.
        d0: Number of valid elements in one logical row.
        d0pad: Padded row width.

    Returns:
        The configured ExternalFunction.
    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Row shape is fixed per kernel instance (each shape JITs its own .o), so pass it
        # as compile-time constants: lets Peano fold the trip count and pipeline the hot loop.
        f"-DCONVERT_PAD_D0={d0}",
        f"-DCONVERT_PAD_D0PAD={d0pad}",
    ]
    # bf16 -> bf16 selects the pad-only kernel body (no dtype conversion); f32 -> bf16 keeps the
    # default convert+pad body.
    if src.dtype == bfloat16:
        compile_flags.append("-DCONVERT_PAD_PAD_ONLY=1")

    return ExternalFunction(
        name="ggml_hsa_convert_pad",
        object_file_name="ggml_hsa_convert_pad_core_function.o",
        source_file=str(current_dir / "convert_pad.cc"),
        arg_types=[
            np.ndarray[(d0,), np.dtype[src.dtype]],
            np.ndarray[(d0pad,), np.dtype[output_tensor.dtype]],
            np.int32,  # d0
            np.int32,  # d0pad
        ],
        compile_flags=compile_flags,
    )
