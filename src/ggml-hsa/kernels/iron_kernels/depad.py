#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the depad operation."""

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

# Largest hardware wrap for a shim/mem-tile DMA dimension (10-bit field). A DMA
# dimension whose element count exceeds this is rejected by the aiecc verifier
# unless the whole transfer is contiguous (and thus linearized). We keep the two
# wrap-checked dims (chunk width and row count) at or below this.
_MAX_DMA_WRAP = (1 << 10) - 1

# Largest hardware wrap for a shim/mem-tile DMA dimension (10-bit field). A DMA
# dimension whose element count exceeds this is rejected by the aiecc verifier
# unless the whole transfer is contiguous (and thus linearized). We keep the two
# wrap-checked dims (chunk width and row count) at or below this.
_MAX_DMA_WRAP = (1 << 10) - 1


def _choose_chunk(d0: int, itemsize: int) -> int:
    """Pick the per-object element count C that tiles one logical row of width d0.

    Each ObjectFifo object carries C valid elements (one contiguous slice of a row).
    C must divide d0 exactly (a fixed-size fifo object streamed nC = d0 // C times
    must cover the row with no remainder) and stay within the 10-bit DMA wrap so the
    pad-stripping access pattern below passes the aiecc verifier. A full row that
    already fits the wrap is streamed as a single object (C == d0), which reproduces
    the original one-row-per-object behavior for the small GEMM/dense shapes.

    Args:
        d0: Valid row width (elements).
        itemsize: Source element size in bytes (unused; kept for future L1 budgeting).

    Returns:
        The chosen chunk width C (divides d0, C <= d0).

    Raises:
        ValueError: If d0 exceeds the wrap limit and has no divisor within it that is
            a multiple of the 512-bit vector width (caller falls back to the host path).
    """
    del itemsize
    if d0 <= _MAX_DMA_WRAP:
        return d0

    # d0 too wide for one object: split into the largest vector-aligned divisor that
    # fits the wrap. Multiple-of-16 keeps depad.cc's vectorized f32->bf16 loop aligned.
    best = 0
    c = 16
    while c <= _MAX_DMA_WRAP:
        if d0 % c == 0:
            best = c
        c += 16
    if best:
        return best

    # No vector-aligned divisor: try any divisor within the wrap.
    for c in range(_MAX_DMA_WRAP, 0, -1):
        if d0 % c == 0:
            return c

    msg = (
        f"depad row width d0={d0} exceeds the DMA wrap limit {_MAX_DMA_WRAP} and has "
        f"no divisor within it; cannot tile on-device."
    )
    raise ValueError(msg)


def _choose_chunk(d0: int, itemsize: int) -> int:
    """Pick the per-object element count C that tiles one logical row of width d0.

    Each ObjectFifo object carries C valid elements (one contiguous slice of a row).
    C must divide d0 exactly (a fixed-size fifo object streamed nC = d0 // C times
    must cover the row with no remainder) and stay within the 10-bit DMA wrap so the
    pad-stripping access pattern below passes the aiecc verifier. A full row that
    already fits the wrap is streamed as a single object (C == d0), which reproduces
    the original one-row-per-object behavior for the small GEMM/dense shapes.

    Args:
        d0: Valid row width (elements).
        itemsize: Source element size in bytes (unused; kept for future L1 budgeting).

    Returns:
        The chosen chunk width C (divides d0, C <= d0).

    Raises:
        ValueError: If d0 exceeds the wrap limit and has no divisor within it that is
            a multiple of the 512-bit vector width (caller falls back to the host path).
    """
    del itemsize
    if d0 <= _MAX_DMA_WRAP:
        return d0

    # d0 too wide for one object: split into the largest vector-aligned divisor that
    # fits the wrap. Multiple-of-16 keeps depad.cc's vectorized f32->bf16 loop aligned.
    best = 0
    c = 16
    while c <= _MAX_DMA_WRAP:
        if d0 % c == 0:
            best = c
        c += 16
    if best:
        return best

    # No vector-aligned divisor: try any divisor within the wrap.
    for c in range(_MAX_DMA_WRAP, 0, -1):
        if d0 % c == 0:
            return c

    msg = (
        f"depad row width d0={d0} exceeds the DMA wrap limit {_MAX_DMA_WRAP} and has "
        f"no divisor within it; cannot tile on-device."
    )
    raise ValueError(msg)


def depad(
    arch: str,
    input_tensors: list,
    output_tensor,
    op_params: bytearray,
    max_workers: int | None = None,
):
    """Build the de-pad IRON program: MUL_MAT post-amble, narrowing each row from d0pad to d0.

    f32 -> bf16 fuses the per-layer cast that would otherwise follow the MUL_MAT
    as a separate CPY (bit-identical to the separate cast).

    The row width d0 may be far larger than AIE tile memory (conv MUL_MAT outputs have
    the batch*OH*OW dimension innermost, e.g. 392000). Rather than stage a whole row in
    L1, each row is streamed as nC = d0 // C contiguous chunks of C elements. The shim
    DMA access pattern reads only the valid [0, d0) prefix of every padded row (skipping
    the [d0, d0pad) tail via the row stride), so the compute kernel sees dense, gap-free
    C-element chunks and does a plain copy/convert. The access pattern orders dims as
    (chunk, row, element): the non-contiguous row stride sits between the two contiguous
    dims, preventing the aiecc canonicalizer from folding them into one oversized
    (> wrap-limit) dimension. See [[ggml-hsa-dma-strided-limit]].

    The de-pad is a grid of n_chunks x d1 independent chunk copies, fanned out across compute tiles
    along the larger of the two axes, so dense-GEMM outputs (n_chunks == 1, many rows) split by rows
    and wide-conv outputs (few rows, large n_chunks) split by chunk-group. Each worker owns an
    independent in/out ObjectFifo (its own shim-DMA path). The worker count scales with the work
    (see ``fan_out_worker_count``) and is capped at the array column count, so small ops (e.g. the
    MNIST FC logits) stay single-worker and avoid per-worker DMA/fence overhead.

    Parameters:
        arch: Target architecture.
        input_tensors: [src] padded source tensor.
        output_tensor: Dense destination tensor.
        op_params: Unused (kept for the dispatch ABI).
        max_workers: Hard cap on compute tiles to fan the copy grid across. Defaults to the array's
            column count; the actual count also scales with the work and is capped at the split axis.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, shape, or non-positive max_workers.
    """
    del op_params  # placement is derived from shapes, not op_params

    if max_workers is not None and max_workers < 1:
        msg = f"depad max_workers must be >= 1; got {max_workers}."
        raise ValueError(msg)

    if len(input_tensors) != 1:
        msg = "depad requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    # Source is the f32 padded GEMM temporary (destination f32 or bf16, the latter fusing the
    # per-layer cast) or an already-bf16 padded temporary (destination must also be bf16, plain
    # de-pad with no conversion).
    if src.dtype == np.float32:
        if output_tensor.dtype not in (np.float32, bfloat16):
            msg = (
                f"depad with an f32 source requires an f32 or bf16 destination; got "
                f"{output_tensor.dtype}."
            )
            raise ValueError(msg)
    elif src.dtype == bfloat16:
        if output_tensor.dtype != bfloat16:
            msg = (
                f"depad with a bf16 source requires a bf16 destination; got "
                f"{output_tensor.dtype}."
            )
            raise ValueError(msg)
    else:
        msg = f"depad requires an f32 or bf16 source; got {src.dtype}."
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "depad tensors must be contiguous in memory."
        raise ValueError(msg)

    # GGML convention: shape[0] is innermost/contiguous.
    d0pad, d1pad = src.shape[0], src.shape[1]
    d0, d1 = output_tensor.shape[0], output_tensor.shape[1]

    if d0pad < d0 or d1pad < d1:
        msg = (
            f"depad source [{d0pad}, {d1pad}] must be >= destination [{d0}, {d1}] "
            f"in both dimensions."
        )
        raise ValueError(msg)

    # Per-object chunk width. Each object is C valid elements; a row is nC = d0 // C of them.
    chunk = _choose_chunk(d0, src.dtype.itemsize)
    n_chunks = d0 // chunk
    num_objects = n_chunks * d1

    # The compute kernel copies/converts a full C-element chunk with no in-kernel padding
    # (the DMA already stripped it), so d0 == d0pad == chunk from the kernel's point of view.
    function = _create_external_function(
        src=src, output_tensor=output_tensor, chunk=chunk
    )

    chunk_in_ty = np.ndarray[(chunk,), np.dtype[src.dtype]]
    chunk_out_ty = np.ndarray[(chunk,), np.dtype[output_tensor.dtype]]

    # The copy grid is n_chunks (chunk-groups per row) x d1 (rows). Fan it out along the larger
    # axis: dense-GEMM outputs (n_chunks == 1) split by rows; wide-conv outputs (small d1, large
    # n_chunks) split by chunk-group. Each worker copies a contiguous band of the chosen axis. The
    # worker count scales with the work so tiny ops (e.g. the MNIST FC logits) stay single-worker.
    del num_objects  # per-worker object counts are computed from the band below
    split_rows = d1 >= n_chunks
    n_units = d1 if split_rows else n_chunks
    n_workers = fan_out_worker_count(arch, d0 * d1, n_units, max_workers)
    base, rem = divmod(n_units, n_workers)
    units_per_worker = [base + (1 if w < rem else 0) for w in range(n_workers)]

    of_ins = [ObjectFifo(chunk_in_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(chunk_out_ty, name=f"out{w}") for w in range(n_workers)]

    def core_fn(of_in, of_out, function, n_objs):
        for _ in range_(n_objs):
            chunk_in = of_in.acquire(1)
            chunk_out = of_out.acquire(1)
            function(chunk_in, chunk_out, chunk, chunk)
            of_in.release(1)
            of_out.release(1)

    # Per-worker band along the chosen axis: (n_chunks_w chunk-groups) x (rows_w rows), and the
    # flat source/destination offset to its first element.
    bands = []
    unit_off = 0
    for w in range(n_workers):
        u = units_per_worker[w]
        if split_rows:
            n_chunks_w, rows_w = n_chunks, u
            src_off, dst_off = unit_off * d0pad, unit_off * d0
        else:
            n_chunks_w, rows_w = u, d1
            src_off, dst_off = unit_off * chunk, unit_off * chunk
        bands.append((n_chunks_w, rows_w, src_off, dst_off))
        unit_off += u

    workers = [
        Worker(
            core_fn,
            fn_args=[of_ins[w].cons(), of_outs[w].prod(), function, bands[w][0] * bands[w][1]],
        )
        for w in range(n_workers)
    ]

    rt = Runtime()
    # Only the first d1 rows carry results; each padded row is d0pad wide, each dense row d0.
    src_ty = np.ndarray[(d0pad * d1,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(d0 * d1,), np.dtype[output_tensor.dtype]]
    # Access pattern dims are (chunk, row, element), outermost-first. The element run (size
    # chunk, stride 1) is the contiguous fifo object; the row dim (size rows_w, stride d0pad on
    # the source / d0 on the destination) sits between the two contiguous dims so the
    # canonicalizer cannot fold chunk*element into one oversized dimension. The two
    # wrap-checked hardware dims are the element width (chunk) and the row count (rows_w), both
    # kept within the 10-bit limit.
    with rt.sequence(src_ty, dst_ty) as (a_in, b_out):
        for worker in workers:
            rt.start(worker)
        out_taps = []
        for w in range(n_workers):
            n_chunks_w, rows_w, src_off, dst_off = bands[w]
            src_tap = TensorAccessPattern(
                (d1, d0pad), src_off, [1, n_chunks_w, rows_w, chunk], [0, chunk, d0pad, 1]
            )
            out_taps.append(
                TensorAccessPattern(
                    (d1, d0), dst_off, [1, n_chunks_w, rows_w, chunk], [0, chunk, d0, 1]
                )
            )
            rt.fill(of_ins[w].prod(), a_in, src_tap)
        # Issue all fills first, then all drains. Wait on every drain: the workers finish out of
        # order (uneven bands), so waiting only on the last-issued drain can signal completion while
        # a slower worker is still writing, and an on-queue consumer would read partial data.
        for w in range(n_workers):
            rt.drain(of_outs[w].cons(), b_out, out_taps[w], wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(src, output_tensor, chunk: int) -> ExternalFunction:
    """Create the ExternalFunction for the depad core function.

    Parameters:
        src: Source tensor (padded temporary).
        output_tensor: Destination tensor.
        chunk: Elements per streamed object (the compute kernel's fixed row width; the DMA
            has already stripped the padding, so d0 == d0pad == chunk in the kernel).

    Returns:
        The configured ExternalFunction.
    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Chunk width is fixed per kernel instance (each shape JITs its own .o), so pass it
        # as a compile-time constant: lets Peano fold the trip count and pipeline the hot loop.
        f"-DDEPAD_D0={chunk}",
    ]
    # The kernel selects its mode (plain copy vs. f32 -> bf16 convert) at compile time via
    # `if constexpr` on INPUT_DTYPE/OUTPUT_DTYPE; no extra flag is needed.

    return ExternalFunction(
        name="ggml_hsa_depad",
        object_file_name="ggml_hsa_depad_core_function.o",
        source_file=str(current_dir / "depad.cc"),
        arg_types=[
            np.ndarray[(chunk,), np.dtype[src.dtype]],
            np.ndarray[(chunk,), np.dtype[output_tensor.dtype]],
            np.int32,  # d0
            np.int32,  # d0pad
        ],
        compile_flags=compile_flags,
    )
