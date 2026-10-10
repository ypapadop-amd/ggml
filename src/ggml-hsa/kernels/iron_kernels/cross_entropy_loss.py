#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_CROSS_ENTROPY_LOSS."""

from pathlib import Path

import numpy as np
from aie.dialects import arith as arith_dialect
from aie.dialects import memref as memref_dialect
from aie.helpers.taplib import TensorAccessPattern
from aie.ir import F32Type, FloatAttr, IndexType, IntegerAttr
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_

from .utils import (
    align_to_arch,
    arch_num_columns,
    arch_to_device,
    divisor_tile_size,
    row_dimensions,
)

# The core's frame exceeds the AIE core's 1024-byte default stack: aiecc measures at least 2048
# bytes on aie2 and 2304 on aie2p (the row-pair exp and the vector log's group buffers), plus two
# runtime routines it has no size for. Without an explicit size the core silently writes past the end of its stack
# into neighbouring core data memory. Newer mlir-aie turns that into a build error ("stack_size is
# absent ... but it needs N bytes"); older toolchains built it and corrupted memory at run time.
# Same pattern as softmax.py and gemm.py.
_STACK_SIZE_BYTES = 3072

# Fewest rows a worker gets when the rows are fanned out: one group of the kernel's vector log.
_MIN_ROWS_PER_WORKER = 16

# Worker 0 takes the other workers' sums through shared memory, and a compute tile has at most
# three neighbours to share it with (the placer fails beyond that: two DMA channels are left for
# the inputs).
_MAX_WORKERS = 4

# Row byte alignment: each row must be a whole number of KERN_VEC_SIZE bytes.
KERN_VEC_SIZE = 8


def cross_entropy_loss(arch: str, input_tensors: list, output_tensor):
    """Build the cross entropy loss IRON program: mean loss over all rows.

    Args:
        arch: Target AIE architecture.
        input_tensors: Input tensors (logits, labels).
        output_tensor: Output scalar tensor.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, shape mismatch, contiguity, or a row
            length that is not already tile-aligned.
    """
    if len(input_tensors) != 2:
        msg = f"Cross entropy loss requires 2 input tensors: {len(input_tensors)}"
        raise ValueError(msg)

    logits_tensor = input_tensors[0]
    labels_tensor = input_tensors[1]

    if not logits_tensor.contiguous:
        msg = "Logits tensor must be contiguous in memory."
        raise ValueError(msg)
    if not labels_tensor.contiguous:
        msg = "Labels tensor must be contiguous in memory."
        raise ValueError(msg)
    if not output_tensor.contiguous:
        msg = "Output tensor must be contiguous in memory."
        raise ValueError(msg)

    if logits_tensor.shape != labels_tensor.shape:
        msg = "Logits and labels tensors must have the same shape."
        raise ValueError(msg)

    row_length, num_rows = row_dimensions(logits_tensor)

    # Round row_length up to a KERN_VEC_SIZE-aligned tile size.
    tile_size = align_to_arch(arch, row_length, logits_tensor.dtype, KERN_VEC_SIZE)

    # The kernel works on whole rows (no intra-row tiling), so a tile size larger than the
    # row itself would mean the ObjectFifo's fixed-size DMA transfer reads past the row
    # boundary; reject rows whose length does not already satisfy the alignment rather
    # than silently padding.
    if tile_size != row_length:
        msg = (
            f"Tile size ({tile_size}) must equal row length ({row_length}) "
            "for cross entropy loss."
        )
        raise ValueError(msg)

    # Fan the rows out over the columns: a row costs ~1 us of core time (the exp and log
    # arithmetic), against a few us per extra worker. Bands are equal, so a worker count must
    # divide num_rows, and each band holds at least one vector log group of rows.
    num_workers = next(
        w
        for w in range(min(arch_num_columns(arch), _MAX_WORKERS), 0, -1)
        if num_rows % w == 0 and (w == 1 or num_rows // w >= _MIN_ROWS_PER_WORKER)
    )

    # Stream as many whole rows per tile as fit, so the per-tile acquire/release and kernel
    # call are paid once per tile rather than once per row: a row is the "element" here.
    row_tys = [
        np.dtype((t.dtype, (row_length,))) for t in (logits_tensor, labels_tensor)
    ]
    rows_per_tile = divisor_tile_size(arch, num_rows // num_workers, row_tys)

    function = _create_external_function(
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
        rows_per_tile=rows_per_tile,
    )

    return create_reduction_program(
        arch=arch,
        function=function,
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
        num_rows=num_rows,
        rows_per_tile=rows_per_tile,
        num_workers=num_workers,
    )


def create_reduction_program(
    arch: str,
    function,
    logits_tensor,
    labels_tensor,
    output_tensor,
    tile_size: int,
    num_rows: int,
    rows_per_tile: int,
    num_workers: int,
):
    """Build the IRON program that sums per-row losses and averages by num_rows.

    The rows are split into equal contiguous bands, one per worker, each streamed linearly from
    its own shim DMA. Every worker sums its band's row losses; worker 0 adds the other workers'
    sums, in worker order, each through a one-element core-to-core fifo, and averages the total.

    Args:
        arch: Target AIE architecture.
        function: External kernel function to invoke per tile.
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output scalar tensor.
        tile_size: Row size.
        num_rows: Number of rows to reduce over.
        rows_per_tile: Rows per object-fifo tile; divides each worker's band.
        num_workers: Number of workers; divides num_rows.

    Returns:
        The resolved IRON program (MLIR module).
    """
    band_rows = num_rows // num_workers
    num_tiles = band_rows // rows_per_tile

    logits_tile_ty = np.ndarray[
        (rows_per_tile * tile_size,), np.dtype[logits_tensor.dtype]
    ]
    labels_tile_ty = np.ndarray[
        (rows_per_tile * tile_size,), np.dtype[labels_tensor.dtype]
    ]
    # The output, and each worker's band sum, is a single scalar.
    scalar_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    of_logits = [
        ObjectFifo(logits_tile_ty, name=f"logits{w}") for w in range(num_workers)
    ]
    of_labels = [
        ObjectFifo(labels_tile_ty, name=f"labels{w}") for w in range(num_workers)
    ]
    of_out = ObjectFifo(scalar_ty, name="out")
    # of_sums[w - 1] carries worker w's band sum to worker 0.
    of_sums = [ObjectFifo(scalar_ty, name=f"sum{w}") for w in range(1, num_workers)]

    def core_fn(of_logits, of_labels, of_result, function, is_root, *of_others):
        # of_result is acquired once and released only at the end so the DMA drains exactly
        # 1 scalar, regardless of num_rows.
        elem_result = of_result.acquire(1)
        c0_index = arith_dialect.ConstantOp(
            IndexType.get(), IntegerAttr.get(IndexType.get(), 0)
        ).result
        zero_f32 = arith_dialect.ConstantOp(
            F32Type.get(), FloatAttr.get(F32Type.get(), 0.0)
        ).result
        memref_dialect.StoreOp(zero_f32, elem_result, [c0_index])

        for _ in range_(num_tiles):
            elem_logits = of_logits.acquire(1)
            elem_labels = of_labels.acquire(1)

            # Adds the tile's row losses to the running total in elem_result.
            function(elem_logits, elem_labels, elem_result, tile_size, rows_per_tile)

            of_logits.release(1)
            of_labels.release(1)

        total_loss = memref_dialect.LoadOp(elem_result, [c0_index]).result
        for of_sum in of_others:
            elem_sum = of_sum.acquire(1)
            band_sum = memref_dialect.LoadOp(elem_sum, [c0_index]).result
            total_loss = arith_dialect.AddFOp(total_loss, band_sum).result
            of_sum.release(1)
        if is_root:
            nr_f32 = arith_dialect.ConstantOp(
                F32Type.get(), FloatAttr.get(F32Type.get(), float(num_rows))
            ).result
            total_loss = arith_dialect.DivFOp(total_loss, nr_f32).result
        memref_dialect.StoreOp(total_loss, elem_result, [c0_index])

        of_result.release(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[
                of_logits[w].cons(),
                of_labels[w].cons(),
                of_out.prod() if w == 0 else of_sums[w - 1].prod(),
                function,
                w == 0,
                *(of_sum.cons() for of_sum in (of_sums if w == 0 else [])),
            ],
            stack_size=_STACK_SIZE_BYTES,
        )
        for w in range(num_workers)
    ]

    num_elements = tile_size * num_rows
    band_elements = tile_size * band_rows
    logits_tensor_ty = np.ndarray[(num_elements,), np.dtype[logits_tensor.dtype]]
    labels_tensor_ty = np.ndarray[(num_elements,), np.dtype[labels_tensor.dtype]]

    def band(w):
        return TensorAccessPattern(
            (num_elements,), w * band_elements, [1, 1, 1, band_elements], [0, 0, 0, 1]
        )

    # One host buffer per ggml tensor, src then dst.
    def sequence(logits_buf, labels_buf, out_buf, logits_prods, labels_prods, out_cons):
        for w in range(num_workers):
            logits_prods[w].fill(logits_buf, band(w))
            labels_prods[w].fill(labels_buf, band(w))
        out_cons.drain(out_buf, wait=True)

    rt = Runtime(
        sequence,
        [
            logits_tensor_ty,
            labels_tensor_ty,
            scalar_ty,
            [of.prod() for of in of_logits],
            [of.prod() for of in of_labels],
            of_out.cons(),
        ],
    )
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _create_external_function(
    logits_tensor,
    labels_tensor,
    output_tensor,
    tile_size: int,
    rows_per_tile: int,
):
    """Create the ExternalFunction wrapping cross_entropy_loss.cc.

    Args:
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output tensor.
        tile_size: Row size.
        rows_per_tile: Rows per tile.

    Returns:
        The configured ExternalFunction.
    """
    arg_types = [
        np.ndarray[(rows_per_tile * tile_size,), np.dtype[logits_tensor.dtype]],
        np.ndarray[(rows_per_tile * tile_size,), np.dtype[labels_tensor.dtype]],
        np.ndarray[(1,), np.dtype[output_tensor.dtype]],  # running total (scalar)
        np.int32,  # row length (N)
        np.int32,  # rows
    ]

    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name="ggml_op_cross_entropy_loss",
        object_file_name="cross_entropy_loss_core_function.o",
        source_file=str(current_dir / "cross_entropy_loss.cc"),
        arg_types=arg_types,
    )
