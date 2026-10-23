#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for diag_mask_inf (causal masking) over dim 0, in tiles of whole rows."""

import math
import struct
from pathlib import Path

import numpy as np
from aie.dialects.arith import index_cast
from aie.ir import IntegerType
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import (
    arch_num_columns,
    arch_to_device,
    batch_slice_tap,
    divisor_tile_size,
    partition_units,
    row_dimensions,
)

_OP_NAME = "GGML_OP_DIAG_MASK_INF"


def diag_mask_inf(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for diag_mask_inf: causal masking over dim 0.

    Streams tiles of whole rows, fanned out over workers in contiguous bands (see _tiling),
    and passes each tile's first global row index to the core function so it can recover the
    per-row causal threshold.

    Args:
        arch: Target AIE architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor (same shape/dtype as the input).
        op_params: n_past packed as a single int32 at byte offset 0.

    Returns:
        The resolved IRON program.
    """
    if len(input_tensors) != 1:
        msg = "Operation requires exactly one input tensor."
        raise ValueError(msg)

    input_tensor = input_tensors[0]

    if not input_tensor.contiguous or not output_tensor.contiguous:
        msg = "Input and output tensors must be contiguous in memory."
        raise ValueError(msg)

    if input_tensor.shape != output_tensor.shape:
        msg = (
            f"Input and output tensors must have the same shape: "
            f"{input_tensor.shape} vs {output_tensor.shape}"
        )
        raise ValueError(msg)

    n_past = struct.unpack_from("i", op_params, 0)[0]

    # Row = dim 0; rows-per-slice (ne1) sets how often the causal pattern repeats.
    row_length, num_rows = row_dimensions(input_tensor)
    shape = input_tensor.shape
    nr = shape[1] if len(shape) >= 2 else 1

    num_elements = row_length * num_rows
    rows_per_tile, n_workers = _tiling(arch, input_tensor, output_tensor)
    tile_size = row_length * rows_per_tile
    num_tiles = num_rows // rows_per_tile

    function = _create_external_function(input_tensor, output_tensor, tile_size)

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)
    of_ins = [ObjectFifo(input_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(output_tile_ty, name=f"out{w}") for w in range(n_workers)]

    # The loop counts rows, one tile per step, so it yields each tile's first global row.
    def ext_core_fn(of_in, of_out, function, first_row, end_row):
        for row0 in range_(first_row, end_row, rows_per_tile):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            row0_i32 = index_cast(IntegerType.get_signless(32), row0)
            function(elem_in, elem_out, row_length, nr, n_past, row0_i32, rows_per_tile)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            ext_core_fn,
            fn_args=[
                of_ins[w].cons(),
                of_outs[w].prod(),
                function,
                first_tiles[w] * rows_per_tile,
                (first_tiles[w] + tiles_per_worker[w]) * rows_per_tile,
            ],
        )
        for w in range(n_workers)
    ]

    def band(w):
        return batch_slice_tap(
            num_tiles, tile_size, first_tiles[w], tiles_per_worker[w]
        )

    # One host buffer per ggml tensor, src then dst. Every drain is waited on: the bands can
    # finish in any order.
    def sequence(in_buf, out_buf, in_prods, out_conses):
        for w in range(n_workers):
            in_prods[w].fill(in_buf, band(w))
        for w in range(n_workers):
            out_conses[w].drain(out_buf, band(w), wait=True)

    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime(
        sequence,
        [
            input_tensor_ty,
            output_tensor_ty,
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


# Elements per worker, from at least two. Measured on aie2 (Phoenix), us per op: 600 elements
# ran in 60..63 on two workers, 70..89 on one (one tile: no double buffering), 66 on four and 80
# on eight; 12288 in 73..75 on four, 83..113 on two and 85..90 on eight; 49152 and more were
# fastest on eight.
_ELEMS_PER_WORKER = 3072

# aie2p: a worker adds a fixed cost to every dispatch, which in a graph (GPT-2 attention's scale ->
# diag_mask_inf -> soft_max) outweighs its share of the bandwidth until the op is large: workers =
# sqrt(elements / this), rounded, at least one, at most one per column.
_AIE2P_WORKER_ELEMENTS = 128 * 1024


def _tiling(arch, input_tensor, output_tensor) -> tuple[int, int]:
    """Rows per tile and worker count.

    The op is bound by data movement, not by the kernel. Each worker streams a contiguous band of
    tiles through its own shim DMA channel pair, so up to two workers per column, but each one
    also adds to the dispatch's fixed cost, so there is one per _ELEMS_PER_WORKER elements (and at
    least two); on aie2p, see _AIE2P_WORKER_ELEMENTS.
    Over the row counts that divide the rows and whose double-buffered tiles fit half of L1,
    this takes the split whose busiest worker has the fewest rows, preferring larger tiles.

    Args:
        arch: Target AIE architecture.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        The (rows_per_tile, n_workers) pair; rows_per_tile divides the row count.
    """
    row_length, num_rows = row_dimensions(input_tensor)
    # One "element" per row: a row of row_length values as a numpy subarray dtype.
    max_rows = divisor_tile_size(
        arch,
        num_rows,
        fifo_dtypes=(
            np.dtype((input_tensor.dtype, (row_length,))),
            np.dtype((output_tensor.dtype, (row_length,))),
        ),
    )
    max_workers = min(
        2 * arch_num_columns(arch), max(2, num_rows * row_length // _ELEMS_PER_WORKER)
    )
    if arch == "aie2p":
        max_workers = min(
            arch_num_columns(arch),
            max(1, round(math.sqrt(num_rows * row_length / _AIE2P_WORKER_ELEMENTS))),
        )

    def busiest_worker_rows(rows):
        num_tiles = num_rows // rows
        return -(-num_tiles // min(max_workers, num_tiles)) * rows

    rows_per_tile = min(
        (r for r in range(max_rows, 0, -1) if num_rows % r == 0),
        key=busiest_worker_rows,
    )
    return rows_per_tile, min(max_workers, num_rows // rows_per_tile)


def _create_external_function(input_tensor, output_tensor, tile_size: int):
    """Create the diag_mask_inf ExternalFunction.

    Args:
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        tile_size: Elements per tile (whole rows).

    Returns:
        The ExternalFunction wrapping diag_mask_inf.cc.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=_OP_NAME.lower(),
        object_file_name=f"{_OP_NAME.lower()}_core_function.o",
        source_file=str(current_dir / "diag_mask_inf.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,  # N (row length)
            np.int32,  # nr (rows per z-slice)
            np.int32,  # n_past
            np.int32,  # row0 (global index of the tile's first row)
            np.int32,  # rows (rows per tile)
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
