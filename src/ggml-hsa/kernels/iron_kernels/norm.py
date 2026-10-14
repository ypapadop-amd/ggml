#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for norm (layer normalization) over dim 0, tiles of whole rows over workers."""

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

from .utils import (
    arch_num_columns,
    arch_to_device,
    batch_slice_tap,
    divisor_tile_size,
    partition_units,
    row_dimensions,
)

_OP_NAME = "GGML_OP_NORM"

# The core's stack, over the 1024-byte default: the kernel's frame measures 1088 bytes on aie2 and
# 2368 on aie2p, plus main's.
_STACK_SIZE_BYTES = {"aie2": 2048, "aie2p": 3072}

# How many more rows than the fewest possible a worker may take to get fewer, larger tiles.
_ROW_SLACK = 1.15


def norm(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for norm: normalize each row over dim 0.

    Streams tiles of whole rows, fanned out over workers in contiguous bands (see _tiling): the
    kernel is compute-bound, so the rows split across cores.

    Args:
        arch: Target AIE architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor (same shape/dtype as the input).
        op_params: eps packed as a single float32 at byte offset 0.

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

    # Pass eps and 1/N as raw int32 bits (reinterpreted to float in the kernel) rather than
    # as float immediates: the peano-compat IR pass mangles float constants that LLVM emits in
    # hex form (e.g. 1e-5 -> invalid "f0x..." literal), whereas an i32 immediate always
    # round-trips.
    eps_bits = struct.unpack_from("i", op_params, 0)[0]

    # Row = dim 0; each row is normalized independently.
    row_length, num_rows = row_dimensions(input_tensor)
    inv_n_bits = int((np.float32(1) / np.float32(row_length)).view(np.int32))

    rows_per_tile, n_workers = _tiling(arch, input_tensor, output_tensor)
    tile_size = row_length * rows_per_tile
    num_tiles = num_rows // rows_per_tile
    num_elements = row_length * num_rows

    function = _create_external_function(input_tensor, output_tensor, tile_size)

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)
    of_ins = [ObjectFifo(input_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(output_tile_ty, name=f"out{w}") for w in range(n_workers)]

    def ext_core_fn(of_in, of_out, function, n_tiles):
        for _ in range_(n_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, row_length, rows_per_tile, eps_bits, inv_n_bits)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            ext_core_fn,
            fn_args=[
                of_ins[w].cons(),
                of_outs[w].prod(),
                function,
                tiles_per_worker[w],
            ],
            stack_size=_STACK_SIZE_BYTES[arch],
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


def _tiling(arch, input_tensor, output_tensor) -> tuple[int, int]:
    """Rows per tile and worker count.

    Each worker streams a contiguous band of tiles through its own shim DMA channel pair, so up
    to two workers per column. Over the row counts that divide the rows and whose
    double-buffered tiles fit half of L1, this takes the largest whose busiest worker has at
    most _ROW_SLACK times the fewest rows any split gives it: a tile costs about as much as a
    few rows (the object fifo round trip and its rows' shared reciprocal square roots), so
    trading a few rows for far fewer tiles wins. On aie2, 500 rows of 10 or of 100 go as 50
    tiles of 10 (7 tiles, 70 rows for the busiest worker), measured faster than 500 tiles of 1
    (63 rows: 167 vs 93 us at 10 wide) and than 20 tiles of 25 (75 rows: 96 us).

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
    max_workers = 2 * arch_num_columns(arch)

    def busiest_worker_rows(rows):
        num_tiles = num_rows // rows
        workers = min(max_workers, num_tiles)
        return -(-num_tiles // workers) * rows

    candidates = [r for r in range(max_rows, 0, -1) if num_rows % r == 0]
    fewest = min(busiest_worker_rows(r) for r in candidates)
    rows_per_tile = next(
        r for r in candidates if busiest_worker_rows(r) <= _ROW_SLACK * fewest
    )
    return rows_per_tile, min(max_workers, num_rows // rows_per_tile)


def _create_external_function(input_tensor, output_tensor, tile_size: int):
    """Create the norm ExternalFunction.

    Args:
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        tile_size: Elements per tile (whole rows).

    Returns:
        The ExternalFunction wrapping norm.cc.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=_OP_NAME.lower(),
        object_file_name=f"{_OP_NAME.lower()}_core_function.o",
        source_file=str(current_dir / "norm.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,  # N (row length)
            np.int32,  # rows in the tile
            np.int32,  # eps (raw float32 bits, reinterpreted in the kernel)
            np.int32,  # 1/N (raw float32 bits)
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
