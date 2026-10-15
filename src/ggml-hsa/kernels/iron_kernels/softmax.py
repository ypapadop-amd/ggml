#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for softmax over dim 0, one row per tile."""

import math
import struct
from pathlib import Path
from typing import Any

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
    fill_drain_program,
    partition_units,
    row_dimensions,
)

# The scalar softmax core's frame exceeds the AIE core's 1024-byte default stack: aiecc measures
# 1088 bytes for the plain variant, and the masked ones carry more live state. Without an explicit
# size the core silently writes past the end of its stack into neighbouring core data memory.
# Newer mlir-aie turns that into a build error ("stack_size is absent ... but it needs N bytes");
# older toolchains built it and corrupted memory at run time. See gemm.py for the same pattern.
_STACK_SIZE_BYTES = 2048

# The plain variant's aie2 kernel runs exp() on 64 lanes of emulated fp32 multiplies, which keep
# temporaries on the stack: the kernel's frame measures up to 3392 bytes over row lengths 1..32
# (the narrow rows' lane path, largest at 3 to 5 elements), plus main's.
_PLAIN_STACK_SIZE_BYTES = 4096
# aie2p's 64-lane passes keep more live 2048-bit values: frames up to 10240 bytes (7872 for the
# rows, 2368 for the exp() they call), plus main's.
_PLAIN_STACK_SIZE_BYTES_AIE2P = 10752


def softmax(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the softmax IRON program, dispatching by input count (1=plain, 2=masked, 3=sinks).

    Args:
        arch: Target AIE architecture.
        input_tensors: Input tensors (input, optional mask, optional sinks).
        output_tensor: Output tensor.
        op_params: Serialized op params (scale, max_bias).

    Returns:
        The resolved IRON program.
    """
    input_tensor_count = len(input_tensors)

    if input_tensor_count < 1 or input_tensor_count > 3:
        msg = f"Operation requires 1, 2, or 3 tensors: {input_tensor_count}"
        raise ValueError(msg)

    input_tensor = input_tensors[0]
    mask_tensor = input_tensors[1] if input_tensor_count >= 2 else None
    sink_tensor = input_tensors[2] if input_tensor_count >= 3 else None

    if not input_tensor.contiguous:
        msg = "Input tensor must be contiguous in memory."
        raise ValueError(msg)

    if not output_tensor.contiguous:
        msg = "Output tensor must be contiguous in memory."
        raise ValueError(msg)

    if input_tensor.shape != output_tensor.shape:
        msg = (
            f"Input and output tensors must have the same shape: {input_tensor.shape} "
            f"vs {output_tensor.shape}"
        )
        raise ValueError(msg)

    if mask_tensor and not mask_tensor.contiguous:
        msg = "Mask tensor must be contiguous in memory."
        raise ValueError(msg)
    if sink_tensor and not sink_tensor.contiguous:
        msg = "Sink tensor must be contiguous in memory."
        raise ValueError(msg)

    if sink_tensor:
        msg = (
            "Softmax with sink tensor is not supported on AIE. "
            "AIE tiles are limited to 2 input DMA channels, but softmax with "
            "mask and sink requires 3 input streams."
        )
        raise ValueError(msg)

    # Currently f16 mask is not supported as we use f32 vector instructions.
    if mask_tensor and mask_tensor.dtype != np.dtype("float32"):
        msg = f"Softmax with {mask_tensor.dtype} mask is not supported."
        raise ValueError(msg)

    # Unpack op_params: scale and max_bias
    scale = struct.unpack_from("f", op_params, 0)[0]
    max_bias = struct.unpack_from("f", op_params, 4)[0]

    op_name = "GGML_OP_SOFT_MAX"

    if input_tensor_count == 1:
        return create_unary_program(
            arch, op_name, input_tensor, output_tensor, scale, max_bias
        )
    if input_tensor_count == 2:
        return create_binary_program(
            arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias
        )
    # input_tensor_count == 3
    return create_ternary_program(
        arch,
        op_name,
        input_tensor,
        mask_tensor,
        sink_tensor,
        output_tensor,
        scale,
        max_bias,
    )


def create_unary_program(arch, op_name, input_tensor, output_tensor, scale, max_bias):
    """Plain softmax without mask or sink (max_bias unused).

    Streams tiles of whole rows, fanned out over workers in contiguous bands (see
    _plain_tiling): the kernel is compute-bound, so the rows split across cores.

    Args:
        arch: Target AIE architecture.
        op_name: GGML op name.
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        scale: Scale applied to the input before softmax.
        max_bias: Unused for this variant.

    Returns:
        The resolved IRON program.
    """
    row_length, num_rows = row_dimensions(input_tensor)
    rows_per_tile, n_workers = _plain_tiling(arch, input_tensor, output_tensor)
    function, num_elements, tile_size = _create_external_function(
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=None,
        sink_tensor=None,
        output_tensor=output_tensor,
        rows_per_tile=rows_per_tile,
    )

    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)
    of_ins = [ObjectFifo(input_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(output_tile_ty, name=f"out{w}") for w in range(n_workers)]

    def ext_core_fn(of_in, of_out, function, n_tiles):
        for _ in range_(n_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, row_length, rows_per_tile, scale, max_bias)
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
            stack_size=_PLAIN_STACK_SIZE_BYTES_AIE2P
            if arch == "aie2p"
            else _PLAIN_STACK_SIZE_BYTES,
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


def create_binary_program(
    arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias
):
    """Masked softmax; max_bias > 0 enables ALiBi positional encoding.

    Args:
        arch: Target AIE architecture.
        op_name: GGML op name.
        input_tensor: Input tensor.
        mask_tensor: ALiBi mask tensor.
        output_tensor: Output tensor.
        scale: Scale applied to the input before softmax.
        max_bias: ALiBi max bias.

    Returns:
        The resolved IRON program.
    """
    func_result = _create_external_function(
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        sink_tensor=None,
        output_tensor=output_tensor,
    )
    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    tile_size_mask = func_result[3]
    func_result[4]  # num_rows_mask (unused)
    num_elements_mask = func_result[5]
    n_head = func_result[6]
    rows_per_head = func_result[7]

    num_tiles_in = num_elements_in // tile_size_in
    num_tiles_mask = num_elements_mask // tile_size_mask

    assert num_elements_in % tile_size_in == 0
    assert num_elements_mask % tile_size_mask == 0
    assert num_elements_in == num_elements_mask
    assert num_tiles_in == num_tiles_mask

    input_tile_ty = np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]
    mask_tile_ty = np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]]

    of_in = ObjectFifo(input_tile_ty, name="in")
    of_mask = ObjectFifo(mask_tile_ty, name="mask")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_mask, of_out, function):
        for tile_idx in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_out = of_out.acquire(1)

            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)

            function(
                elem_in,
                elem_mask,
                elem_out,
                tile_size_in,
                scale,
                max_bias,
                n_head,
                tile_idx_i32,
                rows_per_head,
            )
            of_in.release(1)
            of_mask.release(1)
            of_out.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_in.cons(), of_mask.cons(), of_out.prod(), function],
        stack_size=_STACK_SIZE_BYTES,
    )

    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    return fill_drain_program(
        arch,
        [worker],
        input_tys=[input_tensor_ty, mask_tensor_ty],
        output_ty=output_tensor_ty,
        in_prods=[of_in.prod(), of_mask.prod()],
        out_cons=of_out.cons(),
    )


def create_ternary_program(
    arch,
    op_name,
    input_tensor,
    mask_tensor,
    sink_tensor,
    output_tensor,
    scale,
    max_bias,
):
    """Masked softmax with a per-head sink array (loaded once, indexed by tile).

    Args:
        arch: Target AIE architecture.
        op_name: GGML op name.
        input_tensor: Input tensor.
        mask_tensor: ALiBi mask tensor.
        sink_tensor: Per-head sink logits tensor.
        output_tensor: Output tensor.
        scale: Scale applied to the input before softmax.
        max_bias: ALiBi max bias.

    Returns:
        The resolved IRON program.
    """
    func_result = _create_external_function(
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        sink_tensor=sink_tensor,
        output_tensor=output_tensor,
    )

    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    tile_size_mask = func_result[3]
    func_result[4]  # num_rows_mask (unused)
    num_elements_mask = func_result[5]
    num_sinks = func_result[6]
    rows_per_head = func_result[7]

    # One sink per head, so the number of heads equals the number of sinks.
    n_head = num_sinks

    num_tiles_in = num_elements_in // tile_size_in
    num_tiles_mask = num_elements_mask // tile_size_mask

    assert num_elements_in % tile_size_in == 0
    assert num_elements_mask % tile_size_mask == 0
    assert num_elements_in == num_elements_mask
    assert num_tiles_in == num_tiles_mask

    input_tile_ty = np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]
    mask_tile_ty = np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]]

    # entire sink array passed once, not tiled
    sink_array_ty = np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]]

    of_in = ObjectFifo(input_tile_ty, name="in")
    of_mask = ObjectFifo(mask_tile_ty, name="mask")
    of_sink = ObjectFifo(sink_array_ty, name="sink", depth=1)  # Single buffer
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_mask, of_sink, of_out, function):
        # acquire sink array once at the start
        sink_array = of_sink.acquire(1)

        for tile_idx in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_out = of_out.acquire(1)

            # convert tile_idx from index type to i32
            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)

            function(
                elem_in,
                elem_mask,
                sink_array,
                elem_out,
                tile_size_in,
                tile_idx_i32,
                rows_per_head,
                scale,
                max_bias,
                n_head,
            )

            of_in.release(1)
            of_mask.release(1)
            of_out.release(1)

        # release sink array after all tiles processed
        of_sink.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_in.cons(), of_mask.cons(), of_sink.cons(), of_out.prod(), function],
        stack_size=_STACK_SIZE_BYTES,
    )

    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    sink_tensor_ty = np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    return fill_drain_program(
        arch,
        [worker],
        input_tys=[input_tensor_ty, mask_tensor_ty, sink_tensor_ty],
        output_ty=output_tensor_ty,
        in_prods=[of_in.prod(), of_mask.prod(), of_sink.prod()],
        out_cons=of_out.cons(),
    )


# Rows the plain kernel computes together when they are this short: one per lane (softmax.cc).
_LANE_ROWS = 32
_LANE_ROWS_AIE2P = 64
_LANE_ROWS_MAX_LENGTH = 32

# aie2p: a worker adds a fixed cost to every dispatch, which in a graph outweighs its share of the
# work until the op is large: workers = sqrt(elements / this), rounded to a power of two (see
# _plain_tiling).
_AIE2P_WORKER_ELEMENTS = 16384


def _plain_tiling(arch, input_tensor, output_tensor) -> tuple[int, int]:
    """Rows per tile and worker count for the plain softmax.

    Each worker streams a contiguous band of tiles through its own shim DMA channel pair, so up
    to two workers per column. Over the row counts that divide the rows and whose
    double-buffered tiles fit half of L1, this takes the split whose busiest worker has the
    least work, preferring larger tiles: rows up to _LANE_ROWS_MAX_LENGTH long cost about the
    same per group of _LANE_ROWS rows (_LANE_ROWS_AIE2P on aie2p) however full, longer ones cost
    per row. MNIST's 500 rows of 10 go on aie2 as 20 tiles of 25 rows over 8 workers (3 groups
    for the busiest), not 4 tiles of 125 over 4 (4 groups). aie2p takes fewer workers: see
    _AIE2P_WORKER_ELEMENTS.

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
    lane_rows = (
        (_LANE_ROWS_AIE2P if arch == "aie2p" else _LANE_ROWS)
        if row_length <= _LANE_ROWS_MAX_LENGTH
        else 1
    )
    max_workers = 2 * arch_num_columns(arch)
    if arch == "aie2p":
        # Fastest or within noise of it inside a graph (GPT-2's scale, mask and softmax; MNIST's
        # bias add and softmax): 1 worker for MNIST and every decode shape, 2 at [64, 64, 12],
        # 6-8 at [256, 256, 12], 16 at [1024, 1024, 12]. A power of two divides GPT-2's row
        # counts, so the bands keep whole tiles of many rows.
        workers = math.sqrt(num_rows * row_length / _AIE2P_WORKER_ELEMENTS)
        max_workers = min(max_workers, max(1, 2 ** round(math.log2(workers))))

    def busiest_worker_work(rows):
        num_tiles = num_rows // rows
        workers = min(max_workers, num_tiles)
        return -(-num_tiles // workers) * -(-rows // lane_rows)

    rows_per_tile = min(
        (r for r in range(max_rows, 0, -1) if num_rows % r == 0),
        key=busiest_worker_work,
    )
    return rows_per_tile, min(max_workers, num_rows // rows_per_tile)


def _create_external_function(
    op_name: str,
    input_tensor,
    mask_tensor: Any | None,
    sink_tensor: Any | None,
    output_tensor,
    rows_per_tile: int = 1,
) -> tuple:
    """Create an external function specification for softmax variants.

    Args:
        op_name: GGML op name.
        input_tensor: Input tensor.
        mask_tensor: Optional ALiBi mask tensor.
        sink_tensor: Optional per-head sink logits tensor.
        output_tensor: Output tensor.
        rows_per_tile: Rows per tile; only the plain variant takes more than one.

    Returns:
        A variable-length tuple depending on which of mask_tensor/sink_tensor are set:
          none: (func, num_elements_in, tile_size_in)
          mask only: (..., tile_size_mask, num_rows_mask, num_elements_mask, n_head,
            rows_per_head)
          mask+sink: (..., tile_size_mask, num_rows_mask, num_elements_mask, num_sinks,
            rows_per_head)
    """
    row_length_in, num_rows_in = row_dimensions(input_tensor)

    # Use actual row length - no padding. The host data is contiguous with
    # row_length elements per row, so tile_size must match.
    # The C++ kernel is a pure scalar implementation that works with any size.
    tile_size_in = row_length_in * rows_per_tile

    num_elements_in = row_length_in * num_rows_in

    arg_types = [np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]]
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        f"-DGGML_SOFTMAX_ROW_LEN={row_length_in}",
    ]

    result_extra = []

    if mask_tensor:
        row_length_mask, num_rows_mask = row_dimensions(mask_tensor)
        # Use actual row length - no padding (same reason as input tensor)
        tile_size_mask = row_length_mask
        num_elements_mask = tile_size_mask * num_rows_mask

        arg_types.append(np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]])
        compile_flags.append(f"-DMASK_DTYPE={dtype_to_str(mask_tensor.dtype)}")
        result_extra.extend([tile_size_mask, num_rows_mask, num_elements_mask])

        input_shape = input_tensor.shape
        if len(input_shape) >= 3:
            n_head = input_shape[2]  # ne02
        elif len(input_shape) == 2:
            n_head = 1
        else:
            n_head = 1

        rows_per_head = num_rows_in // n_head if n_head > 0 else 1
        result_extra.extend([n_head, rows_per_head])

    if sink_tensor:
        # sink is 1D: one value per head
        num_sinks = sink_tensor.shape[0]
        rows_per_head = num_rows_in // num_sinks if num_sinks > 0 else 1

        arg_types.append(np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]])
        compile_flags.append(f"-DSINK_DTYPE={dtype_to_str(sink_tensor.dtype)}")
        if mask_tensor:
            result_extra = result_extra[:-2]
        result_extra.extend([num_sinks, rows_per_head])

    # output tensor
    arg_types.append(np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]])
    compile_flags.append(f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}")

    arg_types.append(np.int32)  # tile_size (row length)
    if mask_tensor is None and sink_tensor is None:
        arg_types.append(np.int32)  # rows per tile

    # additional arguments for sink variant
    if sink_tensor is not None:
        arg_types.append(np.int32)  # tile_idx
        arg_types.append(np.int32)  # rows_per_head

    arg_types.append(np.float32)  # scale
    arg_types.append(np.float32)  # max_bias

    # add ALiBi parameters for mask variant (without sink)
    if mask_tensor is not None and sink_tensor is None:
        arg_types.append(np.int32)  # n_head
        arg_types.append(np.int32)  # tile_idx (passed dynamically)
        arg_types.append(np.int32)  # rows_per_head

    # sink variant needs n_head for the ALiBi slope
    if sink_tensor is not None:
        arg_types.append(np.int32)  # n_head
    # determine function name and compile directive
    function_name = op_name.lower()
    if mask_tensor and sink_tensor:
        function_name = function_name + "_with_mask_and_sinks"
        compile_flags.append(f"-D{op_name}_WITH_MASK_AND_SINKS=1")
    elif mask_tensor:
        function_name = function_name + "_with_mask"
        compile_flags.append(f"-D{op_name}_WITH_MASK=1")
    else:
        compile_flags.append(f"-D{op_name}=1")

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=function_name,
        object_file_name=f"{function_name}_core_function.o",
        source_file=str(current_dir / "softmax.cc"),
        arg_types=arg_types,
        compile_flags=compile_flags,
    )

    return (func, num_elements_in, tile_size_in, *result_extra)
