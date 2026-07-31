#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for diag_mask_inf (causal masking) over dim 0, one row per tile."""

import struct
from pathlib import Path

import numpy as np
from aie.dialects.arith import index_cast
from aie.ir import IntegerType
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import fill_drain_program, row_dimensions

_OP_NAME = "GGML_OP_DIAG_MASK_INF"


def diag_mask_inf(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for diag_mask_inf: causal masking over dim 0.

    Streams one row per tile (like softmax) and passes the global row index to the
    core function so it can recover the per-row causal threshold.

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

    # One row per tile: the C++ core is scalar and handles any row length.
    num_elements = row_length * num_rows

    function = _create_external_function(input_tensor, output_tensor, row_length)

    input_tile_ty = np.ndarray[(row_length,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(row_length,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_out, function):
        for tile_idx in range_(num_rows):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
            function(elem_in, elem_out, row_length, nr, n_past, tile_idx_i32)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    return fill_drain_program(
        arch,
        [worker],
        [input_tensor_ty],
        output_tensor_ty,
        [of_in.prod()],
        of_out.cons(),
    )


def _create_external_function(input_tensor, output_tensor, row_length: int):
    """Create the diag_mask_inf ExternalFunction.

    Args:
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        row_length: Elements per row tile.

    Returns:
        The ExternalFunction wrapping diag_mask_inf.cc.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=_OP_NAME.lower(),
        object_file_name=f"{_OP_NAME.lower()}_core_function.o",
        source_file=str(current_dir / "diag_mask_inf.cc"),
        arg_types=[
            np.ndarray[(row_length,), np.dtype[input_tensor.dtype]],
            np.ndarray[(row_length,), np.dtype[output_tensor.dtype]],
            np.int32,  # N (row length)
            np.int32,  # nr (rows per z-slice)
            np.int32,  # n_past
            np.int32,  # tile_idx (global row index)
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
