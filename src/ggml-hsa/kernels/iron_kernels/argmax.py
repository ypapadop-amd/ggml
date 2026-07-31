#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for argmax: index of the max value along dim 0, per row."""

from pathlib import Path

import numpy as np
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import fill_drain_program, row_dimensions


def argmax_op(arch: str, input_tensors: list, output_tensor):
    """Build the argmax IRON program.

    Args:
        arch: Target architecture.
        input_tensors: One F32 tensor [ne0, ne1, ne2, ne3]; ne0 is the row length
            and ne1 * ne2 * ne3 the number of rows.
        output_tensor: I32 tensor holding one index per row.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, contiguity, output size, or dtype.
    """
    if len(input_tensors) != 1:
        msg = "Operation requires exactly one input tensor."
        raise ValueError(msg)

    input_tensor = input_tensors[0]

    if not input_tensor.contiguous:
        msg = "Input tensor must be contiguous in memory."
        raise ValueError(msg)
    if not output_tensor.contiguous:
        msg = "Output tensor must be contiguous in memory."
        raise ValueError(msg)

    row_length, num_rows = row_dimensions(input_tensor)

    if output_tensor.numel() != num_rows:
        msg = (
            f"Output tensor size ({output_tensor.numel()}) does not match the number "
            f"of input rows ({num_rows})."
        )
        raise ValueError(msg)

    if output_tensor.dtype != np.int32:
        msg = f"Output tensor dtype must be int32, got {output_tensor.dtype}."
        raise ValueError(msg)

    function = _create_external_function(
        op_name="GGML_OP_ARGMAX",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        row_length=row_length,
    )

    # One row streamed in, one index streamed out per worker iteration.
    input_tile_ty = np.ndarray[(row_length,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Task for the core to perform with an external function
    def ext_core_fn(of_in, of_out, function):
        for _ in range_(num_rows):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, row_length)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    # Runtime operations to move data to/from the AIE-array
    num_elements_in = row_length * num_rows
    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_rows,), np.dtype[output_tensor.dtype]]

    # Place program components and generate an MLIR module
    return fill_drain_program(
        arch,
        [worker],
        [input_tensor_ty],
        output_tensor_ty,
        [of_in.prod()],
        of_out.cons(),
    )


def _create_external_function(
    op_name: str,
    input_tensor,
    output_tensor,
    row_length: int,
) -> ExternalFunction:
    """Create the ExternalFunction wrapping argmax.cc.

    Args:
        op_name: Operation name (drives function name and compile flags).
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        row_length: Number of elements per row (ne0).

    Returns:
        The configured ExternalFunction.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=f"{op_name.lower()}",
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "argmax.cc"),
        arg_types=[
            np.ndarray[(row_length,), np.dtype[input_tensor.dtype]],
            np.ndarray[(1,), np.dtype[output_tensor.dtype]],
            np.int32,  # row_length (N)
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
