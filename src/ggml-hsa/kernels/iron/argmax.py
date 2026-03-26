#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the argmax operation.

Finds the index of the maximum value along the first dimension (columns) for each row.
"""

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
from aie.iron.placers import SequentialPlacer

from .softmax import get_softmax_dimensions
from .utils import arch_to_device


def argmax_op(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for argmax.

    Computes the index of the maximum value along the first dimension for each row.
    Uses row-by-row processing where each kernel invocation processes one row and
    outputs a single I32 index.

    Parameters
    ----------
        arch: Target architecture.
        input_tensors: List containing exactly one input tensor.
            The tensor must be F32 with shape [ne0, ne1, ne2, ne3] where ne0 is the
            row length (dimension over which argmax is computed) and the product
            ne1 * ne2 * ne3 is the number of rows.
        output_tensor: Output tensor of type I32 with shape [ne1, ne2, ne3]
            containing one index per row indicating the position of the maximum value.
        op_params: Operation parameters (unused for ARGMAX).

    Returns
    -------
        MLIR module representing the IRON program for argmax.

    Raises
    ------
        ValueError: If input_tensors does not contain exactly one tensor.
        ValueError: If input or output tensors are not contiguous in memory.
        ValueError: If output tensor size does not match the number of input rows.
        ValueError: If output tensor dtype is not int32.

    """
    if len(input_tensors) != 1:
        raise ValueError("Operation requires exactly one input tensor.")

    input_tensor = input_tensors[0]

    if not input_tensor.contiguous:
        raise ValueError("Input tensor must be contiguous in memory.")
    if not output_tensor.contiguous:
        raise ValueError("Output tensor must be contiguous in memory.")

    row_length, num_rows = get_softmax_dimensions(input_tensor)

    if output_tensor.numel() != num_rows:
        raise ValueError(
            f"Output tensor size ({output_tensor.numel()}) does not match the number "
            f"of input rows ({num_rows})."
        )

    if output_tensor.dtype != np.int32:
        raise ValueError(
            f"Output tensor dtype must be int32, got {output_tensor.dtype}."
        )

    function = _create_external_function(
        op_name="GGML_OP_ARGMAX",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        row_length=row_length,
    )

    # AIE-array data movement with object fifos
    # Input: one row at a time (F32)
    input_tile_ty = np.ndarray[(row_length,), np.dtype[input_tensor.dtype]]
    # Output: one index per row (I32)
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
    rt = Runtime()
    num_elements_in = row_length * num_rows
    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_rows,), np.dtype[output_tensor.dtype]]

    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def _create_external_function(
    op_name: str,
    input_tensor,
    output_tensor,
    row_length: int,
) -> ExternalFunction:
    """Creates an ExternalFunction specification for argmax.

    The external function wraps the C++ kernel that performs the actual argmax
    computation on the AIE tile. The kernel receives one row of input data and
    outputs a single I32 index.

    Parameters
    ----------
        op_name: Operation name used for function naming and compile flags.
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        row_length: Number of elements per row (ne0 dimension).

    Returns
    -------
        ExternalFunction: Configured external function specification that references
            the argmax.cc source file with appropriate compile flags for dtype and
            vector size configuration.

    """
    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
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
    return func
