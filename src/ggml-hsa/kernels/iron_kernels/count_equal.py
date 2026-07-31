#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for count_equal: number of equal elements between two I32 tensors."""

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

from .utils import fill_drain_program, max_tile_size


def count_equal_op(arch: str, input_tensors: list, output_tensor):
    """Build the count_equal IRON program.

    Args:
        arch: Target architecture.
        input_tensors: Two I32 tensors of identical shape.
        output_tensor: I64 scalar tensor, shape [1, 1, 1, 1].

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, shape mismatch, contiguity, or dtype.
    """
    if len(input_tensors) != 2:
        msg = "Operation requires exactly two input tensors."
        raise ValueError(msg)

    input_tensor0 = input_tensors[0]
    input_tensor1 = input_tensors[1]

    if not input_tensor0.contiguous:
        msg = "First input tensor must be contiguous in memory."
        raise ValueError(msg)
    if not input_tensor1.contiguous:
        msg = "Second input tensor must be contiguous in memory."
        raise ValueError(msg)
    if not output_tensor.contiguous:
        msg = "Output tensor must be contiguous in memory."
        raise ValueError(msg)

    if input_tensor0.shape != input_tensor1.shape:
        msg = f"Input tensor shapes must match: {input_tensor0.shape} != {input_tensor1.shape}"
        raise ValueError(msg)

    if input_tensor0.dtype != np.int32:
        msg = f"First input tensor dtype must be int32, got {input_tensor0.dtype}."
        raise ValueError(msg)
    if input_tensor1.dtype != np.int32:
        msg = f"Second input tensor dtype must be int32, got {input_tensor1.dtype}."
        raise ValueError(msg)

    if output_tensor.dtype != np.int64:
        msg = f"Output tensor dtype must be int64, got {output_tensor.dtype}."
        raise ValueError(msg)

    # Validate output tensor is a scalar
    if output_tensor.numel() != 1:
        msg = (
            "Output tensor must be a single-element I64 scalar (shape [1, 1, 1, 1]), "
            f"but has {output_tensor.numel()} elements."
        )
        raise ValueError(msg)

    shape = output_tensor.shape
    if len(shape) != 4 or any(dim != 1 for dim in shape):
        msg = (
            "Output tensor must have GGML scalar shape [1, 1, 1, 1], "
            f"but has shape {shape}."
        )
        raise ValueError(msg)

    total_elements = input_tensor0.numel()

    # Largest power-of-two width dividing total_elements, so every tile is full-width
    # and the kernel tail loop only handles the sub-vector remainder, not a short tile.
    tile_size = max_tile_size(arch, input_tensor0.dtype, total_elements)
    num_tiles = total_elements // tile_size

    function = _create_external_function(
        op_name="GGML_OP_COUNT_EQUAL",
        input_tensor=input_tensor0,
        tile_size=tile_size,
    )

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor0.dtype]]
    # I64 count as two I32 lanes: IRON ObjectFifos don't support I64.
    output_tile_ty = np.ndarray[(2,), np.dtype[np.int32]]

    of_in0 = ObjectFifo(input_tile_ty, name="in0")
    of_in1 = ObjectFifo(input_tile_ty, name="in1")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in0, of_in1, of_out, function, num_tiles):
        elem_out = of_out.acquire(1)

        for tile_idx in range_(num_tiles):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
            function(elem_in0, elem_in1, elem_out, tile_size, tile_idx_i32)
            of_in0.release(1)
            of_in1.release(1)

        of_out.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[
            of_in0.cons(),
            of_in1.cons(),
            of_out.prod(),
            function,
            num_tiles,
        ],
    )

    input_tensor_ty = np.ndarray[(total_elements,), np.dtype[input_tensor0.dtype]]
    output_tensor_ty = np.ndarray[(2,), np.dtype[np.int32]]

    return fill_drain_program(
        arch,
        [worker],
        [input_tensor_ty, input_tensor_ty],
        output_tensor_ty,
        [of_in0.prod(), of_in1.prod()],
        of_out.cons(),
    )


def _create_external_function(
    op_name: str,
    input_tensor,
    tile_size: int,
) -> ExternalFunction:
    """Create the ExternalFunction wrapping count_equal.cc.

    Args:
        op_name: Operation name (drives function name and compile flags).
        input_tensor: Input tensor.
        tile_size: Number of elements per tile.

    Returns:
        The configured ExternalFunction.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=f"{op_name.lower()}",
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "count_equal.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],  # in0
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],  # in1
            np.ndarray[(2,), np.dtype[np.int32]],  # out (count as 2 x I32)
            np.int32,  # tile_size
            np.int32,  # tile_idx
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        ],
    )
