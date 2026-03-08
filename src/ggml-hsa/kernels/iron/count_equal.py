#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""
IRON kernel implementation for the count_equal operation.

Counts the number of elements that are equal between two I32 input tensors.
The output is a single I64 value, but since IRON doesn't support I64 in ObjectFifos,
we use two I32 values (low and high parts) for the transfer.
"""

from pathlib import Path

import numpy as np

from .utils import (
    arch_to_device,
    suppress_import_pyxrt_msg,
)

suppress_import_pyxrt_msg()

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
from aie.iron.placers import SequentialPlacer

# Tile size for processing - must match TILE_SIZE in count_equal.cc
TILE_SIZE = 1024


def count_equal_op(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    IRON design for count_equal.

    Counts elements that are equal between two I32 input tensors and outputs
    a single I64 scalar with the count. Processes data in tiles and accumulates
    partial counts.

    Since IRON doesn't support I64 types in ObjectFifos, we transfer the count
    as two I32 values (low and high 32 bits). The C++ kernel writes directly
    to the I64 output buffer.

    Parameters:
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        input_tensors (list[TensorDesc]): List containing exactly two input tensors.
            Both tensors must be I32 with the same shape.
        output_tensor (TensorDesc): Output tensor of type I64 with shape [1,1,1,1]
            containing the count of equal elements.
        op_params (bytearray): Operation parameters (unused for COUNT_EQUAL).

    Returns:
        MLIR module representing the IRON program for count_equal.

    Raises:
        ValueError: If input_tensors does not contain exactly two tensors.
        ValueError: If input tensors have different shapes.
        ValueError: If input or output tensors are not contiguous in memory.
        ValueError: If input tensor dtype is not int32.
        ValueError: If output tensor dtype is not int64.
    """

    if len(input_tensors) != 2:
        raise ValueError("Operation requires exactly two input tensors.")

    input_tensor0 = input_tensors[0]
    input_tensor1 = input_tensors[1]

    if not input_tensor0.contiguous:
        raise ValueError("First input tensor must be contiguous in memory.")
    if not input_tensor1.contiguous:
        raise ValueError("Second input tensor must be contiguous in memory.")
    if not output_tensor.contiguous:
        raise ValueError("Output tensor must be contiguous in memory.")

    if input_tensor0.shape != input_tensor1.shape:
        raise ValueError(
            f"Input tensor shapes must match: {input_tensor0.shape} != {input_tensor1.shape}"
        )

    if input_tensor0.dtype != np.int32:
        raise ValueError(
            f"First input tensor dtype must be int32, got {input_tensor0.dtype}."
        )
    if input_tensor1.dtype != np.int32:
        raise ValueError(
            f"Second input tensor dtype must be int32, got {input_tensor1.dtype}."
        )

    if output_tensor.dtype != np.int64:
        raise ValueError(
            f"Output tensor dtype must be int64, got {output_tensor.dtype}."
        )

    if output_tensor.numel() != 1:
        raise ValueError(
            f"Output tensor must be a single-element I64 scalar (shape [1, 1, 1, 1]), "
            f"but has {output_tensor.numel()} elements."
        )
    total_elements = input_tensor0.numel()

    # Handle empty-tensor case explicitly to ensure the worker runs and the
    # output buffer is initialized. When there are no elements, we still
    # process a single "tile" of size 0 so that the kernel can write a
    # deterministic result (zero) to the output.
    if total_elements == 0:
        num_tiles = 1
        last_tile_size = 0
    else:
        num_tiles = (total_elements + TILE_SIZE - 1) // TILE_SIZE
        last_tile_size = total_elements - (num_tiles - 1) * TILE_SIZE
    function = _create_external_function(
        arch=arch,
        op_name="GGML_OP_COUNT_EQUAL",
        input_tensor=input_tensor0,
    )

    # AIE-array data movement with object fifos
    # Input: tiles of I32 elements from both tensors
    input_tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[input_tensor0.dtype]]
    # Output: Two I32 values representing the I64 count (low and high parts)
    # This is needed because IRON doesn't support I64 in ObjectFifos
    output_tile_ty = np.ndarray[(2,), np.dtype[np.int32]]

    of_in0 = ObjectFifo(input_tile_ty, name="in0")
    of_in1 = ObjectFifo(input_tile_ty, name="in1")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Task for the core to perform with an external function
    def ext_core_fn(of_in0, of_in1, of_out, function, num_tiles, last_tile_size):
        # Acquire output buffer once at the start
        elem_out = of_out.acquire(1)
        # Initialize count to 0 - kernel will handle this on first tile

        # Process all tiles
        for tile_idx in range_(num_tiles):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            # Convert tile_idx from index type to i32
            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
            # The kernel uses last_tile_size when processing the last tile
            function(
                elem_in0,
                elem_in1,
                elem_out,
                TILE_SIZE,
                tile_idx_i32,
                num_tiles,
                last_tile_size,
            )
            of_in0.release(1)
            of_in1.release(1)

        of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(
        ext_core_fn,
        fn_args=[
            of_in0.cons(),
            of_in1.cons(),
            of_out.prod(),
            function,
            num_tiles,
            last_tile_size,
        ],
    )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    # Pad input to be multiple of TILE_SIZE
    padded_elements = num_tiles * TILE_SIZE
    input_tensor_ty = np.ndarray[(padded_elements,), np.dtype[input_tensor0.dtype]]
    # Output: 2 x I32 = 8 bytes = 1 x I64
    output_tensor_ty = np.ndarray[(2,), np.dtype[np.int32]]

    with rt.sequence(input_tensor_ty, input_tensor_ty, output_tensor_ty) as (
        a_in0,
        a_in1,
        b_out,
    ):
        rt.start(worker)
        rt.fill(of_in0.prod(), a_in0)
        rt.fill(of_in1.prod(), a_in1)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
) -> ExternalFunction:
    """
    Creates an ExternalFunction specification for count_equal.

    The external function wraps the C++ kernel that performs the actual count_equal
    computation on the AIE tile.

    Parameters:
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        op_name (str): Operation name used for function naming and compile flags
            (e.g., "GGML_OP_COUNT_EQUAL").
        input_tensor (TensorDesc): Input tensor descriptor providing dtype information.

    Returns:
        ExternalFunction: Configured external function specification that references
            the count_equal.cc source file with appropriate compile flags.
    """

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=f"{op_name.lower()}",
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "count_equal.cc"),
        arg_types=[
            np.ndarray[(TILE_SIZE,), np.dtype[input_tensor.dtype]],  # in0
            np.ndarray[(TILE_SIZE,), np.dtype[input_tensor.dtype]],  # in1
            np.ndarray[(2,), np.dtype[np.int32]],  # out (count as 2 x I32)
            np.int32,  # tile_size
            np.int32,  # tile_idx
            np.int32,  # num_tiles
            np.int32,  # last_tile_size
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        ],
    )
    return func
