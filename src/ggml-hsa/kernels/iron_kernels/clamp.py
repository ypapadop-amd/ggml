#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for clamp: elementwise out = clamp(in, min_val, max_val)."""

import struct
from pathlib import Path

import numpy as np
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import (
    arch_aligned_num_elements,
    fill_drain_program,
    max_tile_size,
)


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> tuple[ExternalFunction, int, int]:
    """Create the clamp ExternalFunction.

    Tile size is the largest power-of-two vector width that evenly divides the
    arch-aligned element count, so tiles need no remainder/tail iteration.

    Args:
        arch: Target architecture.
        op_name: Name of the operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        (func, num_elements, tile_size) where num_elements is arch-aligned.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "clamp.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
            np.float32,
            np.float32,
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return func, num_elements, tile_size


def clamp(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the clamp IRON program: output = max(min_val, min(input, max_val)).

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: min_val and max_val as 2 x float32.

    Returns:
        The resolved IRON program (MLIR module).
    """
    if len(input_tensors) != 1:
        msg = "Operation requires exactly one input tensor."
        raise ValueError(msg)

    if input_tensors[0].contiguous is False or output_tensor.contiguous is False:
        msg = "Input and output tensors must be contiguous in memory."
        raise ValueError(msg)

    if input_tensors[0].shape != output_tensor.shape:
        msg = "Input and output tensors must have the same shape."
        raise ValueError(msg)

    input_tensor = input_tensors[0]

    min_val = struct.unpack_from("f", op_params, 0)[0]
    max_val = struct.unpack_from("f", op_params, 4)[0]

    function, num_elements, tile_size = _create_external_function(
        arch=arch,
        op_name="GGML_OP_CLAMP",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )

    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_out, function):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size, min_val, max_val)
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
