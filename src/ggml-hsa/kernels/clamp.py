#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from os import path
from typing import Tuple
import struct

import numpy as np

from utils import suppress_import_pyxrt_msg

suppress_import_pyxrt_msg()

from aie.iron import (
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
    ExternalFunction,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

from build import arch_aligned_num_elements, arch_to_device, max_tile_size


def create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> Tuple[ExternalFunction, int, int]:
    """
    Creates an ExternalFunction specification for clamp op.

    Returns:
        (func, num_elements, tile_size)
    """

    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name="ggml_op_" + op_name,
        object_file_name=f"{op_name}_core_function.o",
        source_file=path.join(current_dir, "clamp.cc"),
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


def ggml_op_clamp(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_CLAMP implementation.

    Clamps each element of the input tensor to the range [min_val, max_val].
    output[i] = max(min_val, min(input[i], max_val))

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params (bytearray): Operation parameters containing min and max values.
    """

    if len(input_tensors) != 1:
        raise ValueError("Operation requires exactly one input tensor.")

    if input_tensors[0].contiguous is False or output_tensor.contiguous is False:
        raise ValueError("Input and output tensors must be contiguous in memory.")

    if input_tensors[0].shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")

    input_tensor = input_tensors[0]

    min_val = struct.unpack_from("f", op_params, 0)[0]
    max_val = struct.unpack_from("f", op_params, 4)[0]

    function, num_elements, tile_size = create_external_function(
        arch=arch,
        op_name="clamp",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )

    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    # AIE-array data movement with object fifos
    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Task for the core to perform with an external function
    def ext_core_fn(of_in, of_out, function):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size, min_val, max_val)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())
