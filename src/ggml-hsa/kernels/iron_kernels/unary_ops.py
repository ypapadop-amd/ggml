#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for unary element-wise operations."""

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
    CoreFunctionSpec,
    arch_aligned_num_elements,
    arch_to_device,
    max_tile_size,
    tiled_tile_size,
)


def _unary_op(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """Element-wise output_tensor = op(input_tensors[0]).

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        function_spec: Core function specification.
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program.

    Raises:
        ValueError: If num_elements is not divisible by tile_size.
    """
    input_tensor = input_tensors[0]

    # Tile size and number of tiles
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size
    num_tiles = num_elements // tile_size
    if num_elements % tile_size != 0:
        msg = (
            f"num_elements ({num_elements}) must be divisible by "
            f"tile_size ({tile_size}) for correct tiling"
        )
        raise ValueError(msg)

    # AIE-array data movement with object fifos
    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Create a worker to run the task on a compute tile
    worker = None
    function = function_spec.external_function

    # Task for the core to perform with an external function
    def ext_core_fn(of_in, of_out, function):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    # Runtime operations to move data to/from the AIE-array
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime()
    with rt.sequence(input_tensor_ty, output_tensor_ty) as t:
        rt.start(worker)
        rt.fill(of_in.prod(), t[0])
        rt.drain(of_out.cons(), t[-1], wait=True)

    # Place program components (assign them resources on the device) and generate MLIR
    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for a unary op.

    Args:
        arch: Target architecture.
        op_name: Name of the unary operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        The core function spec.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{op_name}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


def _create_tiled_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> CoreFunctionSpec:
    """Create a CoreFunctionSpec for a unary op with an L1-budgeted tile size.

    Uses tiled_tile_size (a large tile that divides num_elements) instead of
    max_tile_size, so far fewer, larger tiles are streamed per dispatch. Unlike
    _create_external_function this does NOT pass -DGGML_TILE_SIZE: the tile size
    differs per tensor shape, so N stays a runtime kernel argument and one compiled
    kernel serves every shape (respecting the 32-unique-functions-per-queue limit).

    Args:
        arch: Target architecture.
        op_name: Name of the unary operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        The core function spec.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = tiled_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{op_name}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


def unary_op(
    op_name: str,
    arch: str,
    input_tensors: list,
    output_tensor,
):
    """IRON design for unary element-wise operations.

    Args:
        op_name: Name of the unary operation.
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program.

    Raises:
        ValueError: If the input/output tensor counts, contiguity, or shapes are invalid.
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

    if output_tensor.shape[1:4] != (1, 1, 1):
        msg = f"Unsupported shape ({output_tensor.shape})."
        raise ValueError(msg)

    # RELU streams large, L1-budgeted tiles (via _create_tiled_external_function) to
    # amortize the per-call acquire/release overhead that dominates its device time.
    # The tile divides num_elements exactly, so the existing _unary_op loop (which
    # requires divisibility) drives it unchanged. Other unary ops keep the max_tile_size
    # path with its compile-time GGML_TILE_SIZE fold.
    if op_name == "GGML_UNARY_OP_RELU":
        function_spec = _create_tiled_external_function(
            arch=arch,
            op_name=op_name,
            input_tensor=input_tensors[0],
            output_tensor=output_tensor,
        )
    else:
        function_spec = _create_external_function(
            arch=arch,
            op_name=op_name,
            input_tensor=input_tensors[0],
            output_tensor=output_tensor,
        )

    return _unary_op(
        arch=arch,
        input_tensors=input_tensors,
        function_spec=function_spec,
        output_tensor=output_tensor,
    )
