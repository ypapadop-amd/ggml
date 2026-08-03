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
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_

from .utils import (
    CoreFunctionSpec,
    arch_aligned_num_elements,
    core_function_object,
    fill_drain_program,
    max_tile_size,
    tiled_tile_size,
)

# The vectorized GELU core's frame exceeds the AIE core's 1024-byte default stack (aiecc measures
# 1600 bytes for the f32 path). Without an explicit size the core writes past the end of its own
# stack into neighbouring core data memory; newer mlir-aie turns that into a build error. Shared by
# every unary op built here -- the others need less, but the allocation is per-core and cheap.
# Same pattern as softmax.py, cross_entropy_loss.py and gemm.py.
_STACK_SIZE_BYTES = 2048


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

    worker = Worker(
        ext_core_fn,
        fn_args=[of_in.cons(), of_out.prod(), function],
        stack_size=_STACK_SIZE_BYTES,
    )

    # Runtime operations to move data to/from the AIE-array
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    # Place program components (assign them resources on the device) and generate MLIR
    return fill_drain_program(
        arch,
        [worker],
        input_tys=[input_tensor_ty],
        output_ty=output_tensor_ty,
        in_prods=[of_in.prod()],
        out_cons=of_out.cons(),
    )


# Unary ops with a vectorized body in unary_ops.cc, which stream the large L1-budgeted tile
# instead of the one-vector-register tile. A kernel vectorized in unary_ops.cc but missing
# here fails to compile: transform_vector_n static_asserts on GGML_VECTORIZED_TILING, which
# only membership of this set defines. The reverse (listed here but scalar in the kernel)
# costs nothing but a larger tile.
_VECTORIZED_OPS = frozenset(
    {
        "GGML_OP_SQR",
        "GGML_UNARY_OP_ABS",
        "GGML_UNARY_OP_NEG",
        "GGML_UNARY_OP_RELU",
    }
)


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
    vectorized: bool,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for a unary op.

    Args:
        arch: Target architecture.
        op_name: Name of the unary operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        vectorized: Whether unary_ops.cc has a vectorized body for this op. Selects the
            large L1-budgeted tile over the one-vector-register tile, and defines
            GGML_VECTORIZED_TILING so the vectorized kernels compile.

    Returns:
        The core function spec.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size_fn = tiled_tile_size if vectorized else max_tile_size
    tile_size = tile_size_fn(arch, input_tensor.dtype, num_elements)

    compile_flags = [
        f"-D{op_name}=1",
        f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
    ]
    if vectorized:
        compile_flags.append("-DGGML_VECTORIZED_TILING=1")

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        **core_function_object(f"{op_name.lower()}_core_function"),
        source_file=str(current_dir / "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=compile_flags,
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

    # The vectorized kernels are dominated by object-fifo round trips rather than by
    # compute, so they stream the large L1-budgeted tile instead of the one-vector-register
    # tile. The tile divides num_elements exactly, so the _unary_op loop (which requires
    # divisibility) drives it unchanged. The still-scalar ops keep max_tile_size, where the
    # per-element work dominates and a bigger tile buys little.
    function_spec = _create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
        vectorized=op_name in _VECTORIZED_OPS,
    )

    return _unary_op(
        arch=arch,
        input_tensors=input_tensors,
        function_spec=function_spec,
        output_tensor=output_tensor,
    )
