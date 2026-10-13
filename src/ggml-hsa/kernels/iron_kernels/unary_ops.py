#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for unary element-wise operations."""

from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
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
    arch_num_columns,
    arch_to_device,
    core_function_object,
    divisor_tile_size,
    fill_drain_program,
    max_tile_size,
    partition_units,
    tiled_tile_size,
    vector_lanes,
)

# The vectorized GELU core's frame exceeds the AIE core's 1024-byte default stack (aiecc measures
# 1600 bytes for the f32 path). Without an explicit size the core writes past the end of its own
# stack into neighbouring core data memory; newer mlir-aie turns that into a build error. Shared by
# every unary op built here -- the others need less, but the allocation is per-core and cheap.
# Same pattern as softmax.py, cross_entropy_loss.py and gemm.py. The vectorized SQRT core needs
# at least 2432 bytes on aie2 (aiecc's lower bound: it has no stack information for the soft-float
# libcalls the scalar tail reaches), so this matches binary_ops.py's 4096.
_STACK_SIZE_BYTES = 4096
# aie2p's GELU runs two 32-lane vectors of emulated f32 arithmetic per loop trip, and LLVM spills
# its loop invariants (bf16 limbs and accumulators of the constants) to a 5.3 KB frame.
_AIE2P_GELU_STACK_SIZE_BYTES = 6144


def _unary_op(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
    stack_size: int = _STACK_SIZE_BYTES,
):
    """Element-wise output_tensor = op(input_tensors[0]).

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        function_spec: Core function specification.
        output_tensor: Output tensor.
        stack_size: Stack size of each worker, in bytes.

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

    # One core streams one tile at a time through one shim-DMA channel each way, which bounds the
    # op once the kernel is fast: on aie2 a no-op RELU kernel measured the same as the real one.
    # So the tiles are split into contiguous bands, one per worker, each with its own fifo pair
    # and a linear fill/drain of its slice of the tensor. Up to one worker per array column, the
    # cap of the other fan-out designs here (see fan_out_worker_count), but without their minimum
    # size: on aie2 the split measured no slower down to 16K elements, and RELU 250000 (MNIST)
    # went from one worker's ~490 us to ~305 us. Placing the workers in separate columns
    # measured the same as the default placement.
    n_workers = min(arch_num_columns(arch), num_tiles)
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_ins = [ObjectFifo(input_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(output_tile_ty, name=f"out{w}") for w in range(n_workers)]

    function = function_spec.external_function

    def ext_core_fn(of_in, of_out, function, n_tiles):
        for _ in range_(n_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size)
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
            stack_size=stack_size,
        )
        for w in range(n_workers)
    ]

    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    if n_workers == 1:
        return fill_drain_program(
            arch,
            workers,
            input_tys=[input_tensor_ty],
            output_ty=output_tensor_ty,
            in_prods=[of_ins[0].prod()],
            out_cons=of_outs[0].cons(),
        )

    def band(w):
        return TensorAccessPattern(
            (num_elements,),
            first_tiles[w] * tile_size,
            [1, 1, tiles_per_worker[w], tile_size],
            [0, 0, tile_size, 1],
        )

    # One host buffer per ggml tensor, src then dst. Every drain is waited on: the bands can
    # finish in any order.
    def sequence(in_buf, out_buf, in_prods, out_conses):
        for w in range(n_workers):
            in_prods[w].fill(in_buf, band(w))
        for w in range(n_workers):
            out_conses[w].drain(out_buf, band(w), wait=True)

    rt = Runtime(
        sequence,
        [
            input_tensor_ty,
            output_tensor_ty,
            [of.prod() for of in of_ins],
            [of.cons() for of in of_outs],
        ],
    )
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


# Unary ops with a vectorized body in unary_ops.cc, which stream the large L1-budgeted tile
# instead of the one-vector-register tile. A kernel vectorized in unary_ops.cc but missing
# here fails to compile: transform_vector_n static_asserts on GGML_VECTORIZED_TILING, which
# only membership of this set defines. The reverse (listed here but scalar in the kernel)
# costs nothing but a larger tile.
_VECTORIZED_OPS = frozenset(
    {
        "GGML_OP_SQR",
        "GGML_OP_SQRT",
        "GGML_UNARY_OP_ABS",
        "GGML_UNARY_OP_GELU",
        "GGML_UNARY_OP_NEG",
        "GGML_UNARY_OP_RELU",
        "GGML_UNARY_OP_SGN",
        "GGML_UNARY_OP_STEP",
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
    # Two fifos (in, out), whose types need not match: a converting unary op streams a wider
    # output than input, and the budget has to charge each its own width.
    fifo_dtypes = (input_tensor.dtype, output_tensor.dtype)
    if not vectorized:
        tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)
    elif num_elements % vector_lanes(arch, input_tensor.dtype) == 0:
        # Whole vectors per tile, so the vector body runs aligned.
        tile_size = tiled_tile_size(
            arch, input_tensor.dtype, num_elements, fifo_dtypes=fifo_dtypes
        )
    else:
        # A count no vector divides: any width, run as vectors and a scalar tail. tiled_tile_size
        # would stream tiles of a few elements here (5005 f32: 5005 tiles of 1). A DMA moves
        # whole 4-byte words, so a tile of an odd number of 2-byte elements keeps that fallback.
        tile_size = divisor_tile_size(arch, num_elements, fifo_dtypes=fifo_dtypes)
        if any(tile_size * dtype.itemsize % 4 for dtype in fifo_dtypes):
            tile_size = tiled_tile_size(
                arch, input_tensor.dtype, num_elements, fifo_dtypes=fifo_dtypes
            )

    compile_flags = [
        f"-D{op_name}=1",
        f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
    ]
    if vectorized:
        compile_flags.append("-DGGML_VECTORIZED_TILING=1")
        # Only promise aligned tile boundaries when the tile really is a whole number of vector
        # registers; transform_vector_n static_asserts on this before honouring Aligned=true.
        if tile_size % vector_lanes(arch, output_tensor.dtype) == 0:
            compile_flags.append("-DGGML_TILE_VECTOR_ALIGNED=1")
    if arch == "aie2" and op_name == "GGML_OP_SQR":
        # The emulated f32 multiply is one long chain per vector; within Peano's default cap of 3
        # pipeline stages its loop runs one iteration at a time (II 72 per vector), and 4 stages
        # pipeline it at II 25. Only SQR: the cheaper bodies (RELU, SGN, STEP) schedule worse
        # with it. aie2p is left as it was: unmeasured there.
        compile_flags += ["-mllvm", "--aie-pipeliner-max-stagecount=4"]

    current_dir = Path(__file__).resolve().parent
    # Verified to compile with GGML_HSA_KERNEL_INLINE.
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
        stack_size=(
            _AIE2P_GELU_STACK_SIZE_BYTES
            if arch == "aie2p" and op_name == "GGML_UNARY_OP_GELU"
            else _STACK_SIZE_BYTES
        ),
    )
