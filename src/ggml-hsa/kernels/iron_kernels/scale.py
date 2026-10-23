#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the scale operation."""

import struct
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
    arch_aligned_num_elements,
    arch_to_device,
    divisor_tile_size,
    fan_out_worker_count,
    partition_units,
    vector_lanes,
)


def scale(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for scale: output = input * s + b.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor.
        op_params: s and b packed as 2 x float32 (s at byte offset 0, b at offset 4).
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

    s = struct.unpack_from("f", op_params, 0)[0]
    b = struct.unpack_from("f", op_params, 4)[0]

    function, num_elements, tile_size = _create_external_function(
        arch=arch,
        op_name="GGML_OP_SCALE",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )

    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    # Fan the tiles out across columns, one worker each with its own shim DMA, in contiguous
    # bands moved linearly. With the pipelined vector body one worker is still bound by its
    # compute on a large scale; small ones stay on one worker.
    n_workers = fan_out_worker_count(arch, num_elements, num_tiles)
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)

    of_ins = [ObjectFifo(input_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(output_tile_ty, name=f"out{w}") for w in range(n_workers)]

    def ext_core_fn(of_in, of_out, function, n_tiles):
        for _ in range_(n_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size, s, b)
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
        )
        for w in range(n_workers)
    ]

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

    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime(
        sequence,
        [
            input_tensor_ty,
            output_tensor_ty,
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> tuple[ExternalFunction, int, int]:
    """Create the scale ExternalFunction.

    Args:
        arch: Target architecture.
        op_name: Name of the operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        (func, num_elements, tile_size) where num_elements is arch-aligned.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    # The kernel has a vector body and a scalar tail, so it streams the largest tile that
    # divides the element count and fits L1, rather than one vector register per object-fifo
    # round trip.
    tile_size = divisor_tile_size(
        arch, num_elements, fifo_dtypes=(input_tensor.dtype, output_tensor.dtype)
    )

    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        "-DGGML_VECTORIZED_TILING=1",
    ]
    # Aligned vector accesses are only legal when every tile boundary is a whole vector.
    if tile_size % vector_lanes(arch, output_tensor.dtype) == 0:
        compile_flags.append("-DGGML_TILE_VECTOR_ALIGNED=1")
    if arch == "aie2":
        # The emulated fp32 multiply-add is one long chain per vector; Peano's default cap of 3
        # pipeline stages leaves its loop unpipelined (one ~76-cycle iteration per vector), and 4
        # lets it pipeline at II 19 (21 with unaligned tiles). aie2p is left as it was:
        # unmeasured there.
        compile_flags += ["-mllvm", "--aie-pipeliner-max-stagecount=4"]

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=f"{op_name.lower()}",
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "scale.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
            np.float32,
            np.float32,
        ],
        compile_flags=compile_flags,
    )
    return func, num_elements, tile_size
