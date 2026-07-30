#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for norm (layer normalization) over dim 0, one row per tile."""

import struct
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

from .utils import arch_to_device, row_dimensions

_OP_NAME = "GGML_OP_NORM"


def norm(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """IRON design for norm: normalize each row over dim 0.

    Streams one row per tile (like softmax); the scalar C++ core computes the
    per-row mean/variance and normalizes.

    Args:
        arch: Target AIE architecture.
        input_tensors: List of one input tensor.
        output_tensor: Output tensor (same shape/dtype as the input).
        op_params: eps packed as a single float32 at byte offset 0.

    Returns:
        The resolved IRON program.
    """
    if len(input_tensors) != 1:
        msg = "Operation requires exactly one input tensor."
        raise ValueError(msg)

    input_tensor = input_tensors[0]

    if not input_tensor.contiguous or not output_tensor.contiguous:
        msg = "Input and output tensors must be contiguous in memory."
        raise ValueError(msg)

    if input_tensor.shape != output_tensor.shape:
        msg = (
            f"Input and output tensors must have the same shape: "
            f"{input_tensor.shape} vs {output_tensor.shape}"
        )
        raise ValueError(msg)

    # Pass eps as raw int32 bits (reinterpreted to float in the kernel) rather than
    # as a float immediate: the peano-compat IR pass mangles float constants that
    # LLVM emits in hex form (e.g. 1e-5 -> invalid "f0x..." literal), whereas an
    # i32 immediate always round-trips.
    eps_bits = struct.unpack_from("i", op_params, 0)[0]

    # Row = dim 0; each row is normalized independently.
    row_length, num_rows = row_dimensions(input_tensor)

    # One row per tile: the C++ core is scalar and handles any row length.
    num_elements = row_length * num_rows

    function = _create_external_function(input_tensor, output_tensor, row_length)

    input_tile_ty = np.ndarray[(row_length,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(row_length,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_out, function):
        for _ in range_(num_rows):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, row_length, eps_bits)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    rt = Runtime()
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(input_tensor, output_tensor, row_length: int):
    """Create the norm ExternalFunction.

    Args:
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        row_length: Elements per row tile.

    Returns:
        The ExternalFunction wrapping norm.cc.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=_OP_NAME.lower(),
        object_file_name=f"{_OP_NAME.lower()}_core_function.o",
        source_file=str(current_dir / "norm.cc"),
        arg_types=[
            np.ndarray[(row_length,), np.dtype[input_tensor.dtype]],
            np.ndarray[(row_length,), np.dtype[output_tensor.dtype]],
            np.int32,  # N (row length)
            np.int32,  # eps (raw float32 bits, reinterpreted in the kernel)
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
