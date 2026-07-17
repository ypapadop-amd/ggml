#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for an element-wise dtype conversion (GGML_OP_CPY cast, no shape change).

Both tensors are dense and contiguous with the same number of elements; only the dtype differs
(e.g. f32 -> bf16 or bf16 -> f32). The tensor is flattened to 1D and streamed in tiles, so the
same design serves any shape. This runs a pure cast on the device queue instead of the host copy
path (which drains the queue), letting the cast batch with surrounding dispatches.
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
from ml_dtypes import bfloat16

from .utils import arch_aligned_num_elements, arch_to_device, max_tile_size


def convert(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the element-wise dtype-conversion IRON program.

    Parameters:
        arch: Target architecture.
        input_tensors: [src] dense contiguous tensor.
        output_tensor: dense contiguous tensor, same element count, different dtype.
        op_params: unused (kept for the dispatch ABI).

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, contiguity, or element-count mismatch.

    """
    del op_params

    if len(input_tensors) != 1:
        msg = "convert requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    if not src.contiguous or not output_tensor.contiguous:
        msg = "convert tensors must be contiguous in memory."
        raise ValueError(msg)
    if src.numel() != output_tensor.numel():
        msg = (
            f"convert requires equal element counts; got src {src.numel()}, dst "
            f"{output_tensor.numel()}."
        )
        raise ValueError(msg)

    # Flatten to 1D: a cast is element-wise, so any shape streams as one contiguous run.
    num_elements = arch_aligned_num_elements(arch=arch, tensor=src)
    tile_size = max_tile_size(arch, src.dtype, num_elements)
    num_tiles = num_elements // tile_size

    function = _create_external_function(
        src=src, output_tensor=output_tensor, tile_size=tile_size
    )

    in_tile_ty = np.ndarray[(tile_size,), np.dtype[src.dtype]]
    out_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    of_in = ObjectFifo(in_tile_ty, name="in")
    of_out = ObjectFifo(out_tile_ty, name="out")

    def core_fn(of_in, of_out, function):
        for _ in range_(num_tiles):
            tile_in = of_in.acquire(1)
            tile_out = of_out.acquire(1)
            function(tile_in, tile_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    rt = Runtime()
    src_ty = np.ndarray[(num_elements,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    with rt.sequence(src_ty, dst_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(src, output_tensor, tile_size: int) -> ExternalFunction:
    """Create the ExternalFunction for the convert core function.

    Parameters:
        src: Source tensor.
        output_tensor: Destination tensor (different dtype).
        tile_size: Number of elements per streamed tile.

    Returns:
        The configured ExternalFunction.

    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Tile length is fixed per kernel instance (each shape JITs its own .o), so pass it as a
        # compile-time constant: lets Peano fold the trip count and pipeline the hot loop.
        f"-DCONVERT_N={tile_size}",
    ]
    # Select the kernel body by preprocessor (not if constexpr): the dtype macros expand to concrete
    # types, so both branches of an if constexpr would still be compiled and the unused one fails to
    # type-check. The f32 -> bf16 direction gets the vectorized bit-exact RNE path.
    if src.dtype == np.float32 and output_tensor.dtype == bfloat16:
        compile_flags.append("-DCONVERT_F32_TO_BF16=1")

    return ExternalFunction(
        name="ggml_hsa_convert",
        object_file_name="ggml_hsa_convert_core_function.o",
        source_file=str(current_dir / "convert.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[src.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,  # N
        ],
        compile_flags=compile_flags,
    )
