#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for the MUL_MAT pre-amble: f32 -> bf16 convert + zero-pad.

The source is a dense, contiguous f32 tensor of logical shape [d0, d1] (GGML
convention: d0 = ne[0] innermost/contiguous). The destination is a larger,
pre-zeroed bf16 buffer of shape [d0pad, d1pad] (d0pad >= d0, d1pad >= d1). The
first d1 rows are converted; each is widened from d0 to d0pad by the compute
kernel (which zero-fills the [d0, d0pad) tail), and the trailing rows
[d1, d1pad) are left as the pre-zeroed buffer contents.

Data movement is kept fully linear on both the fill and drain sides: the input
streams d1 contiguous rows of d0 elements, the output streams d1 contiguous rows
of d0pad elements into the front of the destination buffer. Row-widening is done
on the compute tile rather than via a strided shim DMA, because a single large
strided (2D) shim transfer silently exceeds the hardware BD wrap-size limits for
the shapes this kernel sees; linear transfers have no such limit.
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

from .utils import arch_to_device


def convert_pad(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the convert+pad IRON program.

    Parameters:
        arch: Target architecture.
        input_tensors: [src] dense f32 tensor of logical shape [d0, d1].
        output_tensor: padded bf16 tensor of shape [d0pad, d1pad] (d0pad >= d0,
            d1pad >= d1).
        op_params: unused (kept for the dispatch ABI).

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, or shape.

    """
    del op_params  # placement is derived from shapes, not op_params

    if len(input_tensors) != 1:
        msg = "convert_pad requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    # Two modes: f32 -> bf16 (convert + pad) or bf16 -> bf16 (pad only, operand already bf16).
    if output_tensor.dtype != bfloat16:
        msg = f"convert_pad destination must be bfloat16; got {output_tensor.dtype}."
        raise ValueError(msg)
    pad_only = src.dtype == bfloat16
    if src.dtype != np.float32 and not pad_only:
        msg = f"convert_pad source must be float32 or bfloat16; got {src.dtype}."
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "convert_pad tensors must be contiguous in memory."
        raise ValueError(msg)

    # GGML convention: shape[0] is innermost/contiguous.
    d0, d1 = src.shape[0], src.shape[1]
    d0pad, d1pad = output_tensor.shape[0], output_tensor.shape[1]

    if d0pad < d0 or d1pad < d1:
        msg = (
            f"convert_pad destination [{d0pad}, {d1pad}] must be >= source "
            f"[{d0}, {d1}] in both dimensions."
        )
        raise ValueError(msg)

    function = _create_external_function(
        src=src, output_tensor=output_tensor, d0=d0, d0pad=d0pad
    )

    row_in_ty = np.ndarray[(d0,), np.dtype[src.dtype]]
    row_out_ty = np.ndarray[(d0pad,), np.dtype[output_tensor.dtype]]

    of_in = ObjectFifo(row_in_ty, name="in")
    of_out = ObjectFifo(row_out_ty, name="out")

    def core_fn(of_in, of_out, function):
        for _ in range_(d1):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            function(row_in, row_out, d0, d0pad)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    rt = Runtime()
    # Linear fill/drain: the input is d1 contiguous rows of d0; the output is d1
    # contiguous rows of d0pad written to the front of the (pre-zeroed) buffer.
    src_ty = np.ndarray[(d0 * d1,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(d0pad * d1,), np.dtype[output_tensor.dtype]]
    with rt.sequence(src_ty, dst_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    src, output_tensor, d0: int, d0pad: int
) -> ExternalFunction:
    """Create the ExternalFunction for the convert_pad core function.

    Parameters:
        src: Source tensor (f32).
        output_tensor: Destination tensor (bf16).
        d0: Number of valid elements in one logical row.
        d0pad: Padded row width.

    Returns:
        The configured ExternalFunction.

    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Row shape is fixed per kernel instance (each shape JITs its own .o), so pass it
        # as compile-time constants: lets Peano fold the trip count and pipeline the hot loop.
        f"-DCONVERT_PAD_D0={d0}",
        f"-DCONVERT_PAD_D0PAD={d0pad}",
    ]
    # bf16 -> bf16 selects the pad-only kernel body (no dtype conversion); f32 -> bf16 keeps the
    # default convert+pad body.
    if src.dtype == bfloat16:
        compile_flags.append("-DCONVERT_PAD_PAD_ONLY=1")

    return ExternalFunction(
        name="ggml_hsa_convert_pad",
        object_file_name="ggml_hsa_convert_pad_core_function.o",
        source_file=str(current_dir / "convert_pad.cc"),
        arg_types=[
            np.ndarray[(d0,), np.dtype[src.dtype]],
            np.ndarray[(d0pad,), np.dtype[output_tensor.dtype]],
            np.int32,  # d0
            np.int32,  # d0pad
        ],
        compile_flags=compile_flags,
    )
