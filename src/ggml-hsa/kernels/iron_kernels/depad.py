#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for the MUL_MAT post-amble: de-pad an f32 result.

The source is a padded, contiguous f32 buffer of shape [d0pad, d1pad] (GGML
convention: d0pad = ne[0] innermost/contiguous). The destination is a dense
tensor of logical shape [d0, d1] (d0 <= d0pad, d1 <= d1pad). The first d1 rows
are gathered; each is narrowed from d0pad to d0 by the compute kernel (which
copies only the first d0 elements).

Two modes, selected by the destination dtype: f32 -> f32 (plain de-pad) or
f32 -> bf16 (de-pad + convert). The bf16 mode fuses the per-layer f32->bf16 cast
that would otherwise follow the MUL_MAT as a separate CPY, so the padded GEMM
result is narrowed as it is de-padded (bit-identical to the separate cast).

Data movement is kept fully linear on both the fill and drain sides: the input
streams the first d1 rows of the padded buffer (d0pad elements each, contiguous
from the buffer start), and the output streams d1 contiguous rows of d0. Row-
narrowing is done on the compute tile rather than via a strided shim DMA, because
a single large strided (2D) shim transfer silently exceeds the hardware BD
wrap-size limits for the shapes this kernel sees; linear transfers have no such
limit.
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


def depad(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the de-pad IRON program.

    Parameters:
        arch: Target architecture.
        input_tensors: [src] padded f32 buffer of shape [d0pad, d1pad].
        output_tensor: dense f32 tensor of logical shape [d0, d1] (d0 <= d0pad,
            d1 <= d1pad).
        op_params: unused (kept for the dispatch ABI).

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, or shape.

    """
    del op_params  # placement is derived from shapes, not op_params

    if len(input_tensors) != 1:
        msg = "depad requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    # Source is always the f32 padded GEMM temporary. Destination is f32 (plain de-pad) or
    # bf16 (de-pad + convert, fusing the per-layer cast).
    convert = output_tensor.dtype == bfloat16
    if src.dtype != np.float32 or (output_tensor.dtype != np.float32 and not convert):
        msg = (
            f"depad requires an f32 source and an f32 or bf16 destination; got src "
            f"{src.dtype}, dst {output_tensor.dtype}."
        )
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "depad tensors must be contiguous in memory."
        raise ValueError(msg)

    # GGML convention: shape[0] is innermost/contiguous.
    d0pad, d1pad = src.shape[0], src.shape[1]
    d0, d1 = output_tensor.shape[0], output_tensor.shape[1]

    if d0pad < d0 or d1pad < d1:
        msg = (
            f"depad source [{d0pad}, {d1pad}] must be >= destination [{d0}, {d1}] "
            f"in both dimensions."
        )
        raise ValueError(msg)

    function = _create_external_function(
        src=src, output_tensor=output_tensor, d0=d0, d0pad=d0pad
    )

    row_in_ty = np.ndarray[(d0pad,), np.dtype[src.dtype]]
    row_out_ty = np.ndarray[(d0,), np.dtype[output_tensor.dtype]]

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
    # Linear fill/drain: read the first d1 rows of the padded buffer (d0pad each,
    # contiguous from the start) and write d1 dense rows of d0.
    src_ty = np.ndarray[(d0pad * d1,), np.dtype[src.dtype]]
    dst_ty = np.ndarray[(d0 * d1,), np.dtype[output_tensor.dtype]]
    with rt.sequence(src_ty, dst_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    src, output_tensor, d0: int, d0pad: int
) -> ExternalFunction:
    """Create the ExternalFunction for the depad core function.

    Parameters:
        src: Source tensor (f32 padded temporary).
        output_tensor: Destination tensor (f32 or bf16).
        d0: Number of valid elements in one logical row.
        d0pad: Padded input row width.

    Returns:
        The configured ExternalFunction.

    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(src.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        # Row width is fixed per kernel instance (each shape JITs its own .o), so pass it
        # as a compile-time constant: lets Peano fold the trip count and pipeline the hot loop.
        f"-DDEPAD_D0={d0}",
    ]
    # f32 -> bf16 selects the de-pad + convert kernel body; f32 -> f32 keeps the plain copy.
    if output_tensor.dtype == bfloat16:
        compile_flags.append("-DDEPAD_CONVERT_F32_TO_BF16=1")

    return ExternalFunction(
        name="ggml_hsa_depad",
        object_file_name="ggml_hsa_depad_core_function.o",
        source_file=str(current_dir / "depad.cc"),
        arg_types=[
            np.ndarray[(d0pad,), np.dtype[src.dtype]],
            np.ndarray[(d0,), np.dtype[output_tensor.dtype]],
            np.int32,  # d0
            np.int32,  # d0pad
        ],
        compile_flags=compile_flags,
    )
