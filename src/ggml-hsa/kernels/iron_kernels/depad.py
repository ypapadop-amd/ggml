#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for the depad operation."""

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
    """Build the de-pad IRON program: MUL_MAT post-amble, narrowing each row from d0pad to d0.

    f32 -> bf16 fuses the per-layer cast that would otherwise follow the MUL_MAT
    as a separate CPY (bit-identical to the separate cast).

    Parameters:
        arch: Target architecture.
        input_tensors: [src] padded source tensor.
        output_tensor: Dense destination tensor.
        op_params: Unused (kept for the dispatch ABI).

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

    # Source is the f32 padded GEMM temporary (destination f32 or bf16, the latter fusing the
    # per-layer cast) or an already-bf16 padded temporary (destination must also be bf16, plain
    # de-pad with no conversion).
    if src.dtype == np.float32:
        if output_tensor.dtype not in (np.float32, bfloat16):
            msg = (
                f"depad with an f32 source requires an f32 or bf16 destination; got "
                f"{output_tensor.dtype}."
            )
            raise ValueError(msg)
    elif src.dtype == bfloat16:
        if output_tensor.dtype != bfloat16:
            msg = (
                f"depad with a bf16 source requires a bf16 destination; got "
                f"{output_tensor.dtype}."
            )
            raise ValueError(msg)
    else:
        msg = f"depad requires an f32 or bf16 source; got {src.dtype}."
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
        src: Source tensor (padded temporary).
        output_tensor: Destination tensor.
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
    # The kernel selects its mode (plain copy vs. f32 -> bf16 convert) at compile time via
    # `if constexpr` on INPUT_DTYPE/OUTPUT_DTYPE; no extra flag is needed.

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
