#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_POOL_2D (max/avg pooling)."""

import struct
from pathlib import Path

import numpy as np
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

from .utils import fill_drain_program

# GGML pooling op selector (matches enum ggml_op_pool in include/ggml.h).
_GGML_OP_POOL_MAX = 0
_GGML_OP_POOL_AVG = 1


def pool_2d(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the pooling IRON program.

    Args:
        arch: Target architecture.
        input_tensors: List of one input tensor, shape [IW, IH, C, N].
        output_tensor: Output tensor, shape [OW, OH, C, N].
        op_params: {op, k0, k1, s0, s1, p0, p1} as 7 x int32.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, op_params, or
            pooling op.
    """
    if len(input_tensors) != 1:
        msg = "Operation requires exactly one input tensor."
        raise ValueError(msg)

    input_tensor = input_tensors[0]

    if (
        input_tensor.dtype not in (np.float32, bfloat16)
        or output_tensor.dtype != np.float32
    ):
        msg = (
            f"POOL_2D only supports float32/bfloat16 input and float32 output; "
            f"got input dtype={input_tensor.dtype}, output dtype={output_tensor.dtype}."
        )
        raise ValueError(msg)

    if not input_tensor.contiguous or not output_tensor.contiguous:
        msg = "Input and output tensors must be contiguous in memory."
        raise ValueError(msg)

    # op_params: {op, k0, k1, s0, s1, p0, p1} as 7 x int32.
    _POOL_2D_PARAMS_SIZE = 7 * 4  # 7 int32 fields
    if len(op_params) < _POOL_2D_PARAMS_SIZE:
        msg = (
            f"op_params too short: expected at least {_POOL_2D_PARAMS_SIZE} bytes, "
            f"got {len(op_params)}."
        )
        raise ValueError(msg)
    op, k0, k1, s0, s1, p0, p1 = struct.unpack_from("7i", op_params, 0)

    if op not in (_GGML_OP_POOL_MAX, _GGML_OP_POOL_AVG):
        msg = f"Unsupported pooling op: {op}."
        raise ValueError(msg)

    if k0 <= 0 or k1 <= 0:
        msg = f"Kernel dimensions must be positive; got k0={k0}, k1={k1}."
        raise ValueError(msg)

    if s0 <= 0 or s1 <= 0:
        msg = f"Strides must be positive; got s0={s0}, s1={s1}."
        raise ValueError(msg)

    if p0 < 0 or p1 < 0:
        msg = f"Padding must be non-negative; got p0={p0}, p1={p1}."
        raise ValueError(msg)

    iw, ih, in_c, in_n = input_tensor.shape
    ow, oh, out_c, out_n = output_tensor.shape

    if (in_c, in_n) != (out_c, out_n):
        msg = (
            f"Channel/batch mismatch: input {(in_c, in_n)} vs output {(out_c, out_n)}."
        )
        raise ValueError(msg)

    in_plane = iw * ih
    out_plane = ow * oh
    num_planes = in_c * in_n

    function = _create_external_function(
        op_name="GGML_OP_POOL_2D",
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        in_plane=in_plane,
        out_plane=out_plane,
    )

    # AIE-array data movement with object fifos: one channel-plane per tile.
    input_tile_ty = np.ndarray[(in_plane,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(out_plane,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_out, function):
        for _ in range_(num_planes):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, iw, ih, ow, oh, k0, k1, s0, s1, p0, p1, op)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    # Runtime operations to move data to/from the AIE-array.
    input_tensor_ty = np.ndarray[(in_plane * num_planes,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[
        (out_plane * num_planes,), np.dtype[output_tensor.dtype]
    ]

    return fill_drain_program(
        arch,
        [worker],
        [input_tensor_ty],
        output_tensor_ty,
        [of_in.prod()],
        of_out.cons(),
    )


def _create_external_function(
    op_name: str,
    input_tensor,
    output_tensor,
    in_plane: int,
    out_plane: int,
) -> ExternalFunction:
    """Create the ExternalFunction for the pooling core function.

    Args:
        op_name: Operation name (drives function name and compile flags).
        input_tensor: Input tensor.
        output_tensor: Output tensor.
        in_plane: Input channel-plane size (IW * IH).
        out_plane: Output channel-plane size (OW * OH).

    Returns:
        The configured ExternalFunction.
    """
    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "pool_2d.cc"),
        arg_types=[
            np.ndarray[(in_plane,), np.dtype[input_tensor.dtype]],
            np.ndarray[(out_plane,), np.dtype[output_tensor.dtype]],
            np.int32,  # iw
            np.int32,  # ih
            np.int32,  # ow
            np.int32,  # oh
            np.int32,  # k0
            np.int32,  # k1
            np.int32,  # s0
            np.int32,  # s1
            np.int32,  # p0
            np.int32,  # p1
            np.int32,  # op
        ],
        compile_flags=[
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
