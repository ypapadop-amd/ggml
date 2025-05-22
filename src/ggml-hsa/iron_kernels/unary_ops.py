# unary_ops.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from os import path
import numpy as np
from functools import partial

import aie.iron as iron
from aie.iron import ObjectFifo, Kernel, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

from compiler import core_function, CoreFunctionInfo, dtype_to_str


def unary_op(input_tensor, output_tensor, core_function_info: CoreFunctionInfo):
    """Implements output = op(input)."""

    tile_size = 16

    if input_tensor.shape != output_tensor.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({input_tensor.shape} != {output_tensor.shape})."
        )
    num_elements = np.size(input_tensor)
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({input_tensor.dtype} != {output_tensor.dtype})."
        )

    # Define tensor types
    dtype = input_tensor.dtype
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    # External, binary kernel definition
    kernel_fn = Kernel(
        name=core_function_info.exported_function,
        bin_name=core_function_info.object_file,
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    # Input data movement
    of_in = ObjectFifo(tile_ty, name="in")

    # Output data movement
    of_out = ObjectFifo(tile_ty, name="out")

    # Task for the core to perform
    def core_fn(of_in, of_out, func):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod(), kernel_fn])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def unary_op_core_function_info(
    op_name: str, device, input_tensors: list, output_tensor
):
    """Returns a compilation specification for unary ops."""

    assert len(input_tensors) == 1
    assert input_tensors[0].dtype == output_tensor.dtype
    assert input_tensors[0].shape == output_tensor.shape

    current_dir = path.dirname(path.realpath(__file__))
    return CoreFunctionInfo(
        source_file=path.join(current_dir, "unary_ops.cc"),
        exported_function="ggml_op_" + op_name,
        compile_args=[
            f"-DCOMPILE_{op_name.upper()}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )


@core_function(partial(unary_op_core_function_info, op_name="sqr"))
def ggml_op_sqr(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_OP_SQR implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


@core_function(partial(unary_op_core_function_info, op_name="sqrt"))
def ggml_op_sqrt(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_OP_SQRT implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


def ggml_op_log(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_OP_LOG implementation."""
    raise NotImplementedError


def ggml_op_sin(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_OP_SIN implementation."""
    raise NotImplementedError


def ggml_op_cos(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_OP_COS implementation."""
    raise NotImplementedError


@core_function(partial(unary_op_core_function_info, op_name="abs"))
def ggml_unary_op_abs(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_ABS implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


@core_function(partial(unary_op_core_function_info, op_name="sgn"))
def ggml_unary_op_sgn(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_SGN implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


@core_function(partial(unary_op_core_function_info, op_name="neg"))
def ggml_unary_op_neg(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_NEG implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


@core_function(partial(unary_op_core_function_info, op_name="step"))
def ggml_unary_op_step(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_STEP implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


def ggml_unary_op_tanh(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_TANH implementation."""
    raise NotImplementedError


def ggml_unary_op_elu(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_ELU implementation."""
    raise NotImplementedError


@core_function(partial(unary_op_core_function_info, op_name="relu"))
def ggml_unary_op_relu(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_RELU implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


def ggml_unary_op_sigmoid(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_SIGMOID implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_GELU implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu_quick(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_GELU implementation."""
    raise NotImplementedError


def ggml_unary_op_silu(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_SILU implementation."""
    raise NotImplementedError


@core_function(partial(unary_op_core_function_info, op_name="hardswish"))
def ggml_unary_op_hardswish(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_HARDSWISH implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


@core_function(partial(unary_op_core_function_info, op_name="hardsigmoid"))
def ggml_unary_op_hardsigmoid(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_HARDSIGMOID implementation."""
    return unary_op(*input_tensors, output_tensor, core_function_info)


def ggml_unary_op_exp(
    input_tensors: list, output_tensor, core_function_info: CoreFunctionInfo
):
    """GGML_UNARY_OP_EXP implementation."""
    raise NotImplementedError
