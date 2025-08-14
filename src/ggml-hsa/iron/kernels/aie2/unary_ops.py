# unary_ops.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from os import path
import numpy as np

from aie.iron import (
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
    ExternalFunction,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

from utils import arch_to_device


def unary_op(device, input_tensor, output_tensor, func):
    """Implements output = op(input)."""

    if not input_tensor.contiguous or not output_tensor.contiguous:
        raise ValueError("Input and output tensors must be contiguous in memory.")

    if input_tensor.shape != output_tensor.shape:
        raise ValueError(
            f"Incompatible input and output shapes ({input_tensor.shape} != {output_tensor.shape})."
        )

    if func.tile_size(0) != func.tile_size(1):
        raise ValueError(
            f"Input and output tile sizes do not match ({func.tile_size(0)} != {func.tile_size(1)})."
        )

    tile_size = func.tile_size(0)
    num_elements = np.size(input_tensor)
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    if input_tensor.dtype != output_tensor.dtype:
        raise ValueError(
            f"Incompatible input and output data types ({input_tensor.dtype} != {output_tensor.dtype})."
        )

    # Input / output data movement
    input_tile_ty = np.ndarray[(func.tile_size(0),), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(func.tile_size(1),), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Task for the core to perform
    def core_fn(of_in, of_out, func):
        tile_size = func.tile_size(0)
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            func(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to perform the task
    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod(), func])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(device, rt).resolve_program(SequentialPlacer())


def create_external_function(
    op_name: str, input_tensors: list, output_tensor
) -> ExternalFunction:
    """Returns an ExternalFunction specification for unary ops."""

    assert len(input_tensors) == 1
    assert input_tensors[0].dtype == output_tensor.dtype
    assert input_tensors[0].shape == output_tensor.shape

    tile_size = 16
    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name="ggml_op_" + op_name,
        object_file_name=f"{op_name}_core_function.o",
        source_file=path.join(current_dir, "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-DCOMPILE_{op_name.upper()}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return func


def ggml_op_sqr(device: str, input_tensors: list, output_tensor):
    """GGML_OP_SQR implementation."""
    func = create_external_function(
        op_name="sqr", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_op_sqrt(device: str, input_tensors: list, output_tensor):
    """GGML_OP_SQRT implementation."""
    func = create_external_function(
        op_name="sqrt", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_op_log(device: str, input_tensors: list, output_tensor):
    """GGML_OP_LOG implementation."""
    raise NotImplementedError


def ggml_op_sin(device: str, input_tensors: list, output_tensor):
    """GGML_OP_SIN implementation."""
    raise NotImplementedError


def ggml_op_cos(device: str, input_tensors: list, output_tensor):
    """GGML_OP_COS implementation."""
    raise NotImplementedError


def ggml_unary_op_abs(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_ABS implementation."""
    func = create_external_function(
        op_name="abs", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_sgn(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SGN implementation."""
    func = create_external_function(
        op_name="sgn", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_neg(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_NEG implementation."""
    func = create_external_function(
        op_name="neg", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_step(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_STEP implementation."""
    func = create_external_function(
        op_name="step", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_tanh(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_TANH implementation."""
    raise NotImplementedError


def ggml_unary_op_elu(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_ELU implementation."""
    raise NotImplementedError


def ggml_unary_op_relu(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_RELU implementation."""
    func = create_external_function(
        op_name="relu", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_sigmoid(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SIGMOID implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_GELU implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu_quick(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_GELU_QUICK implementation."""
    raise NotImplementedError


def ggml_unary_op_silu(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SILU implementation."""
    raise NotImplementedError


def ggml_unary_op_hardswish(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_HARDSWISH implementation."""
    func = create_external_function(
        op_name="hardswish", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_hardsigmoid(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_HARDSIGMOID implementation."""
    func = create_external_function(
        op_name="hardsigmoid", input_tensors=input_tensors, output_tensor=output_tensor
    )
    return unary_op(arch_to_device(device), *input_tensors, output_tensor, func)


def ggml_unary_op_exp(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_EXP implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu_erf(device: str, input_tensors: list, output_tensor):
    """GGML_UNARY_OP_GELU_ERF implementation."""
    raise NotImplementedError
