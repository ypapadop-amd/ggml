#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

from os import path
from typing import Callable
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

from utils import arch_aligned_num_elements, arch_to_device, max_tile_size


def ggml_op_unary(
    arch: str,
    input_tensors: list,
    function: Callable | ExternalFunction,
    output_tensor,
):
    """
    Implements output_tensor = function(input_tensors[0])

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        function (Callable): Unary operator.
        output_tensor: Output tensor.
    """

    if len(input_tensors) != 1:
        raise ValueError("Operation requires exactly one input tensor.")

    if input_tensors[0].contiguous is False or output_tensor.contiguous is False:
        raise ValueError("Input and output tensors must be contiguous in memory.")

    if input_tensors[0].shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")

    if output_tensor.shape[1:4] != (1, 1, 1):
        raise ValueError(f"Unsupported shape ({output_tensor.shape}).")

    input_tensor = input_tensors[0]

    # Find tile size and number of tiles
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = None
    num_tiles = None
    if isinstance(function, ExternalFunction):
        tile_size = function.tile_size(0)
    else:
        tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)
    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    # AIE-array data movement with object fifos
    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Create a worker to run the task on a compute tile
    worker = None
    if isinstance(function, ExternalFunction):
        # Task for the core to perform with an external function
        def ext_core_fn(of_in, of_out, function):
            # Number of sub-vector "tile" iterations
            for _ in range_(num_tiles):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                function(elem_in, elem_out, tile_size)
                of_in.release(1)
                of_out.release(1)

        worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])
    else:

        def core_fn(of_in, of_out):
            # Number of sub-vector "tile" iterations
            for _ in range_(num_tiles):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                for i in range_(tile_size):
                    elem_out[i] = function(elem_in[i])
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> ExternalFunction:
    """
    Creates an ExternalFunction specification for unary ops.

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.
    """

    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name="ggml_op_" + op_name,
        object_file_name=f"{op_name}_core_function.o",
        source_file=path.join(current_dir, "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-DCOMPILE_{op_name.upper()}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return func


def ggml_op_sqr(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_SQR implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="sqr",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_op_sqrt(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_SQRT implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_op_log(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_LOG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_op_sin(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_SIN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_op_cos(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_COS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_abs(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_ABS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="abs",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_sgn(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_SGN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="sgn",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_neg(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_NEG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="neg",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_step(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_STEP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="step",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_tanh(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_TANH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_elu(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_ELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_relu(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_RELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="relu",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_sigmoid(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_SIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_gelu(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_GELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_quick(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_GELU_QUICK implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_silu(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_SILU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_hardswish(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_HARDSWISH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="hardswish",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_hardsigmoid(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_HARDSIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """

    core_function = create_external_function(
        arch=arch,
        op_name="hardsigmoid",
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )
    return ggml_op_unary(
        arch=arch,
        function=core_function,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )


def ggml_unary_op_exp(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_EXP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_erf(arch: str, input_tensors: list, output_tensor):
    """
    GGML_UNARY_OP_GELU_ERF implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
    """
    raise NotImplementedError
