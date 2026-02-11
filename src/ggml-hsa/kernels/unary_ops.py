#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

from dataclasses import dataclass
from os import path
import numpy as np

from utils import suppress_import_pyxrt_msg

suppress_import_pyxrt_msg()

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

from build import arch_aligned_num_elements, arch_to_device, max_tile_size


@dataclass(frozen=True)
class CoreFunctionSpec:
    """Specification for a core function to be used in unary operations.

    Attributes:
        external_function (ExternalFunction): The external function to be called for the unary operation.
        num_elements (int): The total number of elements in the input/output tensors.
    """

    external_function: ExternalFunction
    num_elements: int

    @property
    def tile_size(self) -> int:
        """Returns the tile size used by the external function."""
        return self.external_function.tile_size(0)


def apply_unary_op(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """
    Implements output_tensor = op(input_tensors[0])

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Input tensors.
        function_spec (CoreFunctionSpec): Unary operator specification.
        output_tensor: Output tensor.
    """

    input_tensor = input_tensors[0]

    # Tile size and number of tiles
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size
    num_tiles = num_elements // tile_size
    if num_elements % tile_size != 0:
        raise ValueError(
            f"num_elements ({num_elements}) must be divisible by tile_size ({tile_size}) "
            "for correct tiling"
        )

    # AIE-array data movement with object fifos
    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Create a worker to run the task on a compute tile
    worker = None
    function = function_spec.external_function

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

    # Runtime operations to move data to/from the AIE-array
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime()
    with rt.sequence(input_tensor_ty, output_tensor_ty) as t:
        rt.start(worker)
        rt.fill(of_in.prod(), t[0])
        rt.drain(of_out.cons(), t[-1], wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> CoreFunctionSpec:
    """
    Creates a specification for unary ops.

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    Returns:
        CoreFunctionSpec: Specification for the core function to be used in unary ops.
    """

    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = max_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=path.join(current_dir, "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{op_name}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


def ggml_op_unary(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
):
    """
    Unary operation implementation.

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the unary operation.
        input_tensors (list): List of one input tensor.
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

    function_spec = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )

    return apply_unary_op(
        arch=arch,
        input_tensors=input_tensors,
        function_spec=function_spec,
        output_tensor=output_tensor,
    )


def ggml_op_sqr(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SQR implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_OP_SQR",
    )


def ggml_op_sqrt(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SQRT implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_log(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_LOG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_sin(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SIN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_op_cos(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_COS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_abs(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ABS implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_ABS",
    )


def ggml_unary_op_sgn(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SGN implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_SGN",
    )


def ggml_unary_op_neg(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_NEG implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_NEG",
    )


def ggml_unary_op_step(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_STEP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_STEP",
    )


def ggml_unary_op_tanh(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_TANH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_elu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_relu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_RELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_RELU",
    )


def ggml_unary_op_sigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_quick(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU_QUICK implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_silu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_SILU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_hardswish(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_HARDSWISH implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_HARDSWISH",
    )


def ggml_unary_op_hardsigmoid(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_HARDSIGMOID implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_HARDSIGMOID",
    )


def ggml_unary_op_exp(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_EXP implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_gelu_erf(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_GELU_ERF implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_xielu(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_XIELU implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    raise NotImplementedError


def ggml_unary_op_floor(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_FLOOR implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_FLOOR",
    )


def ggml_unary_op_ceil(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_CEIL implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_CEIL",
    )


def ggml_unary_op_round(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_ROUND implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_ROUND",
    )


def ggml_unary_op_trunc(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_UNARY_OP_TRUNC implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of one input tensor.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """

    return ggml_op_unary(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_name="GGML_UNARY_OP_TRUNC",
    )
