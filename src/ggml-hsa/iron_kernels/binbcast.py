# binbcast.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import operator
import numpy as np
import pytest

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4
from aie.iron.controlflow import range_


def vector_vector_op(input0, input1, op, output):
    """Implements output = input0 op input1"""
    if input0.shape != input1.shape:
        raise ValueError(
            f"Input shapes are not the equal ({input0.shape} != {input1.shape})."
        )
    if input0.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({input0.shape} != {output.shape})."
        )
    if len(np.shape(input0)) != 1:
        raise ValueError("Function only supports vectors.")
    num_elements = np.size(input0)
    n = 16
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    num_elements_div_n = num_elements // n

    if input0.dtype != input1.dtype:
        raise ValueError(
            f"Input data types are not the same ({input0.dtype} != {input1.dtype})."
        )
    if input0.dtype != output.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({input0.dtype} != {output.dtype})."
        )
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_elements_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = op(elem_in1[i], elem_in2[i])
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def ggml_op_add(input_tensors: list, output_tensor):
    """GGML_OP_ADD implementation."""
    return vector_vector_op(*input_tensors, lambda x, y: x + y, output_tensor)


def ggml_op_sub(input_tensors: list, output_tensor):
    """GGML_OP_SUB implementation."""
    return vector_vector_op(*input_tensors, lambda x, y: x - y, output_tensor)


def ggml_op_mul(input_tensors: list, output_tensor):
    """GGML_OP_MUL implementation."""
    return vector_vector_op(*input_tensors, lambda x, y: x * y, output_tensor)


def ggml_op_div(input_tensors: list, output_tensor):
    """GGML_OP_DIV implementation."""
    return vector_vector_op(*input_tensors, lambda x, y: x / y, output_tensor)


@iron.jit(is_placed=False)
def ggml_op_add_jit(input0, input1, output):
    return ggml_op_add([input0, input1], output)


@iron.jit(is_placed=False)
def ggml_op_sub_jit(input0, input1, output):
    return ggml_op_sub([input0, input1], output)


@iron.jit(is_placed=False)
def ggml_op_mul_jit(input0, input1, output):
    return ggml_op_mul([input0, input1], output)


@iron.jit(is_placed=False)
def ggml_op_div_jit(input0, input1, output):
    return ggml_op_div([input0, input1], output)


@pytest.mark.parametrize("num_elements", [16, 256, 4096])
@pytest.mark.parametrize("dtype", [np.int32])
@pytest.mark.parametrize(
    "function, op",
    [
        (ggml_op_add_jit, operator.add),
        (ggml_op_sub_jit, operator.sub),
        (ggml_op_mul_jit, operator.mul),
        (ggml_op_div_jit, operator.floordiv),
    ],
)
def test_ggml_op_add(function, op, dtype, num_elements):
    iron.set_current_device(NPU1Col4())

    # Construct two input random tensors and an output zeroed tensor
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launch the kernel with the given arguments
    function(input0, input1, output)

    assert np.array_equal(op(input0.numpy(), input1.numpy()), output.numpy())
