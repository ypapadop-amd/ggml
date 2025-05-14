# unary_ops.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col4
from aie.iron.controlflow import range_


def unary_op(input, op, output):
    """Implements output = op(input)."""
    if input.shape != output.shape:
        raise ValueError(
            f"Input and output shapes are not the equal ({input.shape} != {output.shape})."
        )
    num_elements = np.size(input)
    n = 16
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    num_elements_div_n = num_elements // n

    if input.dtype != output.dtype:
        raise ValueError(
            f"Input and output data types are not the same ({input.dtype} != {output.dtype})."
        )
    dtype = input.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_elements_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = op(elem_in[i])
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def ggml_op_sqr(input_tensors: list, output_tensor):
    """GGML_OP_SQR implementation."""
    return unary_op(*input_tensors, lambda x: x * x, output_tensor)


def ggml_op_sqrt(input_tensors: list, output_tensor):
    """GGML_OP_SQRT implementation."""
    raise NotImplementedError


def ggml_op_log(input_tensors: list, output_tensor):
    """GGML_OP_LOG implementation."""
    raise NotImplementedError


def ggml_op_sin(input_tensors: list, output_tensor):
    """GGML_OP_SIN implementation."""
    raise NotImplementedError


def ggml_op_cos(input_tensors: list, output_tensor):
    """GGML_OP_COS implementation."""
    raise NotImplementedError


def ggml_unary_op_abs(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_ABS implementation."""
    raise NotImplementedError


def ggml_unary_op_sgn(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SGN implementation."""
    raise NotImplementedError


def ggml_unary_op_neg(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_NEG implementation."""
    return unary_op(*input_tensors, lambda x: -x, output_tensor)


def ggml_unary_op_step(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_STEP implementation."""
    raise NotImplementedError


def ggml_unary_op_tanh(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_TANH implementation."""
    raise NotImplementedError


def ggml_unary_op_elu(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_ELU implementation."""
    raise NotImplementedError


def ggml_unary_op_relu(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_RELU implementation."""
    raise NotImplementedError


def ggml_unary_op_sigmoid(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SIGMOID implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_GELU implementation."""
    raise NotImplementedError


def ggml_unary_op_gelu_quick(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_GELU implementation."""
    raise NotImplementedError


def ggml_unary_op_silu(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_SILU implementation."""
    raise NotImplementedError


def ggml_unary_op_hardswish(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_HARDSWISH implementation."""
    raise NotImplementedError


def ggml_unary_op_hardsigmoid(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_HARDSIGMOID implementation."""
    raise NotImplementedError


def ggml_unary_op_exp(input_tensors: list, output_tensor):
    """GGML_UNARY_OP_EXP implementation."""
    raise NotImplementedError


@iron.jit(is_placed=False)
def ggml_op_sqr_jit(input_tensor, output_tensor):
    return ggml_op_sqr([input_tensor], output_tensor)


@pytest.mark.parametrize("num_elements", [16, 256, 4096])
@pytest.mark.parametrize("dtype", [np.int32])
@pytest.mark.parametrize(
    "function, op",
    [
        (ggml_op_sqr_jit, lambda x: x * x),
    ],
)
def test_ggml_op_unary(function, op, dtype, num_elements):
    iron.set_current_device(NPU1Col4())

    # Construct two input random tensors and an output zeroed tensor
    input_tensor = iron.randint(-100, 100, (num_elements,), dtype=dtype, device="npu")
    output_tensor = iron.zeros_like(input_tensor)

    # JIT-compile the kernel then launch the kernel with the given arguments
    function(input_tensor, output_tensor)

    assert np.array_equal(op(input_tensor.numpy()), output_tensor.numpy())
