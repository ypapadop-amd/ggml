#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

import operator
from typing import Callable
import numpy as np
import pytest

from utils import suppress_import_pyxrt_msg

suppress_import_pyxrt_msg()

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

from build import arch_aligned_num_elements, arch_to_device, max_tile_size


def apply_binary_op(arch: str, input_tensors: list, function: Callable, output_tensor):
    """
    Implements output_tensor = op(*input_tensors)

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Input tensors.
        function (Callable): Binary operator.
        output_tensor: Output tensor.
    """

    num_elements = arch_aligned_num_elements(arch=arch, tensor=output_tensor)
    tile_size = max_tile_size(arch, output_tensor.dtype, num_elements)
    num_tiles = num_elements // tile_size

    # Define a task that will run on a compute tile
    def core_body(of_in0, of_in1, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(tile_size):
                elem_out[i] = function(elem_in0[i], elem_in1[i])
            of_in0.release(1)
            of_in1.release(1)
            of_out.release(1)

    # AIE-array data movement with object fifos
    input_tile_tys = [
        (np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]])
        for input_tensor in input_tensors
    ]
    of_ins = [
        ObjectFifo(input_tile_ty, name=f"in{index}")
        for index, input_tile_ty in enumerate(input_tile_tys)
    ]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_out = ObjectFifo(output_tile_ty, name="out")

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[x.cons() for x in of_ins] + [of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    input_tensor_tys = [
        np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
        for input_tensor in input_tensors
    ]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime()
    with rt.sequence(*input_tensor_tys, output_tensor_ty) as t:
        rt.start(worker)
        [rt.fill(of_in.prod(), t[i]) for i, of_in in enumerate(of_ins)]
        rt.drain(of_out.cons(), t[-1], wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def ggml_op_binary(arch: str, input_tensors: list, function: Callable, output_tensor):
    """
    Binary operation implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        function (Callable): Binary operator.
        output_tensor: Output tensor.
    """

    if len(input_tensors) != 2:
        raise ValueError("Operation requires exactly two input tensors.")

    if (
        any(t.contiguous is False for t in input_tensors)
        or output_tensor.contiguous is False
    ):
        raise ValueError("Input and output tensors must be contiguous in memory.")

    for input_tensor in input_tensors:
        if input_tensor.shape != output_tensor.shape:
            raise ValueError(
                f"Input and output tensors must have the same shape: {input_tensor.shape} != {output_tensor.shape}"
            )

    if output_tensor.shape[1:4] != (1, 1, 1):
        raise ValueError(f"Unsupported shape ({output_tensor.shape}).")

    return apply_binary_op(
        arch=arch,
        input_tensors=input_tensors,
        function=function,
        output_tensor=output_tensor,
    )


def ggml_op_add(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_ADD implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    return ggml_op_binary(
        arch=arch,
        input_tensors=input_tensors,
        function=lambda x, y: x + y,
        output_tensor=output_tensor,
    )


def ggml_op_sub(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SUB implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    return ggml_op_binary(
        arch=arch,
        input_tensors=input_tensors,
        function=lambda x, y: x - y,
        output_tensor=output_tensor,
    )


def ggml_op_mul(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_MUL implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    return ggml_op_binary(
        arch=arch,
        input_tensors=input_tensors,
        function=lambda x, y: x * y,
        output_tensor=output_tensor,
    )


def ggml_op_div(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_DIV implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
        op_params: Operation parameters.
    """
    return ggml_op_binary(
        arch=arch,
        input_tensors=input_tensors,
        function=lambda x, y: x / y,
        output_tensor=output_tensor,
    )


@pytest.mark.parametrize("num_elements", [16, 256, 4096])
@pytest.mark.parametrize("dtype", [np.int32])
@pytest.mark.parametrize(
    "function, op",
    [
        (ggml_op_add, operator.add),
        (ggml_op_sub, operator.sub),
        (ggml_op_mul, operator.mul),
        (ggml_op_div, operator.floordiv),
    ],
)
def test_ggml_op_binary(function, op, dtype, num_elements):
    # Construct two input random tensors and an output zeroed tensor
    input_tensor0 = iron.randint(
        1, 100, (num_elements, 1, 1, 1), dtype=dtype, device="npu"
    )
    input_tensor1 = iron.randint(
        1, 100, (num_elements, 1, 1, 1), dtype=dtype, device="npu"
    )
    output_tensor = iron.zeros_like(input_tensor0)
    input_tensor0.contiguous = True
    input_tensor1.contiguous = True
    output_tensor.contiguous = True

    arch = None
    device = iron.get_current_device()
    if device == aie.iron.Device.NPU1:
        arch = "aie2"
    elif device == aie.iron.Device.NPU2:
        arch = "aie2p"
    else:
        raise ValueError(f"Unsupported device: {device}")

    # JIT-compile the kernel then launch the kernel with the given arguments
    @iron.jit(is_placed=False)
    def jit_wrapper(input_tensor0, input_tensor1, output_tensor):
        return function(arch, [input_tensor0, input_tensor1], output_tensor)

    jit_wrapper(input_tensor0, input_tensor1, output_tensor)

    assert np.array_equal(
        op(input_tensor0.numpy(), input_tensor1.numpy()), output_tensor.numpy()
    )
