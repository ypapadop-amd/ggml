# softmax/softmax.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

from os import path
import numpy as np

from aie.iron import dtype_to_str, ExternalFunction

from build import arch_aligned_num_elements, arch_to_device, max_tile_size


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


def ggml_op_softmax(arch: str, input_tensors: list, output_tensor):
    """
    GGML_OP_SOFT_MAX implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of two input tensors.
        output_tensor: Output tensor.
    """

    if len(input_tensors) != 1:
        raise ValueError("Operation requires exactly two input tensors.")

    if (
        any(not input_tensor.contiguous for input_tensor in input_tensors)
        or output_tensor.contiguous is False
    ):
        raise ValueError("Input and output tensors must be contiguous in memory.")

    if input_tensors[0].shape != input_tensors[1].shape:
        raise ValueError("Input tensors must have the same shape.")

    raise NotImplementedError("GGML_OP_SOFT_MAX is not implemented yet.")
