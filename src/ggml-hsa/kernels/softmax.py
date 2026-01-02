#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from os import path
from typing import Tuple, Optional, Any

import numpy as np
import struct

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


def ggml_op_softmax(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """
    GGML_OP_SOFTMAX implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of 1-3 input tensors:
            - input_tensors[0]: Input tensor (required)
            - input_tensors[1]: Mask tensor (optional)
            - input_tensors[2]: Positional encoding tensor (optional)
        output_tensor: Output tensor.
        op_params (bytearray): Operation parameters (scale, max_bias).
    """

    input_tensor_count = len(input_tensors)
    
    if input_tensor_count < 1 or input_tensor_count > 3:
        raise ValueError(f"Operation requires 1, 2, or 3 tensors: {input_tensor_count}")

    input_tensor = input_tensors[0]
    mask_tensor = input_tensors[1] if input_tensor_count >= 2 else None
    pos_tensor = input_tensors[2] if input_tensor_count >= 3 else None
    
    # Validate contiguity
    if not input_tensor.contiguous:
        raise ValueError("Input tensor must be contiguous in memory.")
    if not output_tensor.contiguous:
        raise ValueError("Output tensor must be contiguous in memory.")
    if mask_tensor is not None and not mask_tensor.contiguous:
        raise ValueError("Mask tensor must be contiguous in memory.")
    if pos_tensor is not None and not pos_tensor.contiguous:
        raise ValueError("Positional encoding tensor must be contiguous in memory.")

    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")

    # Unpack op_params: scale and max_bias
    scale = struct.unpack_from("f", op_params, 0)[0]
    max_bias = struct.unpack_from("f", op_params, 4)[0]
    
    op_name = "softmax"
    
    if input_tensor_count == 1:
        return create_unary_program(arch, op_name, input_tensor, output_tensor, scale, max_bias)
    elif input_tensor_count == 2:
        return create_binary_program(arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias)
    else:  # input_tensor_count == 3
        return create_ternary_program(arch, op_name, input_tensor, mask_tensor, pos_tensor, output_tensor, scale, max_bias)


def create_unary_program(arch, op_name, input_tensor, output_tensor, scale, max_bias):
    """Softmax without mask or positional encoding."""
    function, num_elements, tile_size = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=None,
        pos_tensor=None,
        output_tensor=output_tensor,
    )

    num_tiles = num_elements // tile_size
    assert num_elements % tile_size == 0

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_out, function):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size, scale, max_bias)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    rt = Runtime()
    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    with rt.sequence(input_tensor_ty, output_tensor_ty) as (a_in, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_binary_program(arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias):
    """Softmax with mask tensor."""
    func_result = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        pos_tensor=None,
        output_tensor=output_tensor,
    )
    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    num_elements_mask = func_result[3]
    tile_size_mask = func_result[4]

    num_tiles_in = num_elements_in // tile_size_in
    num_tiles_mask = num_elements_mask // tile_size_mask
    
    assert num_elements_in % tile_size_in == 0
    assert num_elements_mask % tile_size_mask == 0
    assert num_elements_in == num_elements_mask
    assert num_tiles_in == num_tiles_mask

    input_tile_ty = np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]
    mask_tile_ty = np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]]
    
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_mask = ObjectFifo(mask_tile_ty, name="mask")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_mask, of_out, function):
        for _ in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_mask, elem_out, tile_size_in, scale, max_bias)
            of_in.release(1)
            of_mask.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_mask.cons(), of_out.prod(), function])

    rt = Runtime()

    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    with rt.sequence(input_tensor_ty, mask_tensor_ty, output_tensor_ty) as (a_in, a_mask, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_mask.prod(), a_mask)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_ternary_program(arch, op_name, input_tensor, mask_tensor, pos_tensor, output_tensor, scale, max_bias):
    """Softmax with mask tensor and positional encoding tensor."""
    func_result = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        pos_tensor=pos_tensor,
        output_tensor=output_tensor,
    )
    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    num_elements_mask = func_result[3]
    tile_size_mask = func_result[4]
    num_elements_pos = func_result[5]
    tile_size_pos = func_result[6]

    num_tiles_in = num_elements_in // tile_size_in
    num_tiles_mask = num_elements_mask // tile_size_mask
    num_tiles_pos = num_elements_pos // tile_size_pos
    
    assert num_elements_in % tile_size_in == 0
    assert num_elements_mask % tile_size_mask == 0
    assert num_elements_pos % tile_size_pos == 0
    assert num_elements_in == num_elements_mask
    assert num_tiles_in == num_tiles_mask

    input_tile_ty = np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]
    mask_tile_ty = np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]]
    pos_tile_ty = np.ndarray[(tile_size_pos,), np.dtype[pos_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]]
    
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_mask = ObjectFifo(mask_tile_ty, name="mask")
    of_pos = ObjectFifo(pos_tile_ty, name="pos")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_mask, of_pos, of_out, function):
        for _ in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_pos = of_pos.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_mask, elem_pos, elem_out, tile_size_in, scale, max_bias)
            of_in.release(1)
            of_mask.release(1)
            of_pos.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_mask.cons(), of_pos.cons(), of_out.prod(), function])

    rt = Runtime()
    
    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    pos_tensor_ty = np.ndarray[(num_elements_pos,), np.dtype[pos_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    with rt.sequence(input_tensor_ty, mask_tensor_ty, pos_tensor_ty, output_tensor_ty) as (a_in, a_mask, a_pos, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_mask.prod(), a_mask)
        rt.fill(of_pos.prod(), a_pos)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_external_function(
    arch: str,
    op_name: str,
    input_tensor: Any,
    mask_tensor: Optional[Any],
    pos_tensor: Optional[Any],
    output_tensor: Any,
) -> Tuple:
    """
    Creates an ExternalFunction specification for softmax variants.

    Returns:
        If no mask or pos tensor:
            (func, num_elements_in, tile_size_in)
        If mask tensor only:
            (func, num_elements_in, tile_size_in, num_elements_mask, tile_size_mask)
        If mask and pos tensor:
            (func, num_elements_in, tile_size_in, num_elements_mask, tile_size_mask, num_elements_pos, tile_size_pos)
    """

    num_elements_in = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size_in = max_tile_size(arch, input_tensor.dtype, num_elements_in)

    arg_types = [np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]]
    compile_flags = [f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}"]

    result_extra = []

    if mask_tensor is not None:
        num_elements_mask = arch_aligned_num_elements(arch=arch, tensor=mask_tensor)
        tile_size_mask = max_tile_size(arch, mask_tensor.dtype, num_elements_mask)

        arg_types.append(np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]])
        compile_flags.append(f"-DMASK_DTYPE={dtype_to_str(mask_tensor.dtype)}")
        result_extra.extend([num_elements_mask, tile_size_mask])

    if pos_tensor is not None:
        num_elements_pos = arch_aligned_num_elements(arch=arch, tensor=pos_tensor)
        tile_size_pos = max_tile_size(arch, pos_tensor.dtype, num_elements_pos)

        arg_types.append(np.ndarray[(tile_size_pos,), np.dtype[pos_tensor.dtype]])
        compile_flags.append(f"-DPOS_DTYPE={dtype_to_str(pos_tensor.dtype)}")
        result_extra.extend([num_elements_pos, tile_size_pos])

    arg_types.extend([
        np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]],
        np.int32,      # tile_size
        np.float32,    # scale
        np.float32,    # max_bias
    ])
    
    compile_flags.append(f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}")
    
    # Determine function name and compile directive based on which optional tensors are present
    if mask_tensor is not None and pos_tensor is not None:
        function_name = f"ggml_op_{op_name}_with_mask_and_pos"
        compile_flags.append("-DCOMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_POS")
    elif mask_tensor is not None:
        function_name = f"ggml_op_{op_name}_with_mask"
        compile_flags.append("-DCOMPILE_GGML_OP_SOFTMAX_WITH_MAX")
    else:
        function_name = f"ggml_op_{op_name}"
        compile_flags.append("-DCOMPILE_GGML_OP_SOFTMAX")

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name=function_name,
        object_file_name=f"{op_name}_core_function.o",
        source_file=path.join(current_dir, "softmax.cc"),
        arg_types=arg_types,
        compile_flags=compile_flags,
    )

    return (func, num_elements_in, tile_size_in, *result_extra)