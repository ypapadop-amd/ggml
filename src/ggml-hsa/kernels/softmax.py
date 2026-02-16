#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

import struct
from os import path
from typing import Any, Optional, Tuple

import numpy as np
from aie.dialects.arith import index_cast
from aie.ir import IntegerType
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_
from aie.iron.placers import SequentialPlacer

from build import (
    align_to_arch,
    arch_to_device,
)


def get_softmax_dimensions(tensor) -> Tuple[int, int]:
    """
    Extract softmax dimensions from tensor shape.

    GGML convention: softmax is over dimension 0 (ne00).
    GGML shape ordering: (ne00, ne01, ne02, ne03) where ne00 is innermost.

    Parameters:
        tensor: Input tensor with shape in GGML order.

    Returns:
        Tuple of (row_length, num_rows) where:
            - row_length = ne00 (dimension over which softmax is computed)
            - num_rows = ne01 * ne02 * ne03 (number of independent rows)
    """
    shape = tensor.shape

    if len(shape) == 1:
        # shape = (ne00,)
        return shape[0], 1
    elif len(shape) == 2:
        # shape = (ne00, ne01)
        return shape[0], shape[1]
    elif len(shape) == 3:
        # shape = (ne00, ne01, ne02)
        return shape[0], shape[1] * shape[2]
    elif len(shape) == 4:
        # shape = (ne00, ne01, ne02, ne03)
        return shape[0], shape[1] * shape[2] * shape[3]
    else:
        raise ValueError(f"Unsupported tensor rank: {len(shape)}")


# Vector size for AIE kernel vector operations
KERN_VEC_SIZE = 8


def ggml_op_soft_max(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_OP_SOFT_MAX implementation.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of 1-3 input tensors:
            - input_tensors[0]: Input tensor (required)
            - input_tensors[1]: Mask tensor (optional)
            - input_tensors[2]: Sink tensor (optional)
        output_tensor: Output tensor.
        op_params (bytearray): Operation parameters (scale, max_bias).
    """

    input_tensor_count = len(input_tensors)

    if input_tensor_count < 1 or input_tensor_count > 3:
        raise ValueError(f"Operation requires 1, 2, or 3 tensors: {input_tensor_count}")

    input_tensor = input_tensors[0]
    mask_tensor = input_tensors[1] if input_tensor_count >= 2 else None
    sink_tensor = input_tensors[2] if input_tensor_count >= 3 else None

    if not input_tensor.contiguous:
        raise ValueError("Input tensor must be contiguous in memory.")
    if not output_tensor.contiguous:
        raise ValueError("Output tensor must be contiguous in memory.")
    if mask_tensor is not None and not mask_tensor.contiguous:
        raise ValueError("Mask tensor must be contiguous in memory.")

    if sink_tensor is not None:
        raise ValueError(
            "Softmax with sink tensor is not supported on AIE. "
            "AIE tiles are limited to 2 input DMA channels, but softmax with "
            "mask and sink requires 3 input streams."
        )
    # Uncomment when the DMA-channel issue is resolved
    # if sink_tensor is not None and not sink_tensor.contiguous:
    #    raise ValueError("Sink tensor must be contiguous in memory.")

    # Currently f16 mask is not supported as we use f32 vector instructions.
    if mask_tensor is not None and mask_tensor.dtype == np.dtype("bfloat16"):
        raise ValueError("Softmax with bfloat16 mask is not supported on AIE.")

    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Input and output tensors must have the same shape.")

    # Unpack op_params: scale and max_bias
    scale = struct.unpack_from("f", op_params, 0)[0]
    max_bias = struct.unpack_from("f", op_params, 4)[0]

    op_name = "GGML_OP_SOFT_MAX"

    if input_tensor_count == 1:
        return create_unary_program(
            arch, op_name, input_tensor, output_tensor, scale, max_bias
        )
    elif input_tensor_count == 2:
        return create_binary_program(
            arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias
        )
    else:  # input_tensor_count == 3
        return create_ternary_program(
            arch,
            op_name,
            input_tensor,
            mask_tensor,
            sink_tensor,
            output_tensor,
            scale,
            max_bias,
        )


def create_unary_program(arch, op_name, input_tensor, output_tensor, scale, max_bias):
    """Softmax without mask or sink encoding."""
    function, num_elements, tile_size = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=None,
        sink_tensor=None,
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


def create_binary_program(
    arch, op_name, input_tensor, mask_tensor, output_tensor, scale, max_bias
):
    """Softmax with mask tensor."""
    func_result = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        sink_tensor=None,
        output_tensor=output_tensor,
    )
    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    tile_size_mask = func_result[3]
    num_rows_mask = func_result[4]
    num_elements_mask = func_result[5]
    n_head = func_result[6]
    rows_per_head = func_result[7]

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
        for tile_idx in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_out = of_out.acquire(1)

            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)

            function(
                elem_in,
                elem_mask,
                elem_out,
                tile_size_in,
                scale,
                max_bias,
                n_head,
                tile_idx_i32,
                rows_per_head,
            )
            of_in.release(1)
            of_mask.release(1)
            of_out.release(1)

    worker = Worker(
        ext_core_fn, fn_args=[of_in.cons(), of_mask.cons(), of_out.prod(), function]
    )

    rt = Runtime()

    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    with rt.sequence(input_tensor_ty, mask_tensor_ty, output_tensor_ty) as (
        a_in,
        a_mask,
        b_out,
    ):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_mask.prod(), a_mask)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_ternary_program(
    arch,
    op_name,
    input_tensor,
    mask_tensor,
    sink_tensor,
    output_tensor,
    scale,
    max_bias,
):
    """
    Softmax with mask tensor and sink tensor.

    Sink tensor contains one value per head. The kernel receives the full
    sink array and indexes into it based on tile_idx and rows_per_head.
    """
    func_result = create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensor,
        mask_tensor=mask_tensor,
        sink_tensor=sink_tensor,
        output_tensor=output_tensor,
    )

    function = func_result[0]
    num_elements_in = func_result[1]
    tile_size_in = func_result[2]
    tile_size_mask = func_result[3]
    num_rows_mask = func_result[4]
    num_elements_mask = func_result[5]
    num_sinks = func_result[6]
    rows_per_head = func_result[7]

    num_tiles_in = num_elements_in // tile_size_in
    num_tiles_mask = num_elements_mask // tile_size_mask

    assert num_elements_in % tile_size_in == 0
    assert num_elements_mask % tile_size_mask == 0
    assert num_elements_in == num_elements_mask
    assert num_tiles_in == num_tiles_mask

    input_tile_ty = np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]
    mask_tile_ty = np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]]

    # entire sink array passed once, not tiled
    sink_array_ty = np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]]

    of_in = ObjectFifo(input_tile_ty, name="in")
    of_mask = ObjectFifo(mask_tile_ty, name="mask")
    of_sink = ObjectFifo(sink_array_ty, name="sink", depth=1)  # Single buffer
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_in, of_mask, of_sink, of_out, function):
        # acquire sink array once at the start
        sink_array = of_sink.acquire(1)

        for tile_idx in range_(num_tiles_in):
            elem_in = of_in.acquire(1)
            elem_mask = of_mask.acquire(1)
            elem_out = of_out.acquire(1)

            # convert tile_idx from index type to i32
            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)

            function(
                elem_in,
                elem_mask,
                sink_array,
                elem_out,
                tile_size_in,
                tile_idx_i32,
                rows_per_head,
                scale,
                max_bias,
            )

            of_in.release(1)
            of_mask.release(1)
            of_out.release(1)

        # release sink array after all tiles processed
        of_sink.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_in.cons(), of_mask.cons(), of_sink.cons(), of_out.prod(), function],
    )

    rt = Runtime()

    input_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[input_tensor.dtype]]
    mask_tensor_ty = np.ndarray[(num_elements_mask,), np.dtype[mask_tensor.dtype]]
    sink_tensor_ty = np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements_in,), np.dtype[output_tensor.dtype]]

    with rt.sequence(
        input_tensor_ty, mask_tensor_ty, sink_tensor_ty, output_tensor_ty
    ) as (a_in, a_mask, a_sink, b_out):
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_mask.prod(), a_mask)
        rt.fill(of_sink.prod(), a_sink)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_external_function(
    arch: str,
    op_name: str,
    input_tensor: Any,
    mask_tensor: Optional[Any],
    sink_tensor: Optional[Any],
    output_tensor: Any,
) -> Tuple:
    """
    Creates an external function specification for softmax variants.

    Returns:
        If no mask or sink tensor:
            (func, num_elements_in, tile_size_in)
        If mask tensor only:
            (func, num_elements_in, tile_size_in, tile_size_mask, num_rows_mask, num_elements_mask, n_head, rows_per_head)
        If mask and sink tensor:
            (func, num_elements_in, tile_size_in, tile_size_mask, num_rows_mask, num_elements_mask, num_sinks, rows_per_head)
    """
    row_length_in, num_rows_in = get_softmax_dimensions(input_tensor)

    # Probably aligning doesn't make sense as we already do not allow unaligned
    # inputs but keeping this as we'll need it in the future
    tile_size_in = align_to_arch(arch, row_length_in, input_tensor.dtype, KERN_VEC_SIZE)

    # Currently we do not support unaligned row sizes as we use vector
    # instructions with a fixed length.
    if row_length_in % KERN_VEC_SIZE != 0:
        raise ValueError(
            f"Input row length ({row_length_in}) must be a multiple of {KERN_VEC_SIZE}. "
        )
    num_elements_in = tile_size_in * num_rows_in

    arg_types = [np.ndarray[(tile_size_in,), np.dtype[input_tensor.dtype]]]
    compile_flags = [
        f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
        f"-DKERN_VEC_SIZE={KERN_VEC_SIZE}",
    ]

    result_extra = []

    if mask_tensor is not None:
        row_length_mask, num_rows_mask = get_softmax_dimensions(mask_tensor)
        tile_size_mask = align_to_arch(
            arch, row_length_mask, mask_tensor.dtype, KERN_VEC_SIZE
        )
        tile_size_mask = align_to_arch(
            arch, row_length_mask, mask_tensor.dtype, KERN_VEC_SIZE
        )
        num_elements_mask = tile_size_mask * num_rows_mask

        if row_length_mask % KERN_VEC_SIZE != 0:
            raise ValueError(
                f"Mask row length ({row_length_mask}) must be a multiple of {KERN_VEC_SIZE}. "
            )

        arg_types.append(np.ndarray[(tile_size_mask,), np.dtype[mask_tensor.dtype]])
        compile_flags.append(f"-DMASK_DTYPE={dtype_to_str(mask_tensor.dtype)}")
        result_extra.extend([tile_size_mask, num_rows_mask, num_elements_mask])

        input_shape = input_tensor.shape
        if len(input_shape) >= 3:
            n_head = input_shape[2]  # ne02
        elif len(input_shape) == 2:
            n_head = 1
        else:
            n_head = 1

        rows_per_head = num_rows_in // n_head if n_head > 0 else 1
        result_extra.extend([n_head, rows_per_head])

    if sink_tensor is not None:
        # sink is 1D: one value per head
        num_sinks = sink_tensor.shape[0]
        rows_per_head = num_rows_in // num_sinks if num_sinks > 0 else 1

        arg_types.append(np.ndarray[(num_sinks,), np.dtype[sink_tensor.dtype]])
        compile_flags.append(f"-DSINK_DTYPE={dtype_to_str(sink_tensor.dtype)}")
        if mask_tensor is not None:
            result_extra = result_extra[:-2]
        result_extra.extend([num_sinks, rows_per_head])

    # output tensor
    arg_types.append(np.ndarray[(tile_size_in,), np.dtype[output_tensor.dtype]])
    compile_flags.append(f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}")

    arg_types.append(np.int32)  # tile_size

    # additional arguments for sink variant
    if sink_tensor is not None:
        arg_types.append(np.int32)  # tile_idx
        arg_types.append(np.int32)  # rows_per_head

    arg_types.append(np.float32)  # scale
    arg_types.append(np.float32)  # max_bias

    # add ALiBi parameters for mask variant (without sink)
    if mask_tensor is not None and sink_tensor is None:
        arg_types.append(np.int32)  # n_head
        arg_types.append(np.int32)  # tile_idx (passed dynamically)
        arg_types.append(np.int32)  # rows_per_head
    # determine function name and compile directive
    function_name = op_name.lower()
    if mask_tensor is not None and sink_tensor is not None:
        function_name = function_name + "_with_mask_and_sinks"
        compile_flags.append(f"-D{op_name}_WITH_MASK_AND_SINKS=1")
    elif mask_tensor is not None:
        function_name = function_name + "_with_mask"
        compile_flags.append(f"-D{op_name}_WITH_MASK=1")
    else:
        compile_flags.append(f"-D{op_name}=1")

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name=function_name,
        object_file_name=f"{function_name}_core_function.o",
        source_file=path.join(current_dir, "softmax.cc"),
        arg_types=arg_types,
        compile_flags=compile_flags,
    )

    return (func, num_elements_in, tile_size_in, *result_extra)
