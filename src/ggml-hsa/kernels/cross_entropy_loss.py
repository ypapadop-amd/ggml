#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

from os import path
from typing import Tuple

import numpy as np
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


def get_cross_entropy_loss_dimensions(tensor) -> Tuple[int, int]:
    """
    Extract cross entropy loss dimensions from tensor shape.

    GGML convention: cross entropy loss is computed over dimension 0 (ne00).
    GGML shape ordering: (ne00, ne01, ne02, ne03) where ne00 is innermost.

    Parameters:
        tensor: Input tensor with shape in GGML order.

    Returns:
        Tuple of (row_length, num_rows) where:
            - row_length = ne00 (dimension over which loss is computed per row)
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


def ggml_op_cross_entropy_loss(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
):
    """
    GGML_OP_CROSS_ENTROPY_LOSS implementation.

    Cross entropy loss computes: -sum(labels * log(softmax(logits))) / num_rows
    where the softmax is computed with numerical stability.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): List of 2 input tensors:
            - input_tensors[0]: Logits tensor (predictions before softmax)
            - input_tensors[1]: Labels tensor (ground truth, often one-hot encoded)
        output_tensor: Output scalar tensor containing the loss value.
        op_params (bytearray): Operation parameters (currently unused).
    """

    if len(input_tensors) != 2:
        raise ValueError(f"Cross entropy loss requires 2 input tensors: {len(input_tensors)}")

    logits_tensor = input_tensors[0]
    labels_tensor = input_tensors[1]

    if not logits_tensor.contiguous:
        raise ValueError("Logits tensor must be contiguous in memory.")
    if not labels_tensor.contiguous:
        raise ValueError("Labels tensor must be contiguous in memory.")
    if not output_tensor.contiguous:
        raise ValueError("Output tensor must be contiguous in memory.")

    if logits_tensor.shape != labels_tensor.shape:
        raise ValueError("Logits and labels tensors must have the same shape.")

    row_length, num_rows = get_cross_entropy_loss_dimensions(logits_tensor)

    # Currently we do not support unaligned row sizes as we use vector
    # instructions with a fixed length.
    if row_length % KERN_VEC_SIZE != 0:
        raise ValueError(
            f"Row length ({row_length}) must be a multiple of {KERN_VEC_SIZE}."
        )

    # Align tile size to architecture requirements
    tile_size = align_to_arch(arch, row_length, logits_tensor.dtype, KERN_VEC_SIZE)

    # For cross entropy loss, we process one row at a time
    # Each tile contains one row of data
    if tile_size != row_length:
        raise ValueError(
            f"Tile size ({tile_size}) must equal row length ({row_length}) "
            "for cross entropy loss."
        )

    # Create external function
    function = create_external_function(
        arch=arch,
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
    )

    # Create the program with reduction pattern
    # We'll process each row and accumulate partial losses
    return create_reduction_program(
        arch=arch,
        function=function,
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
        num_rows=num_rows,
    )


def create_reduction_program(
    arch: str,
    function,
    logits_tensor,
    labels_tensor,
    output_tensor,
    tile_size: int,
    num_rows: int,
):
    """
    Create Iron program for cross entropy loss with reduction across rows.

    The kernel computes the loss for each row and the host reduces them.
    """
    num_tiles = num_rows

    logits_tile_ty = np.ndarray[(tile_size,), np.dtype[logits_tensor.dtype]]
    labels_tile_ty = np.ndarray[(tile_size,), np.dtype[labels_tensor.dtype]]
    # Each tile outputs a single scalar loss value
    output_tile_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    of_logits = ObjectFifo(logits_tile_ty, name="logits")
    of_labels = ObjectFifo(labels_tile_ty, name="labels")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_logits, of_labels, of_out, function):
        for _ in range_(num_tiles):
            elem_logits = of_logits.acquire(1)
            elem_labels = of_labels.acquire(1)
            elem_out = of_out.acquire(1)

            function(elem_logits, elem_labels, elem_out, tile_size)

            of_logits.release(1)
            of_labels.release(1)
            of_out.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_logits.cons(), of_labels.cons(), of_out.prod(), function]
    )

    rt = Runtime()
    logits_tensor_ty = np.ndarray[(tile_size * num_rows,), np.dtype[logits_tensor.dtype]]
    labels_tensor_ty = np.ndarray[(tile_size * num_rows,), np.dtype[labels_tensor.dtype]]
    # Output is an array of per-row losses that will be reduced
    output_array_ty = np.ndarray[(num_rows,), np.dtype[output_tensor.dtype]]

    with rt.sequence(logits_tensor_ty, labels_tensor_ty, output_array_ty) as (
        a_logits,
        a_labels,
        b_out,
    ):
        rt.start(worker)
        rt.fill(of_logits.prod(), a_logits)
        rt.fill(of_labels.prod(), a_labels)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def create_external_function(
    arch: str,
    logits_tensor,
    labels_tensor,
    output_tensor,
    tile_size: int,
):
    """
    Creates an external function specification for cross entropy loss.

    Returns:
        ExternalFunction object configured for the kernel.
    """
    arg_types = [
        np.ndarray[(tile_size,), np.dtype[logits_tensor.dtype]],  # logits
        np.ndarray[(tile_size,), np.dtype[labels_tensor.dtype]],  # labels
        np.ndarray[(1,), np.dtype[output_tensor.dtype]],           # output (scalar)
        np.int32,                                                   # tile_size (N)
    ]

    compile_flags = [
        f"-DINPUT_DTYPE0={dtype_to_str(logits_tensor.dtype)}",
        f"-DINPUT_DTYPE1={dtype_to_str(labels_tensor.dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        f"-DKERN_VEC_SIZE={KERN_VEC_SIZE}",
        "-DCOMPILE_GGML_OP_CROSS_ENTROPY_LOSS",
    ]

    current_dir = path.dirname(path.realpath(__file__))
    func = ExternalFunction(
        name="ggml_op_cross_entropy_loss",
        object_file_name="cross_entropy_loss_core_function.o",
        source_file=path.join(current_dir, "cross_entropy_loss.cc"),
        arg_types=arg_types,
        compile_flags=compile_flags,
    )

    return func
