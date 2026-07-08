#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_CROSS_ENTROPY_LOSS.

Computes -sum(labels * log_softmax(logits)) / num_rows with numerically stable
log-softmax (max subtraction). One row is processed per worker iteration and
per-row losses are accumulated on-tile into a single scalar.
"""

from pathlib import Path

import numpy as np
from aie.dialects import arith as arith_dialect
from aie.dialects import memref as memref_dialect
from aie.ir import F32Type, FloatAttr, IndexType, IntegerAttr
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_

from .utils import align_to_arch, arch_to_device


def get_cross_entropy_loss_dimensions(tensor) -> tuple[int, int]:
    """Return (row_length, num_rows) for a GGML-ordered tensor.

    Loss is computed over dim 0 (ne00), so row_length = ne00 and
    num_rows = ne01 * ne02 * ne03.

    Parameters:
        tensor: GGML-ordered tensor of rank 1 to 4.

    Returns:
        The (row_length, num_rows) pair.

    Raises:
        ValueError: If the tensor rank is unsupported.

    """
    shape = tensor.shape

    if len(shape) == 1:
        # shape = (ne00,)
        return shape[0], 1
    if len(shape) == 2:
        # shape = (ne00, ne01)
        return shape[0], shape[1]
    if len(shape) == 3:
        # shape = (ne00, ne01, ne02)
        return shape[0], shape[1] * shape[2]
    if len(shape) == 4:
        # shape = (ne00, ne01, ne02, ne03)
        return shape[0], shape[1] * shape[2] * shape[3]
    msg = f"Unsupported tensor rank: {len(shape)}"
    raise ValueError(msg)


# Vector size for AIE kernel vector operations
KERN_VEC_SIZE = 8


def cross_entropy_loss(arch: str, input_tensors: list, output_tensor):
    """Build the cross entropy loss IRON program.

    Parameters:
        arch: Target architecture.
        input_tensors: [logits, labels] of identical shape.
        output_tensor: Output scalar tensor holding the loss.

    """
    if len(input_tensors) != 2:
        msg = f"Cross entropy loss requires 2 input tensors: {len(input_tensors)}"
        raise ValueError(msg)

    logits_tensor = input_tensors[0]
    labels_tensor = input_tensors[1]

    if not logits_tensor.contiguous:
        msg = "Logits tensor must be contiguous in memory."
        raise ValueError(msg)
    if not labels_tensor.contiguous:
        msg = "Labels tensor must be contiguous in memory."
        raise ValueError(msg)
    if not output_tensor.contiguous:
        msg = "Output tensor must be contiguous in memory."
        raise ValueError(msg)

    if logits_tensor.shape != labels_tensor.shape:
        msg = "Logits and labels tensors must have the same shape."
        raise ValueError(msg)

    row_length, num_rows = get_cross_entropy_loss_dimensions(logits_tensor)

    # Align tile size to architecture requirements
    tile_size = align_to_arch(arch, row_length, logits_tensor.dtype, KERN_VEC_SIZE)

    # For cross entropy loss, we process one row at a time
    # Each tile contains one row of data
    if tile_size != row_length:
        msg = (
            f"Tile size ({tile_size}) must equal row length ({row_length}) "
            "for cross entropy loss."
        )
        raise ValueError(msg)

    # Create external function
    function = _create_external_function(
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
    )

    # Create the program with on-tile reduction
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
    """Build the IRON program that reduces per-row losses on-tile.

    The output FIFO element is acquired once. For each row the accumulated value
    is saved, the kernel overwrites the buffer with the row's loss, and the sum
    is stored back. After all rows the total is divided by num_rows and released,
    so the DMA drains exactly one float.

    Parameters:
        arch: Target architecture.
        function: The per-row loss external function.
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output tensor.
        tile_size: Number of elements per tile (row length).
        num_rows: Number of rows to process.

    """
    num_tiles = num_rows

    logits_tile_ty = np.ndarray[(tile_size,), np.dtype[logits_tensor.dtype]]
    labels_tile_ty = np.ndarray[(tile_size,), np.dtype[labels_tensor.dtype]]
    # Each tile of output is a single scalar loss value
    output_tile_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    of_logits = ObjectFifo(logits_tile_ty, name="logits")
    of_labels = ObjectFifo(labels_tile_ty, name="labels")
    of_out = ObjectFifo(output_tile_ty, name="out")

    def ext_core_fn(of_logits, of_labels, of_out, function):
        # Acquire the output buffer ONCE — we will accumulate into it
        # across all rows and release it only after the final result
        # is computed. This produces exactly 1 scalar for the DMA to drain.
        elem_out = of_out.acquire(1)

        # Create constants for memref indexing and arithmetic
        c0_index = arith_dialect.ConstantOp(
            IndexType.get(), IntegerAttr.get(IndexType.get(), 0)
        ).result
        zero_f32 = arith_dialect.ConstantOp(
            F32Type.get(), FloatAttr.get(F32Type.get(), 0.0)
        ).result
        nr_f32 = arith_dialect.ConstantOp(
            F32Type.get(), FloatAttr.get(F32Type.get(), float(num_rows))
        ).result

        # Initialize accumulated loss to zero
        memref_dialect.StoreOp(zero_f32, elem_out, [c0_index])

        for _ in range_(num_tiles):
            elem_logits = of_logits.acquire(1)
            elem_labels = of_labels.acquire(1)

            # Load accumulated loss BEFORE the kernel overwrites the buffer
            prev_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result

            # Kernel computes this row's loss -> writes to elem_out[0]
            #   loss_out[0] = -sum(labels * log_softmax(logits))
            function(elem_logits, elem_labels, elem_out, tile_size)

            # Load per-row loss that the kernel just wrote
            row_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result

            # Accumulate: new_total = prev_loss + row_loss
            new_total = arith_dialect.AddFOp(prev_loss, row_loss).result
            memref_dialect.StoreOp(new_total, elem_out, [c0_index])

            of_logits.release(1)
            of_labels.release(1)

        # Divide accumulated loss by num_rows to get the average,
        # matching the CPU reference: dp[0] = -sum_all_rows / num_rows
        total_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result
        avg_loss = arith_dialect.DivFOp(total_loss, nr_f32).result
        memref_dialect.StoreOp(avg_loss, elem_out, [c0_index])

        # Release the single output element — DMA drains this 1 float
        of_out.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_logits.cons(), of_labels.cons(), of_out.prod(), function],
    )

    rt = Runtime()
    logits_tensor_ty = np.ndarray[
        (tile_size * num_rows,), np.dtype[logits_tensor.dtype]
    ]
    labels_tensor_ty = np.ndarray[
        (tile_size * num_rows,), np.dtype[labels_tensor.dtype]
    ]

    output_scalar_ty = np.ndarray[(1,), np.dtype[output_tensor.dtype]]

    with rt.sequence(logits_tensor_ty, labels_tensor_ty, output_scalar_ty) as (
        a_logits,
        a_labels,
        b_out,
    ):
        rt.start(worker)
        rt.fill(of_logits.prod(), a_logits)
        rt.fill(of_labels.prod(), a_labels)
        rt.drain(of_out.cons(), b_out, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    logits_tensor,
    labels_tensor,
    output_tensor,
    tile_size: int,
):
    """Create the ExternalFunction wrapping cross_entropy_loss.cc.

    Parameters:
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output tensor.
        tile_size: Number of elements per tile (equals row length).

    Returns:
        The configured ExternalFunction.

    """
    arg_types = [
        np.ndarray[(tile_size,), np.dtype[logits_tensor.dtype]],  # logits
        np.ndarray[(tile_size,), np.dtype[labels_tensor.dtype]],  # labels
        np.ndarray[(1,), np.dtype[output_tensor.dtype]],  # output (scalar)
        np.int32,  # tile_size (N)
    ]

    compile_flags = []

    current_dir = Path(__file__).resolve().parent
    return ExternalFunction(
        name="ggml_op_cross_entropy_loss",
        object_file_name="cross_entropy_loss_core_function.o",
        source_file=str(current_dir / "cross_entropy_loss.cc"),
        arg_types=arg_types,
        compile_flags=compile_flags,
    )
