#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for GGML_OP_CROSS_ENTROPY_LOSS."""

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

from .utils import align_to_arch, arch_to_device, row_dimensions

# Vector size for AIE kernel vector operations
KERN_VEC_SIZE = 8


def cross_entropy_loss(arch: str, input_tensors: list, output_tensor):
    """Build the cross entropy loss IRON program: mean loss over all rows.

    Args:
        arch: Target AIE architecture.
        input_tensors: Input tensors (logits, labels).
        output_tensor: Output scalar tensor.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, shape mismatch, contiguity, or a row
            length that is not already tile-aligned.
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

    row_length, num_rows = row_dimensions(logits_tensor)

    # Round row_length up to a KERN_VEC_SIZE-aligned tile size.
    tile_size = align_to_arch(arch, row_length, logits_tensor.dtype, KERN_VEC_SIZE)

    # This kernel processes exactly one row per tile (no intra-row tiling), so a tile
    # size larger than the row itself would mean the ObjectFifo's fixed-size DMA
    # transfer reads past the row boundary; reject rows whose length does not already
    # satisfy the alignment rather than silently padding.
    if tile_size != row_length:
        msg = (
            f"Tile size ({tile_size}) must equal row length ({row_length}) "
            "for cross entropy loss."
        )
        raise ValueError(msg)

    function = _create_external_function(
        logits_tensor=logits_tensor,
        labels_tensor=labels_tensor,
        output_tensor=output_tensor,
        tile_size=tile_size,
    )

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
    """Build the IRON program that sums per-row losses and averages by num_rows.

    Args:
        arch: Target AIE architecture.
        function: External kernel function to invoke per row.
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output scalar tensor.
        tile_size: Row/tile size.
        num_rows: Number of rows to reduce over.

    Returns:
        The resolved IRON program (MLIR module).
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
        # Acquired once and released only at the end so the DMA drains exactly 1 scalar,
        # regardless of num_rows.
        elem_out = of_out.acquire(1)

        c0_index = arith_dialect.ConstantOp(
            IndexType.get(), IntegerAttr.get(IndexType.get(), 0)
        ).result
        zero_f32 = arith_dialect.ConstantOp(
            F32Type.get(), FloatAttr.get(F32Type.get(), 0.0)
        ).result
        nr_f32 = arith_dialect.ConstantOp(
            F32Type.get(), FloatAttr.get(F32Type.get(), float(num_rows))
        ).result

        memref_dialect.StoreOp(zero_f32, elem_out, [c0_index])

        for _ in range_(num_tiles):
            elem_logits = of_logits.acquire(1)
            elem_labels = of_labels.acquire(1)

            # Must read the running total before calling function(), which overwrites
            # elem_out with this row's loss.
            prev_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result

            function(elem_logits, elem_labels, elem_out, tile_size)

            row_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result
            new_total = arith_dialect.AddFOp(prev_loss, row_loss).result
            memref_dialect.StoreOp(new_total, elem_out, [c0_index])

            of_logits.release(1)
            of_labels.release(1)

        total_loss = memref_dialect.LoadOp(elem_out, [c0_index]).result
        avg_loss = arith_dialect.DivFOp(total_loss, nr_f32).result
        memref_dialect.StoreOp(avg_loss, elem_out, [c0_index])

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

    Args:
        logits_tensor: Logits tensor.
        labels_tensor: Labels tensor.
        output_tensor: Output tensor.
        tile_size: Row/tile size.

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
