#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON kernel implementation for binary element-wise operations."""

from dataclasses import dataclass
from pathlib import Path

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

from .utils import (
    CoreFunctionSpec,
    arch_aligned_num_elements,
    arch_to_device,
    max_tile_size,
)


def _ggml_can_repeat(t0_shape: tuple, t1_shape: tuple) -> bool:
    """Whether t0 can be repeated to fill t1 (GGML broadcast: t1[i] % t0[i] == 0).

    Args:
        t0_shape: Shape of the tensor to be repeated.
        t1_shape: Target shape to fill.

    Returns:
        True if t0 can be broadcast to t1.
    """
    return all(t1_shape[i] % t0_shape[i] == 0 for i in range(4))


def _binary_op(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """Element-wise output_tensor = op(*input_tensors).

    Args:
        arch: Target architecture.
        input_tensors: Input tensors [src0, src1].
        function_spec: Core function specification.
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: If num_elements is not divisible by tile_size.
    """
    # Tile size and number of tiles
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size
    num_tiles = num_elements // tile_size
    if num_elements % tile_size != 0:
        msg = f"Number of elements ({num_elements}) must be divisible by tile size ({tile_size})."
        raise ValueError(msg)

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
    worker = None
    function = function_spec.external_function

    # Task for the core to perform with an external function
    def ext_core_fn(of_in0, of_in1, of_out, function):
        # Number of sub-vector "tile" iterations
        for _ in range_(num_tiles):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in0, elem_in1, elem_out, tile_size)
            of_in0.release(1)
            of_in1.release(1)
            of_out.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[x.cons() for x in of_ins] + [of_out.prod(), function],
    )

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
    return Program(arch_to_device(arch), rt).resolve_program()


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for an element-wise binary op.

    Args:
        arch: Target architecture.
        op_name: Name of the operation.
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.

    Returns:
        The configured CoreFunctionSpec.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=output_tensor)
    tile_size = max_tile_size(arch, output_tensor.dtype, num_elements)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(tile_size,), np.dtype[input_tensors[1].dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{op_name}=1",
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


@dataclass(frozen=True)
class BroadcastFunctionSpec:
    """Core function and shapes for a broadcast binary op (src1 repeated).

    Attributes:
        external_function: External function implementing the operation.
        num_elements_out: Total number of output elements.
        num_elements_src1: Total number of src1 elements.
        src1_ne: src1 shape as (ne0, ne1, ne2, ne3).
        dst_ne: Destination shape as (ne0, ne1, ne2, ne3).
    """

    external_function: ExternalFunction
    num_elements_out: int
    num_elements_src1: int
    src1_ne: tuple[int, int, int, int]  # (ne0, ne1, ne2, ne3)
    dst_ne: tuple[int, int, int, int]  # (ne0, ne1, ne2, ne3)

    @property
    def tile_size(self) -> int:
        """Tile size used by the external function."""
        return self.external_function.tile_size(0)


def _create_broadcast_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> BroadcastFunctionSpec:
    """Create the BroadcastFunctionSpec for a broadcast binary op.

    src1 is smaller than src0/dst; the kernel gets the full src1 buffer and
    uses modulo indexing to repeat it.

    Args:
        arch: Target architecture.
        op_name: Name of the operation.
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.

    Returns:
        The configured BroadcastFunctionSpec.
    """
    num_elements_out = arch_aligned_num_elements(arch=arch, tensor=output_tensor)
    num_elements_src1 = arch_aligned_num_elements(arch=arch, tensor=input_tensors[1])
    tile_size = max_tile_size(arch, output_tensor.dtype, num_elements_out)

    # Extract shapes as 4-element tuples for multi-dimensional broadcast indexing
    src1_ne = input_tensors[1].shape
    dst_ne = output_tensor.shape

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=f"{op_name.lower()}_broadcast",
        object_file_name=f"{op_name.lower()}_broadcast_core_function.o",
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],  # src0 tile
            np.ndarray[
                (num_elements_src1,), np.dtype[input_tensors[1].dtype]
            ],  # full src1
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],  # output tile
            np.int32,  # tile_size
            np.int32,  # tile_idx
            np.int32,  # src1_ne[0]
            np.int32,  # src1_ne[1]
            np.int32,  # src1_ne[2]
            np.int32,  # src1_ne[3]
            np.int32,  # dst_ne[0]
            np.int32,  # dst_ne[1]
            np.int32,  # dst_ne[2]
        ],
        compile_flags=[
            f"-D{op_name}_BROADCAST=1",
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return BroadcastFunctionSpec(
        external_function=func,
        num_elements_out=num_elements_out,
        num_elements_src1=num_elements_src1,
        src1_ne=src1_ne,
        dst_ne=dst_ne,
    )


# Binary ops with a row-tiled broadcast kernel in binary_ops.cc (GGML_OP_<op>_ROW).
_ROW_BROADCAST_OPS = frozenset(
    {"GGML_OP_ADD", "GGML_OP_SUB", "GGML_OP_MUL", "GGML_OP_DIV"}
)


def _create_row_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for a row-tiled broadcast kernel.

    src1 is a single row (ne0 elements) reused across all dst rows. The tile
    is exactly one dst row, so tile_size == ne0.

    Args:
        arch: Target architecture.
        op_name: Name of the binary operation (e.g. "GGML_OP_ADD").
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.

    Returns:
        The configured CoreFunctionSpec.

    Raises:
        ValueError: If the output element count is not divisible by ne0.
    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=output_tensor)
    ne0 = output_tensor.shape[0]
    tile_size = ne0

    if num_elements % tile_size != 0:
        msg = f"Output elements ({num_elements}) not divisible by row length ({tile_size})."
        raise ValueError(msg)

    # src1 is streamed whole through a depth-1 fifo, so its object type is the arch-aligned
    # element count, not ne0. The two differ when the aligned count rounds up (a 16-bit dtype
    # with an odd ne0 pads to ne0 + 1), and the declared arg type has to match the object the
    # kernel is handed or the IRON signatures disagree with the DMA size. The kernel still
    # reads only the first N == ne0 elements, N being a separate runtime argument.
    num_elements_src1 = arch_aligned_num_elements(arch=arch, tensor=input_tensors[1])

    row_op_name = f"{op_name}_ROW"
    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=row_op_name.lower(),
        object_file_name=f"{row_op_name.lower()}_core_function.o",
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{row_op_name}=1",
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


def _binary_op_row(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """Row-tiled broadcast: out[row] = op(src0[row], src1) (one src1 row reused).

    Args:
        arch: Target architecture.
        input_tensors: Input tensors [src0, src1]; src1 is one ne0-element row.
        function_spec: Core function specification (tile_size == ne0).
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program (MLIR module).
    """
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size  # == ne0
    num_tiles = num_elements // tile_size
    num_elements_src1 = arch_aligned_num_elements(arch=arch, tensor=input_tensors[1])

    src0_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]]
    src1_row_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    of_src0 = ObjectFifo(src0_tile_ty, name="src0")
    of_src1 = ObjectFifo(src1_row_ty, depth=1, name="src1")  # loaded once, reused
    of_out = ObjectFifo(out_tile_ty, name="out")

    function = function_spec.external_function

    def ext_core_fn(of_src0, of_src1, of_out, function):
        src1_buf = of_src1.acquire(1)  # one src1 row, reused across all tiles
        for _ in range_(num_tiles):
            src0_tile = of_src0.acquire(1)
            out_tile = of_out.acquire(1)
            function(src0_tile, src1_buf, out_tile, tile_size)
            of_src0.release(1)
            of_out.release(1)
        of_src1.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_src0.cons(), of_src1.cons(), of_out.prod(), function],
    )

    # Buffers in src order then dst (kernarg layout contract).
    src0_ty = np.ndarray[(num_elements,), np.dtype[input_tensors[0].dtype]]
    src1_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    rt = Runtime()
    with rt.sequence(src0_ty, src1_ty, out_ty) as (a, b, c):
        rt.start(worker)
        rt.fill(of_src0.prod(), a)
        rt.fill(of_src1.prod(), b)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def _binary_op_broadcast(
    arch: str,
    input_tensors: list,
    function_spec: BroadcastFunctionSpec,
    output_tensor,
):
    """Broadcast binary op: src1 loaded fully once, src0 streamed in tiles.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors [src0, src1].
        function_spec: Broadcast function specification.
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: If num_elements_out is not divisible by tile_size.
    """
    num_elements_out = function_spec.num_elements_out
    num_elements_src1 = function_spec.num_elements_src1
    tile_size = function_spec.tile_size
    num_tiles = num_elements_out // tile_size
    src1_ne = function_spec.src1_ne
    dst_ne = function_spec.dst_ne

    if num_elements_out % tile_size != 0:
        msg = f"Number of elements ({num_elements_out}) must be divisible by tile size ({tile_size})."
        raise ValueError(msg)

    # ObjectFifos for data movement
    src0_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]]
    src1_full_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    of_src0 = ObjectFifo(src0_tile_ty, name="src0")
    of_src1 = ObjectFifo(src1_full_ty, depth=1, name="src1")  # depth=1, load once
    of_out = ObjectFifo(out_tile_ty, name="out")

    function = function_spec.external_function

    def ext_core_fn(of_src0, of_src1, of_out, function):
        # Acquire src1 once (full buffer)
        src1_buf = of_src1.acquire(1)

        for tile_idx in range_(num_tiles):
            src0_tile = of_src0.acquire(1)
            out_tile = of_out.acquire(1)

            tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
            # Pass shape elements as individual scalars (compile-time constants)
            function(
                src0_tile,
                src1_buf,
                out_tile,
                tile_size,
                tile_idx_i32,
                src1_ne[0],
                src1_ne[1],
                src1_ne[2],
                src1_ne[3],
                dst_ne[0],
                dst_ne[1],
                dst_ne[2],
            )

            of_src0.release(1)
            of_out.release(1)

        of_src1.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_src0.cons(), of_src1.cons(), of_out.prod(), function],
    )

    # Runtime operations to move data to/from the AIE-array
    src0_ty = np.ndarray[(num_elements_out,), np.dtype[input_tensors[0].dtype]]
    src1_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_ty = np.ndarray[(num_elements_out,), np.dtype[output_tensor.dtype]]

    rt = Runtime()
    with rt.sequence(src0_ty, src1_ty, out_ty) as (a, b, c):
        rt.start(worker)
        rt.fill(of_src0.prod(), a)
        rt.fill(of_src1.prod(), b)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()


def binary_op(
    op_name: str,
    arch: str,
    input_tensors: list,
    output_tensor,
):
    """IRON design for binary ops (element-wise, or broadcasting src1 to src0/dst).

    Args:
        op_name: Name of the operation.
        arch: Target architecture.
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, contiguity, shape mismatch, or
            unsupported broadcast.
    """
    if len(input_tensors) != 2:
        msg = "Operation requires exactly two input tensors."
        raise ValueError(msg)

    if (
        any(t.contiguous is False for t in input_tensors)
        or output_tensor.contiguous is False
    ):
        msg = "Input and output tensors must be contiguous in memory."
        raise ValueError(msg)

    src0_shape = input_tensors[0].shape
    src1_shape = input_tensors[1].shape
    dst_shape = output_tensor.shape

    # src0 must match output shape
    if src0_shape != dst_shape:
        msg = f"src0 shape must match output: {src0_shape} != {dst_shape}"
        raise ValueError(msg)

    # Check if broadcasting is needed
    needs_broadcast = src1_shape != dst_shape

    # Row-broadcast fast path: src1 is a single row replicated over every dst row.
    # The kernels operate on the operands directly (no per-element cast), so this is
    # gated on all three dtypes matching; a mismatched-dtype operand falls through to
    # the generic broadcast path, which casts per element.
    src1_is_row = (
        src1_shape[0] == dst_shape[0]
        and src1_shape[1] == 1
        and src1_shape[2] == 1
        and src1_shape[3] == 1
    )
    same_dtype = (
        input_tensors[0].dtype == input_tensors[1].dtype
        and input_tensors[0].dtype == output_tensor.dtype
    )
    if op_name in _ROW_BROADCAST_OPS and needs_broadcast and src1_is_row and same_dtype:
        function_spec = _create_row_external_function(
            arch=arch,
            op_name=op_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        )
        return _binary_op_row(
            arch=arch,
            input_tensors=input_tensors,
            function_spec=function_spec,
            output_tensor=output_tensor,
        )

    if needs_broadcast:
        # Validate broadcasting is supported per GGML semantics
        # ggml_can_repeat(src1, dst) checks if src1 can be repeated to fill dst
        if not _ggml_can_repeat(src1_shape, dst_shape):
            msg = f"Cannot broadcast: {src1_shape} -> {dst_shape}"
            raise ValueError(msg)

        function_spec = _create_broadcast_external_function(
            arch=arch,
            op_name=op_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        )

        return _binary_op_broadcast(
            arch=arch,
            input_tensors=input_tensors,
            function_spec=function_spec,
            output_tensor=output_tensor,
        )
    # Non-broadcast path: standard element-wise operation
    function_spec = _create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )

    return _binary_op(
        arch=arch,
        input_tensors=input_tensors,
        function_spec=function_spec,
        output_tensor=output_tensor,
    )
