#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

"""
IRON kernel implementation for binary element-wise operations.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import (
    suppress_import_pyxrt_msg,
    arch_aligned_num_elements,
    arch_to_device,
    max_tile_size,
)

suppress_import_pyxrt_msg()

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
from aie.dialects.arith import index_cast
from aie.ir import IntegerType


def _ggml_can_repeat(t0_shape: tuple, t1_shape: tuple) -> bool:
    """Python reimplementation of ggml_can_repeat.

    Checks if tensor t0 can be repeated to fill tensor t1.
    This is the GGML broadcast semantic: t1->ne[i] % t0->ne[i] == 0 for all dims.

    From ggml.c:
        bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
            return (t1->ne[0]%t0->ne[0] == 0) &&
                   (t1->ne[1]%t0->ne[1] == 0) &&
                   (t1->ne[2]%t0->ne[2] == 0) &&
                   (t1->ne[3]%t0->ne[3] == 0);
        }

    Parameters:
        t0_shape: Shape of the smaller tensor to be repeated.
        t1_shape: Shape of the larger tensor to fill.

    Returns:
        True if t0 can be repeated to fill t1.
    """
    for i in range(4):
        if t1_shape[i] % t0_shape[i] != 0:
            return False
    return True


@dataclass(frozen=True)
class CoreFunctionSpec:
    """Specification for a core function to be used in binary operations.

    Attributes:
        external_function (ExternalFunction): The external function to be called for the binary operation.
        num_elements (int): The total number of elements in the input/output tensors.
    """

    external_function: ExternalFunction
    num_elements: int

    @property
    def tile_size(self) -> int:
        """Returns the tile size used by the external function."""
        return self.external_function.tile_size(0)


def _binary_op(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """
    Implements output_tensor = op(*input_tensors)

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Input tensors.
        function_spec (CoreFunctionSpec): Binary operator specification.
        output_tensor: Output tensor.
    """

    # Tile size and number of tiles
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size
    num_tiles = num_elements // tile_size
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be divisible by tile size ({tile_size})."
        )

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
    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def _create_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> CoreFunctionSpec:
    """
    Creates a specification for binary ops.

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the operation.
        input_tensors (list): List of input tensors.
        output_tensor: Output tensor.

    Returns:
        CoreFunctionSpec: Specification for the core function to be used in binary ops.
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
    """Specification for a broadcast binary operation.

    Attributes:
        external_function (ExternalFunction): The external function for broadcast op.
        num_elements_out (int): Total number of elements in output (and src0).
        num_elements_src1 (int): Total number of elements in src1 (smaller).
        src1_ne (tuple): Shape of src1 as 4-element tuple (ne0, ne1, ne2, ne3).
        dst_ne (tuple): Shape of dst as 4-element tuple (ne0, ne1, ne2, ne3).
    """

    external_function: ExternalFunction
    num_elements_out: int
    num_elements_src1: int
    src1_ne: tuple  # (ne0, ne1, ne2, ne3)
    dst_ne: tuple  # (ne0, ne1, ne2, ne3)

    @property
    def tile_size(self) -> int:
        """Returns the tile size used by the external function."""
        return self.external_function.tile_size(0)


def _create_broadcast_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> BroadcastFunctionSpec:
    """
    Creates a specification for broadcast binary ops.

    In broadcast mode, src1 is smaller than src0/dst and gets repeated.
    The kernel receives the full src1 buffer and uses modulo indexing.

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the operation.
        input_tensors (list): List of input tensors [src0, src1].
        output_tensor: Output tensor.

    Returns:
        BroadcastFunctionSpec: Specification for broadcast binary ops.
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


def _binary_op_broadcast(
    arch: str,
    input_tensors: list,
    function_spec: BroadcastFunctionSpec,
    output_tensor,
):
    """
    Binary op with broadcasting - src1 loaded fully once, src0 streamed in tiles.

    Parameters:
        arch (str): Target architecture.
        input_tensors (list): Input tensors [src0, src1].
        function_spec (BroadcastFunctionSpec): Broadcast operation specification.
        output_tensor: Output tensor.
    """
    num_elements_out = function_spec.num_elements_out
    num_elements_src1 = function_spec.num_elements_src1
    tile_size = function_spec.tile_size
    num_tiles = num_elements_out // tile_size
    src1_ne = function_spec.src1_ne
    dst_ne = function_spec.dst_ne

    if num_elements_out % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements_out}) must be divisible by tile size ({tile_size})."
        )

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

    return Program(arch_to_device(arch), rt).resolve_program(SequentialPlacer())


def binary_op(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
):
    """
    IRON generic design for binary operations.

    Supports both element-wise operations (same shape) and broadcasting
    (src1 smaller, gets repeated to match src0/dst).

    Parameters:
        arch (str): Target architecture.
        op_name (str): Name of the operation.
        input_tensors (list): List of two input tensors [src0, src1].
        output_tensor: Output tensor.
    """

    if len(input_tensors) != 2:
        raise ValueError("Operation requires exactly two input tensors.")

    if (
        any(t.contiguous is False for t in input_tensors)
        or output_tensor.contiguous is False
    ):
        raise ValueError("Input and output tensors must be contiguous in memory.")

    src0_shape = input_tensors[0].shape
    src1_shape = input_tensors[1].shape
    dst_shape = output_tensor.shape

    # src0 must match output shape
    if src0_shape != dst_shape:
        raise ValueError(f"src0 shape must match output: {src0_shape} != {dst_shape}")

    # Check if broadcasting is needed
    needs_broadcast = src1_shape != dst_shape

    if needs_broadcast:
        # Validate broadcasting is supported per GGML semantics
        # ggml_can_repeat(src1, dst) checks if src1 can be repeated to fill dst
        if not _ggml_can_repeat(src1_shape, dst_shape):
            raise ValueError(f"Cannot broadcast: {src1_shape} -> {dst_shape}")

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
    else:
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
