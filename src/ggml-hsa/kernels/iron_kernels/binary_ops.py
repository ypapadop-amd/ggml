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
from aie.helpers.taplib import TensorAccessPattern
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
    _arch_params,
    align_to_arch,
    arch_aligned_num_elements,
    arch_num_columns,
    arch_to_device,
    core_function_object,
    fill_drain_program,
    max_tile_size,
    partition_units,
    shape_defines,
    tiled_tile_size,
    vector_lanes,
)

# These cores' frames exceed the AIE core's 1024-byte default stack once the kernel is IR-linked
# into them rather than linked as a .o -- i.e. under GGML_HSA_KERNEL_INLINE=1. Without an explicit
# size the core would write past the end of its own stack into neighbouring core data memory,
# which is why newer mlir-aie makes this a build error rather than a warning.
#
# Measured by aiecc on aie2p, f32, with inlining on:
#   element-wise (_binary_op)      2368 bytes
#   row          (_binary_op_row)  1088 bytes for ADD; MUL still fits under 1024
# 4096 is the next power of two above the largest, leaving headroom so a small codegen change
# does not re-break the build. It is charged per core against 64 KB of core data memory, against
# a tile budget that already intends to use only half, so the cost is not material.
#
# Declared for every worker in this file, not just the paths measured over: the margin is
# op-dependent and small, so sizing only what trips today leaves the same trap for the next op or
# shape. Same pattern as unary_ops.py, softmax.py, cross_entropy_loss.py and gemm.py.
_STACK_SIZE_BYTES = 4096


# Fewest tiles worth a worker of their own. The shared fan_out_worker_count asks for 256K
# elements per worker, which a binary op passes only well above MNIST sizes; but these designs are
# DMA- and per-tile-bound once the kernel is vectorized, and spreading tiles over columns paid off
# far below that. Measured on aie2, ADD, min us, 1 -> 4 workers: 500 rows of 10 (500 tiles)
# 263 -> 166, 512 rows of 100 242 -> 167, 256x256 element-wise (64 tiles) 212 -> 169, 16 tiles of
# 1024 143 -> 139; but 8 rows of 32 117 -> 131 and 16 rows of 64 123 -> 133, the fixed cost of
# the extra fills, drains and cores showing.
_MIN_TILES_PER_WORKER = 16


def _worker_count(arch: str, num_tiles: int) -> int:
    """Workers (one per column) to spread num_tiles tiles over, at least _MIN_TILES_PER_WORKER each.

    Args:
        arch: Target architecture.
        num_tiles: Tiles the op streams.

    Returns:
        The worker count (>= 1).
    """
    return max(1, min(arch_num_columns(arch), num_tiles // _MIN_TILES_PER_WORKER))


def _ggml_can_repeat(t0_shape: tuple, t1_shape: tuple) -> bool:
    """Whether t0 can be repeated to fill t1 (GGML broadcast: t1[i] % t0[i] == 0).

    Args:
        t0_shape: Shape of the tensor to be repeated.
        t1_shape: Target shape to fill.

    Returns:
        True if t0 can be broadcast to t1.
    """
    return all(t1_shape[i] % t0_shape[i] == 0 for i in range(4))


def _banded_program(
    arch: str,
    workers: list,
    *,
    num_elements: int,
    tile_size: int,
    tiles_per_worker: list[int],
    first_tiles: list[int],
    input_tys: list,
    output_ty,
    in_prods: list[list],
    out_conses: list,
    banded: list[bool],
):
    """Program for workers that each stream a contiguous band of the tiles.

    Worker w handles tiles [first_tiles[w], first_tiles[w] + tiles_per_worker[w]) of every
    banded input and of the output, each moved linearly by its own column's shim DMA. A
    non-banded input is filled whole into every worker (an operand each worker reuses).

    Args:
        arch: Target architecture.
        workers: One Worker per band.
        num_elements: Elements in each banded tensor.
        tile_size: Elements per tile.
        tiles_per_worker: Tiles in each worker's band.
        first_tiles: Index of each worker's first tile.
        input_tys: Host buffer type per input tensor, in src order.
        output_ty: Host buffer type for the output tensor.
        in_prods: in_prods[i][w] is worker w's producer for input i.
        out_conses: Output consumer per worker.
        banded: Per input, whether it is split in bands (else filled whole into each worker).

    Returns:
        The resolved IRON program (MLIR module).
    """
    num_inputs = len(input_tys)
    n_workers = len(workers)

    def band(w):
        return TensorAccessPattern(
            (num_elements,),
            first_tiles[w] * tile_size,
            [1, 1, tiles_per_worker[w], tile_size],
            [0, 0, tile_size, 1],
        )

    # Bound positionally by Runtime: (*input buffers, output buffer, *producer lists,
    # consumer list). Every drain is waited on: the bands can finish in any order.
    def sequence(*args):
        in_bufs = args[:num_inputs]
        out_buf = args[num_inputs]
        prods = args[num_inputs + 1 : -1]
        conses = args[-1]
        for i in range(num_inputs):
            for w in range(n_workers):
                if banded[i]:
                    prods[i][w].fill(in_bufs[i], band(w))
                else:
                    prods[i][w].fill(in_bufs[i])
        for w in range(n_workers):
            conses[w].drain(out_buf, band(w), wait=True)

    rt = Runtime(sequence, [*input_tys, output_ty, *in_prods, out_conses])
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


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
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    # Fan the tiles out across columns in contiguous bands, one worker and shim DMA each: with
    # the vector body, one column's DMA and the per-tile round trips are what bound the op.
    n_workers = _worker_count(arch, num_tiles)
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)

    def suffix(w):
        return "" if n_workers == 1 else f"_{w}"

    of_ins = [
        [
            ObjectFifo(input_tile_ty, name=f"in{index}{suffix(w)}")
            for index, input_tile_ty in enumerate(input_tile_tys)
        ]
        for w in range(n_workers)
    ]
    of_outs = [
        ObjectFifo(output_tile_ty, name=f"out{suffix(w)}") for w in range(n_workers)
    ]
    function = function_spec.external_function

    # Task for the core to perform with an external function
    def ext_core_fn(of_in0, of_in1, of_out, function, n_tiles):
        # Number of sub-vector "tile" iterations
        for _ in range_(n_tiles):
            elem_in0 = of_in0.acquire(1)
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in0, elem_in1, elem_out, tile_size)
            of_in0.release(1)
            of_in1.release(1)
            of_out.release(1)

    # Create a worker per band to run the task on a compute tile
    workers = [
        Worker(
            ext_core_fn,
            fn_args=[x.cons() for x in of_ins[w]]
            + [of_outs[w].prod(), function, tiles_per_worker[w]],
            stack_size=_STACK_SIZE_BYTES,
        )
        for w in range(n_workers)
    ]

    # Runtime operations to move data to/from the AIE-array
    input_tensor_tys = [
        np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
        for input_tensor in input_tensors
    ]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    # Place program components (assign them resources on the device) and generate an MLIR module
    if n_workers == 1:
        return fill_drain_program(
            arch,
            workers,
            input_tys=input_tensor_tys,
            output_ty=output_tensor_ty,
            in_prods=[of_in.prod() for of_in in of_ins[0]],
            out_cons=of_outs[0].cons(),
        )
    return _banded_program(
        arch,
        workers,
        num_elements=num_elements,
        tile_size=tile_size,
        tiles_per_worker=tiles_per_worker,
        first_tiles=first_tiles,
        input_tys=input_tensor_tys,
        output_ty=output_tensor_ty,
        in_prods=[[of_ins[w][i].prod() for w in range(n_workers)] for i in range(2)],
        out_conses=[of_out.cons() for of_out in of_outs],
        banded=[True, True],
    )


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
    # One shared element count feeds all three fifos, and it is also the exact length of the
    # output transfer: the HSA buffer type allocates precisely ggml_nbytes for a non-quantized
    # tensor (ggml-hsa.cpp, get_alloc_size), so streaming more than the output holds writes past
    # what ggml reserved.
    #
    # That rules out rounding the count up to satisfy align_to_arch's 4-byte rule -- including
    # the rounding arch_aligned_num_elements would do on the output itself, which is why the
    # logical count is validated here rather than an already-rounded one. A 1-element bf16
    # output rounds to 2 and would then pass any check made after the fact, while the kernel
    # still transfers two elements from a one-element tensor.
    #
    # A shape whose logical count does not already satisfy every fifo's dtype is therefore
    # rejected, and dispatch falls back, rather than being served by a mis-sized kernel. f32 is
    # unaffected: a 4-byte element type never needs rounding.
    num_elements = output_tensor.numel()
    for fifo_dtype in (
        input_tensors[0].dtype,
        input_tensors[1].dtype,
        output_tensor.dtype,
    ):
        if align_to_arch(arch, num_elements, fifo_dtype) != num_elements:
            msg = (
                f"Element count ({num_elements}) is not 4-byte aligned for fifo dtype "
                f"{fifo_dtype}; rounding it up would transfer more than the output holds."
            )
            raise ValueError(msg)

    # Three fifos here (in0, in1, out), not the unary default of two, and GGML does not
    # require src1 to share the output type -- so charge the budget each fifo's own width.
    tile_size = tiled_tile_size(
        arch,
        output_tensor.dtype,
        num_elements,
        fifo_dtypes=(
            input_tensors[0].dtype,
            input_tensors[1].dtype,
            output_tensor.dtype,
        ),
    )

    current_dir = Path(__file__).resolve().parent
    # Verified to compile with GGML_HSA_KERNEL_INLINE.
    func = ExternalFunction(
        name=op_name.lower(),
        **core_function_object(f"{op_name.lower()}_core_function"),
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
            # L1-budgeted tile, so the shared transform_vector_walk_n may use its vector body.
            # This builder serves ADD/SUB/MUL/DIV; DIV's vector body (div_n) does not use it
            # and simply ignores the flag. The generic broadcast path keeps the one-register tile and is
            # deliberately left without it.
            "-DGGML_VECTORIZED_TILING=1",
            *(
                ["-DGGML_TILE_VECTOR_ALIGNED=1"]
                if tile_size % vector_lanes(arch, output_tensor.dtype) == 0
                else []
            ),
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
        **core_function_object(f"{op_name.lower()}_broadcast_core_function"),
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
    src1_rows: bool = False,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for a row-tiled broadcast kernel.

    src1 is a single row (ne0 elements) reused across all dst rows, or with src1_rows, a src1
    of several ne0-element rows, each dst row reading the one GGML's repeat maps it to. The
    tile is exactly one dst row, so tile_size == ne0.

    Args:
        arch: Target architecture.
        op_name: Name of the binary operation (e.g. "GGML_OP_ADD").
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.
        src1_rows: src1 has several rows (the GGML_OP_<op>_ROWS kernels, which also take the
            dst row index).

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

    row_op_name = f"{op_name}_ROWS" if src1_rows else f"{op_name}_ROW"
    # The shapes that map a dst row to its src1 row, baked in so the kernel's divisions are by
    # constants.
    src1_ne, dst_ne = input_tensors[1].shape, output_tensor.shape
    row_defines = (
        shape_defines(
            "GGML_BCAST",
            {
                "DST_NE1": dst_ne[1],
                "DST_NE2": dst_ne[2],
                "SRC1_NE1": src1_ne[1],
                "SRC1_NE2": src1_ne[2],
                "SRC1_NE3": src1_ne[3],
            },
        )
        if src1_rows
        else []
    )
    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=row_op_name.lower(),
        **core_function_object(f"{row_op_name.lower()}_core_function"),
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
            *([np.int32, np.int32] if src1_rows else []),
        ],
        compile_flags=[
            f"-D{row_op_name}=1",
            *row_defines,
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
            # Tile is one whole dst row, so the shared transform_vector_walk_n may use its vector
            # body. (For a narrow row that tile is still below one vector register; the body
            # then degrades to its scalar tail.) The generic broadcast path keeps the
            # one-register tile and is deliberately left without this flag.
            "-DGGML_VECTORIZED_TILING=1",
            *(
                ["-DGGML_TILE_VECTOR_ALIGNED=1"]
                if tile_size % vector_lanes(arch, output_tensor.dtype) == 0
                else []
            ),
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)


def _create_scalar_rows_external_function(
    arch: str,
    op_name: str,
    input_tensors: list,
    output_tensor,
) -> tuple[CoreFunctionSpec, int] | None:
    """Create the CoreFunctionSpec for a src1 of one element per dst row (a conv bias).

    src1 is 1 along dst's leading dims (at least ne0), so dst folds into rows of the longest
    run of leading dims that fits a tile, each row combined with the one src1 element GGML's
    repeat maps it to. A tile holds the largest whole number of rows dividing the row count
    that fits half the core's data memory next to the resident src1.

    Args:
        arch: Target architecture.
        op_name: Name of the binary operation (e.g. "GGML_OP_ADD").
        input_tensors: Two input tensors [src0, src1], all three tensors of one dtype.
        output_tensor: Output tensor.

    Returns:
        The CoreFunctionSpec and the row length, or None when not even one row of ne0
        elements fits a tile or a row is not a whole number of 4-byte words.
    """
    src1_ne, dst_ne = input_tensors[1].shape, output_tensor.shape
    dtype = output_tensor.dtype
    num_elements = output_tensor.numel()
    num_elements_src1 = arch_aligned_num_elements(arch=arch, tensor=input_tensors[1])
    budget = (
        _arch_params(arch)["core_data_mem_bytes"] // 2
        - num_elements_src1 * input_tensors[1].dtype.itemsize
    )
    cap = budget // (2 * (input_tensors[0].dtype.itemsize + dtype.itemsize))

    # Fold the leading dims src1 is 1 in, as many as fit a tile.
    k, row_len = 0, 1
    while k < 4 and src1_ne[k] == 1 and row_len * dst_ne[k] <= cap:
        row_len *= dst_ne[k]
        k += 1
    if k == 0 or (row_len * dtype.itemsize) % 4 != 0:
        return None
    n_rows = num_elements // row_len
    rows = next(r for r in range(cap // row_len, 0, -1) if n_rows % r == 0)
    tile_size = rows * row_len

    # dst rows over the remaining dims, src1 viewed as rows of one element.
    rest_dst = (*dst_ne[k:], 1, 1, 1)
    rest_src1 = (*src1_ne[k:], 1, 1, 1)
    row_op_name = f"{op_name}_SCALAR_ROWS"
    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=row_op_name.lower(),
        **core_function_object(f"{row_op_name.lower()}_core_function"),
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]],
            np.ndarray[(tile_size,), np.dtype[dtype]],
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        compile_flags=[
            f"-D{row_op_name}=1",
            *shape_defines(
                "GGML_BCAST",
                {
                    "DST_NE1": rest_dst[0],
                    "DST_NE2": rest_dst[1],
                    "SRC1_NE1": rest_src1[0],
                    "SRC1_NE2": rest_src1[1],
                    "SRC1_NE3": rest_src1[2],
                },
            ),
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(dtype)}",
            # Rows are whole tiles' worth of elements, so the shared transform_vector_walk_n may
            # use its vector body; it checks each row's alignment.
            "-DGGML_VECTORIZED_TILING=1",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements), row_len


def _binary_op_row(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
    src1_rows: bool = False,
    scalar_row_len: int = 0,
):
    """Row-tiled broadcast: out[row] = op(src0[row], src1) (one src1 row reused).

    With src1_rows, src1 holds several rows and the kernel also gets the dst row index, to
    pick the src1 row GGML's repeat maps that dst row to. With scalar_row_len, a tile holds
    tile_size // scalar_row_len rows of that many elements, each combined with one src1
    element (the GGML_OP_<op>_SCALAR_ROWS kernels).

    Args:
        arch: Target architecture.
        input_tensors: Input tensors [src0, src1]; src1 is one ne0-element row, or several.
        function_spec: Core function specification (tile_size == ne0, or whole scalar rows).
        output_tensor: Output tensor.
        src1_rows: src1 has several rows (the GGML_OP_<op>_ROWS kernels).
        scalar_row_len: Elements per row for the GGML_OP_<op>_SCALAR_ROWS kernels, else 0.

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

    # Rows fan out across columns in contiguous bands, as in _binary_op; each worker gets its
    # own copy of the src1 row.
    n_workers = _worker_count(arch, num_tiles)
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)

    def suffix(w):
        return "" if n_workers == 1 else f"_{w}"

    of_src0s = [
        ObjectFifo(src0_tile_ty, name=f"src0{suffix(w)}") for w in range(n_workers)
    ]
    # loaded once, reused
    of_src1s = [
        ObjectFifo(src1_row_ty, depth=1, name=f"src1{suffix(w)}")
        for w in range(n_workers)
    ]
    of_outs = [
        ObjectFifo(out_tile_ty, name=f"out{suffix(w)}") for w in range(n_workers)
    ]

    function = function_spec.external_function

    def ext_core_fn(of_src0, of_src1, of_out, function, n_tiles, first_tile):
        src1_buf = of_src1.acquire(1)  # src1, reused across all tiles
        for tile_idx in range_(n_tiles):
            src0_tile = of_src0.acquire(1)
            out_tile = of_out.acquire(1)
            if scalar_row_len:
                # first dst row = (first_tile + tile_idx) * rows, formed in the kernel
                rows = tile_size // scalar_row_len
                tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
                function(
                    src0_tile,
                    src1_buf,
                    out_tile,
                    scalar_row_len,
                    rows,
                    first_tile * rows,
                    tile_idx_i32,
                )
            elif src1_rows:
                # dst row = first_tile + tile_idx, added in the kernel
                tile_idx_i32 = index_cast(IntegerType.get_signless(32), tile_idx)
                function(
                    src0_tile, src1_buf, out_tile, tile_size, first_tile, tile_idx_i32
                )
            else:
                function(src0_tile, src1_buf, out_tile, tile_size)
            of_src0.release(1)
            of_out.release(1)
        of_src1.release(1)

    workers = [
        Worker(
            ext_core_fn,
            fn_args=[
                of_src0s[w].cons(),
                of_src1s[w].cons(),
                of_outs[w].prod(),
                function,
                tiles_per_worker[w],
                first_tiles[w],
            ],
            stack_size=_STACK_SIZE_BYTES,
        )
        for w in range(n_workers)
    ]

    # Buffers in src order then dst (kernarg layout contract).
    src0_ty = np.ndarray[(num_elements,), np.dtype[input_tensors[0].dtype]]
    src1_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    if n_workers == 1:
        return fill_drain_program(
            arch,
            workers,
            input_tys=[src0_ty, src1_ty],
            output_ty=out_ty,
            in_prods=[of_src0s[0].prod(), of_src1s[0].prod()],
            out_cons=of_outs[0].cons(),
        )
    return _banded_program(
        arch,
        workers,
        num_elements=num_elements,
        tile_size=tile_size,
        tiles_per_worker=tiles_per_worker,
        first_tiles=first_tiles,
        input_tys=[src0_ty, src1_ty],
        output_ty=out_ty,
        in_prods=[[of.prod() for of in of_src0s], [of.prod() for of in of_src1s]],
        out_conses=[of.cons() for of in of_outs],
        banded=[True, False],
    )


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
        stack_size=_STACK_SIZE_BYTES,
    )

    # Runtime operations to move data to/from the AIE-array
    src0_ty = np.ndarray[(num_elements_out,), np.dtype[input_tensors[0].dtype]]
    src1_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_ty = np.ndarray[(num_elements_out,), np.dtype[output_tensor.dtype]]

    return fill_drain_program(
        arch,
        [worker],
        input_tys=[src0_ty, src1_ty],
        output_ty=out_ty,
        in_prods=[of_src0.prod(), of_src1.prod()],
        out_cons=of_out.cons(),
    )


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

    # src1 rows as long as dst's, repeated along the other dimensions: tile by dst row, with
    # each row reading its whole src1 row, rather than mapping every element to src1 (the
    # generic path below, which divides per element). A row whose bytes are not a whole number
    # of 4-byte words for some operand cannot be one DMA tile, and takes the generic path.
    rows_tileable = all(
        (dst_shape[0] * dtype.itemsize) % 4 == 0
        for dtype in (
            input_tensors[0].dtype,
            input_tensors[1].dtype,
            output_tensor.dtype,
        )
    )
    if (
        op_name in _ROW_BROADCAST_OPS
        and needs_broadcast
        and src1_shape[0] == dst_shape[0]
        and rows_tileable
        and _ggml_can_repeat(src1_shape, dst_shape)
    ):
        function_spec = _create_row_external_function(
            arch=arch,
            op_name=op_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            src1_rows=True,
        )
        return _binary_op_row(
            arch=arch,
            input_tensors=input_tensors,
            function_spec=function_spec,
            output_tensor=output_tensor,
            src1_rows=True,
        )

    # src1 one element per run of dst's leading dims (a conv bias [1, 1, C]): tile by whole
    # runs, each combined with its src1 element, rather than mapping every element to src1.
    # aie2p only so far: aie2 keeps the generic path below.
    if (
        arch == "aie2p"
        and op_name in _ROW_BROADCAST_OPS
        and needs_broadcast
        and same_dtype
        and src1_shape[0] == 1
        and _ggml_can_repeat(src1_shape, dst_shape)
    ):
        scalar_rows = _create_scalar_rows_external_function(
            arch=arch,
            op_name=op_name,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        )
        if scalar_rows is not None:
            function_spec, row_len = scalar_rows
            return _binary_op_row(
                arch=arch,
                input_tensors=input_tensors,
                function_spec=function_spec,
                output_tensor=output_tensor,
                scalar_row_len=row_len,
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
