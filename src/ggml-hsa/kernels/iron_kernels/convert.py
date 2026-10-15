#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""IRON design for an element-wise dtype conversion (GGML_OP_CPY cast, no shape change).

Both tensors are dense and contiguous with the same number of elements; only the dtype differs
(f32 <-> bf16, f16 -> bf16, or bf16 -> f16). The tensor is flattened to 1D and streamed in
tiles,
so the same design serves any shape. This runs a pure cast on the device queue instead of the host copy
path (which drains the queue), letting the cast batch with surrounding dispatches.

Tiles are as large as half of L1 allows, and a large tensor's tiles are fanned out over one worker
per column (fewer on aie2p) in contiguous bands, each moved by its own shim DMA.
"""

import math
from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    ExternalFunction,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    dtype_to_str,
)
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

from .utils import (
    arch_num_columns,
    arch_to_device,
    divisor_tile_size,
    fan_out_worker_count,
    partition_units,
)

# Shim-DMA transfers are issued in whole 4-byte words (see align_to_arch), so a tensor whose
# byte size is not a multiple of this cannot be streamed in full.
_DMA_ALIGNMENT_BYTES = 4

# Fewest elements worth a worker (and a shim DMA) of its own. A single column streams about
# 5 GB/s, so even a few tens of thousands of elements gain from all four columns on aie2 (a
# 250000-element f32 -> bf16 measured 465 us on one worker, 283 on four).
_MIN_ELEMS_PER_WORKER = 8 * 1024

# aie2p: a worker adds a fixed cost to every dispatch, which in a graph, where every conversion sits
# between two other kernels, outweighs its share of the bandwidth until the cast is large: workers =
# sqrt(elements / this), rounded to a power of two (see _worker_cap).
_AIE2P_WORKER_ELEMENTS = 128 * 1024


def convert(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
    """Build the element-wise dtype-conversion IRON program.

    Args:
        arch: Target architecture.
        input_tensors: [src] dense contiguous tensor.
        output_tensor: dense contiguous tensor, same element count, different dtype.
        op_params: unused (kept for the dispatch ABI).

    Returns:
        The resolved IRON program (MLIR module).

    Raises:
        ValueError: On invalid tensor count, dtype, contiguity, element-count mismatch, or a
            byte size that is not a whole number of DMA words.

    """
    del op_params

    if len(input_tensors) != 1:
        msg = "convert requires exactly one input tensor."
        raise ValueError(msg)

    src = input_tensors[0]

    # convert.cc implements the narrowings f32 -> bf16 and bf16 -> f16 (both replicating the
    # host integer RNE bit-for-bit), the widening bf16 -> f32, f16 -> bf16 (exact widening then the
    # same host-identical RNE), and the same-dtype copy. Any other pair would silently fall into
    # its static_cast branch, whose rounding is Peano's rather than the host reference's, so reject
    # it here.
    supported_pairs = (
        (np.float32, bfloat16),
        (np.float32, np.float32),
        (bfloat16, np.float32),
        (bfloat16, np.float16),
        (bfloat16, bfloat16),
        (np.float16, bfloat16),
    )
    if not any(
        src.dtype == src_dt and output_tensor.dtype == dst_dt
        for src_dt, dst_dt in supported_pairs
    ):
        msg = (
            f"convert supports float32 <-> bfloat16, float16 -> bfloat16, and bfloat16 "
            f"-> float16; got src {src.dtype}, dst {output_tensor.dtype}."
        )
        raise ValueError(msg)
    if not src.contiguous or not output_tensor.contiguous:
        msg = "convert tensors must be contiguous in memory."
        raise ValueError(msg)
    if src.numel() != output_tensor.numel():
        msg = (
            f"convert requires equal element counts; got src {src.numel()}, dst "
            f"{output_tensor.numel()}."
        )
        raise ValueError(msg)

    # One element count declares both the source and the destination transfer, so it has to be
    # valid for both dtypes. Rounding it up to satisfy the stricter one is not an option here:
    # ggml allocates exactly ggml_nbytes, so a count above the tensor's own element count makes
    # the shim DMA read and write past the end of the buffer. Instead require that both tensors
    # are already a whole number of DMA words -- for a 2-byte dtype that means an even element
    # count -- and stream exactly that many elements. Anything else is left to the host copy path.
    for name, t in (("source", src), ("destination", output_tensor)):
        if (t.numel() * t.dtype.itemsize) % _DMA_ALIGNMENT_BYTES != 0:
            msg = (
                f"convert {name} of {t.numel()} x {t.dtype} is not a multiple of "
                f"{_DMA_ALIGNMENT_BYTES} bytes; cannot be streamed without over-running the "
                f"allocation."
            )
            raise ValueError(msg)

    # IRON has no f16 element type (dtype_to_str rejects np.float16), so an f16 tensor is streamed
    # as i16 -- same 2-byte element, same DMA descriptors -- and convert.cc reinterprets the bits.
    # np.dtype, not the bare scalar type: TensorDesc guarantees src.dtype is an np.dtype, and the
    # two are not interchangeable at every use site below.
    src_iron_dtype = np.dtype(np.int16) if src.dtype == np.float16 else src.dtype
    dst_iron_dtype = (
        np.dtype(np.int16) if output_tensor.dtype == np.float16 else output_tensor.dtype
    )

    # Flatten to 1D: a cast is element-wise, so any shape streams as one contiguous run.
    num_elements = src.numel()
    lanes, tile_size = _lanes_and_tile_size(
        arch, num_elements, (src_iron_dtype, dst_iron_dtype)
    )
    num_tiles = num_elements // tile_size

    function = _create_external_function(
        src=src,
        src_iron_dtype=src_iron_dtype,
        output_tensor=output_tensor,
        dst_iron_dtype=dst_iron_dtype,
        tile_size=tile_size,
        lanes=lanes,
    )

    in_tile_ty = np.ndarray[(tile_size,), np.dtype[src_iron_dtype]]
    out_tile_ty = np.ndarray[(tile_size,), np.dtype[dst_iron_dtype]]

    # Fan the tiles out across columns, one worker each with its own shim DMA, in contiguous
    # bands moved linearly; small casts stay on one worker.
    n_workers = fan_out_worker_count(
        arch,
        num_elements,
        num_tiles,
        max_workers=_worker_cap(arch, num_elements),
        min_elems_per_worker=_MIN_ELEMS_PER_WORKER,
    )
    tiles_per_worker, first_tiles = partition_units(n_workers, num_tiles)
    of_ins = [ObjectFifo(in_tile_ty, name=f"in{w}") for w in range(n_workers)]
    of_outs = [ObjectFifo(out_tile_ty, name=f"out{w}") for w in range(n_workers)]

    def core_fn(of_in, of_out, function, n_tiles):
        for _ in range_(n_tiles):
            tile_in = of_in.acquire(1)
            tile_out = of_out.acquire(1)
            function(tile_in, tile_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[
                of_ins[w].cons(),
                of_outs[w].prod(),
                function,
                tiles_per_worker[w],
            ],
        )
        for w in range(n_workers)
    ]

    def band(w):
        return TensorAccessPattern(
            (num_elements,),
            first_tiles[w] * tile_size,
            [1, 1, tiles_per_worker[w], tile_size],
            [0, 0, tile_size, 1],
        )

    # One host buffer per ggml tensor, src then dst. Every drain is waited on: the bands can
    # finish in any order. A band's tiles are written back only after they were read, and no
    # band reads another's elements, so the backend may convert in place (dst aliasing src).
    def sequence(in_buf, out_buf, in_prods, out_conses):
        for w in range(n_workers):
            in_prods[w].fill(in_buf, band(w))
        for w in range(n_workers):
            out_conses[w].drain(out_buf, band(w), wait=True)

    src_ty = np.ndarray[(num_elements,), np.dtype[src_iron_dtype]]
    dst_ty = np.ndarray[(num_elements,), np.dtype[dst_iron_dtype]]
    rt = Runtime(
        sequence,
        [
            src_ty,
            dst_ty,
            [of_in.prod() for of_in in of_ins],
            [of_out.cons() for of_out in of_outs],
        ],
    )
    return Program(arch_to_device(arch), rt, workers=workers).resolve_program()


def _worker_cap(arch: str, num_elements: int) -> int | None:
    """Most workers to fan a cast of num_elements over, or None for one per column.

    On aie2p, fastest or within noise of it inside a graph (an ADD between three casts, f16 or
    f32 <-> bf16, [768, N]): 1 worker up to N = 256, 2 at 512 and 1024, 4 at 2048 and 4096.
    Alone, a cast prefers one worker per column at every size.

    Args:
        arch: Target architecture.
        num_elements: Total number of elements to convert.

    Returns:
        The cap on aie2p, at most the column count; None (the column count) elsewhere.
    """
    if arch != "aie2p":
        return None
    workers = math.sqrt(num_elements / _AIE2P_WORKER_ELEMENTS)
    return min(arch_num_columns(arch), 2 ** max(0, round(math.log2(workers))))


def _lanes_and_tile_size(
    arch: str, num_elements: int, fifo_dtypes: tuple[np.dtype, np.dtype]
) -> tuple[int, int]:
    """Vector width and tile length for streaming num_elements through convert.cc.

    The kernel only makes aligned vector accesses (an unaligned store rewrites the 32-byte
    blocks around it, which can reach past the tile into a buffer the DMA is using), so a tile
    must be a whole number of vectors. The vector holds one 512-bit register of the wider type,
    or half of one when only that divides num_elements (a bf16 [500, 500] has 250000 elements, a
    multiple of 16 but not of 32); anything narrower runs the scalar loop (lanes 1).

    Args:
        arch: Target architecture.
        num_elements: Total number of elements to convert.
        fifo_dtypes: The streamed source and destination element types.

    Returns:
        (lanes, tile_size): the vector width in elements (1 for no vector path), and the largest
        multiple of it that divides num_elements and fits half of L1 double-buffered.
    """
    full = 512 // (8 * max(d.itemsize for d in fifo_dtypes))
    lanes = next(v for v in (full, full // 2, 1) if num_elements % v == 0)
    if lanes == 1:
        return 1, divisor_tile_size(arch, num_elements, fifo_dtypes)
    # A tile of t vectors is t elements of a lanes-wide subarray type.
    vectors = divisor_tile_size(
        arch, num_elements // lanes, [np.dtype((d, (lanes,))) for d in fifo_dtypes]
    )
    return lanes, vectors * lanes


def _create_external_function(
    src, src_iron_dtype, output_tensor, dst_iron_dtype, tile_size: int, lanes: int
) -> ExternalFunction:
    """Create the ExternalFunction for the convert core function.

    Args:
        src: Source tensor.
        src_iron_dtype: Element type the source is streamed as. Equal to ``src.dtype``
            except for f16, which IRON cannot express and which rides as ``np.int16``.
        output_tensor: Destination tensor (different dtype).
        dst_iron_dtype: Element type the destination is streamed as. Equal to
            ``output_tensor.dtype`` except for f16, which rides as ``np.int16``.
        tile_size: Number of elements per streamed tile.
        lanes: Vector width in elements, dividing tile_size; 1 for no vector path.

    Returns:
        The configured ExternalFunction.

    """
    current_dir = Path(__file__).resolve().parent
    compile_flags = [
        # The C signature follows the streamed element type, so an f16 source or destination is
        # declared i16 and reinterpreted inside the kernel.
        f"-DINPUT_DTYPE={dtype_to_str(src_iron_dtype)}",
        f"-DOUTPUT_DTYPE={dtype_to_str(dst_iron_dtype)}",
        # Tile length is fixed per kernel instance (each shape JITs its own .o), so pass it as a
        # compile-time constant: lets Peano fold the trip count and pipeline the hot loop.
        f"-DCONVERT_N={tile_size}",
        f"-DCONVERT_LANES={lanes}",
    ]
    # Select the kernel body by preprocessor (not if constexpr): the dtype macros expand to concrete
    # types, so both branches of an if constexpr would still be compiled and the unused one fails to
    # type-check. Each direction has its own bit-exact body; the same-dtype copies share one.
    if src.dtype == bfloat16 and output_tensor.dtype == np.float16:
        compile_flags.append("-DCONVERT_BF16_TO_F16=1")
    elif src.dtype == np.float16 and output_tensor.dtype == bfloat16:
        compile_flags.append("-DCONVERT_F16_TO_BF16=1")
    elif src.dtype == np.float32 and output_tensor.dtype == bfloat16:
        compile_flags.append("-DCONVERT_F32_TO_BF16=1")
    elif src.dtype == bfloat16 and output_tensor.dtype == np.float32:
        compile_flags.append("-DCONVERT_BF16_TO_F32=1")

    return ExternalFunction(
        name="ggml_hsa_convert",
        object_file_name="ggml_hsa_convert_core_function.o",
        source_file=str(current_dir / "convert.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[src_iron_dtype]],
            np.ndarray[(tile_size,), np.dtype[dst_iron_dtype]],
            np.int32,  # N
        ],
        compile_flags=compile_flags,
    )
