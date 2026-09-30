#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""IRON design for matrix multiplication (GEMM): C = A @ B."""

import argparse
from pathlib import Path

import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
from aie.iron import ExternalFunction, dtype_to_str, str_to_dtype
from aie.iron.controlflow import range_

from .gemm_c_plan import (
    SHIM_C_BD_IDS,
    core_schedule,
    make_grid,
    mem_buffers,
    mem_tasks,
    sends,
    shim_rb_chunks,
    validate,
)

# DMA channels of the hand-written C path. The A/B ObjectFifos take the others; the S2MM join
# needs n_aie_rows consecutive mem-tile channels starting at C_MEM_S2MM_CHANNEL_BASE.
C_CORE_MM2S_CHANNEL = 0
C_MEM_S2MM_CHANNEL_BASE = 2
C_MEM_MM2S_CHANNEL = 2
C_SHIM_S2MM_CHANNEL = 0

# Per-device, per-dtype (r, s, t) microkernel MAC-instruction dimensions (M, K, N of the native
# mmul shape used by mm.cc); must match the r/s/t used by the matmul_vectorized_* wrappers in
# aie2/mm.cc ("npu") and aie2p/mm.cc ("npu2").
microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


# Per-device, per-dtype (row_expand, col_expand) mmul expansion factors: how many
# mmul subtiles the vectorized microkernel consumes per loop step in the rowA and
# colB dimensions. These are the divisibility contracts the mm.cc wrappers document
# on their rowA/colB template parameters, and they differ per dtype on aie2 because
# each dtype dispatches to a different wrapper:
#   npu  bf16 -> matmul_vectorized_4x4      (rowA % 4, colB % 4)
#   npu  i8   -> matmul_vectorized_4x2_mmul (rowA % 4, colB % 2)
#   npu  i16  -> matmul_vectorized_2x2_mmul (rowA % 2, colB % 2)
#   npu2 all  -> matmul_vectorized_2x2_mmul (rowA % 2, colB % 2)
# Since rowA = m / r and colB = n / t, a tile must satisfy
# m % (row_expand * r) == 0 and n % (col_expand * t) == 0.
microkernel_expansion_map = {
    "npu": {
        "bf16": (4, 4),
        "i8": (4, 2),
        "i16": (2, 2),
    },
    "npu2": {
        "bf16": (2, 2),
        "i8": (2, 2),
        "i16": (2, 2),
    },
}


def resolve_expansion(dev, dtype_in_str):
    """Return the (row_expand, col_expand) mmul expansion for a device/dtype.

    Args:
        dev: Target device ("npu" or "npu2").
        dtype_in_str: Input dtype name (e.g. "bf16", "i8", "i16").

    Returns:
        The (row_expand, col_expand) tuple from microkernel_expansion_map.
    """
    return microkernel_expansion_map[dev][dtype_in_str]


def resolve_mac_dims(dev, dtype_in_str, emulate_bf16_mmul_with_bfp16=False):
    """Return the (r, s, t) microkernel MAC-instruction dims for a device/dtype.

    Args:
        dev: Target device ("npu" or "npu2").
        dtype_in_str: Input dtype name (e.g. "bf16", "i8", "i16").
        emulate_bf16_mmul_with_bfp16: Whether npu2 bf16 mmul is emulated via bfp16.

    Returns:
        The (r, s, t) tuple from microkernel_mac_dim_map.
    """
    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dev == "npu2" and dtype_in_str == "bf16":
        return mac_dims[emulate_bf16_mmul_with_bfp16]
    return mac_dims


# Per-core L1 data-memory budget (bytes) for the double-buffered A/B/C
# object-FIFO tiles. AIE2/AIE2P cores have 64 KiB of local data memory; we cap
# the working set well below that to leave room for the stack (see the
# stack_size note on the compute cores below) and scalar locals.
L1_TILE_BUDGET_BYTES = 48 * 1024

# Inclusive upper bound of the aiex.npu.dma_memcpy_nd stride range ([1, 2^20]).
# The column-major B/C shim transfers this design emits have outer strides of
# n * n_aie_cols * K and M * n * n_aie_cols respectively; these grow with the
# N tile and cross the limit for large M/K, so the tile search must exclude
# them (otherwise aiecc fails with "Stride N exceeds the [1:1048576] range").
#
# Both of those strides belong to the same outermost dimension, whose size is
# N // (n * n_aie_cols) -- the number of column groups the herd sweeps. When
# that size is 1 the dimension is never stepped, so the stride is dead and the
# bound does not apply. That is the common case for the im2col GEMMs behind
# CONV_2D, where N is one column group wide but M is the whole batch*OH*OW
# extent and no tile could otherwise keep M * n * n_aie_cols in range.
DMA_MAX_STRIDE = 1 << 20


def select_gemm_tile(
    dev,
    M,
    N,
    K,
    dtype_in,
    dtype_out,
    r,
    s,
    t,
    row_expand=2,
    col_expand=2,
    n_aie_rows=4,
    fifo_depth=2,
    max_tile=256,
):
    """Pick the largest valid per-core (m, k, n) GEMM tile for a problem size.

    The whole-array design distributes the MxKxN GEMM across an
    n_aie_rows x n_aie_cols herd; each core computes (m, n) output tiles while
    reducing over k. Larger per-core tiles amortize DMA / object-FIFO /
    synchronization overhead and improve MAC-array utilization, so we maximize
    the per-core tile volume (m * k * n) subject to:

      * microkernel divisibility for the wrapper's mmul expansion:
        m % (row_expand*r) == 0, k % s == 0, n % (col_expand*t) == 0;
      * array tiling: M % (m * n_aie_rows) == 0, K % k == 0,
        N % (n * n_aie_cols) == 0;
      * even work distribution: (M // m) * (N // n) is a multiple of the number
        of cores, so every core gets the same number of tiles;
      * the per-core L1 budget for the double-buffered A/B/C tiles;
      * the shim-DMA buffer-descriptor stride range, which the column-major B/C
        transfers cross for large M/K (a hard aiecc failure, not a miscompute) --
        but only when the dimension those strides step actually has size > 1.

    Ties are broken toward a larger output tile (m * n, which amortizes the C
    zero-init and drain), then a larger k.

    Args:
        dev: Target device ("npu" or "npu2").
        M: Full GEMM problem dimension M.
        N: Full GEMM problem dimension N.
        K: Full GEMM problem dimension K.
        dtype_in: NumPy input dtype (for element size).
        dtype_out: NumPy output dtype (for element size).
        r: Microkernel MAC dim r for the input dtype.
        s: Microkernel MAC dim s for the input dtype.
        t: Microkernel MAC dim t for the input dtype.
        row_expand: mmul subtiles per rowA loop step (see resolve_expansion).
        col_expand: mmul subtiles per colB loop step (see resolve_expansion).
        n_aie_rows: AIE array rows (4 on both npu and npu2).
        fifo_depth: Object-FIFO depth (double buffering).
        max_tile: Upper bound on any single tile dimension.

    Returns:
        A (m, k, n) tuple of per-core tile dimensions.

    Raises:
        ValueError: If no tile satisfies every constraint, naming the ones that
            eliminated even the smallest candidate.
    """
    n_aie_cols = 8 if dev == "npu2" else 4
    n_cores = n_aie_rows * n_aie_cols

    size_in = np.dtype(dtype_in).itemsize
    size_out = np.dtype(dtype_out).itemsize

    # Microkernel-granular step sizes for this wrapper's mmul expansion. Using a
    # blanket 2x2 here would under-constrain the aie2 bf16/i8 wrappers, which
    # consume 4 mmul subtiles per rowA (and, for bf16, per colB) loop step.
    gm, gk, gn = row_expand * r, s, col_expand * t

    def working_set(m, k, n):
        return fifo_depth * (size_in * m * k + size_in * k * n + size_out * m * n)

    def valid(m, k, n):
        if M % (m * n_aie_rows) or K % k or N % (n * n_aie_cols):
            return False
        if ((M // m) * (N // n)) % n_cores:
            return False
        # Column-major B/C shim-DMA outer strides must fit the BD stride range,
        # unless the dimension they step has size 1 and never applies them.
        if N // (n * n_aie_cols) > 1 and (
            M * n * n_aie_cols > DMA_MAX_STRIDE or n * n_aie_cols * K > DMA_MAX_STRIDE
        ):
            return False
        return working_set(m, k, n) <= L1_TILE_BUDGET_BYTES

    best_key = None
    best_tile = None
    for m in range(gm, min(M, max_tile) + 1, gm):
        for k in range(gk, min(K, max_tile) + 1, gk):
            for n in range(gn, min(N, max_tile) + 1, gn):
                if not valid(m, k, n):
                    continue
                key = (m * k * n, m * n, k)
                if best_key is None or key > best_key:
                    best_key = key
                    best_tile = (m, k, n)

    if best_tile is not None:
        return best_tile

    # No candidate satisfied every constraint. Raise rather than returning the
    # granular minimum: my_matmul only re-checks microkernel granularity and array
    # tiling, not the L1 budget or the shim-DMA stride, so a minimum tile that
    # happens to pass those two sails through and fails deep inside aiecc ("Stride
    # N exceeds the [1:1048576] range") with nothing left to explain why. Callers
    # treat this as "kernel not supported" and fall back to the CPU.
    reasons = []
    if gm > M or gk > K or gn > N:
        reasons.append(
            f"problem is smaller than the microkernel granularity: "
            f"(M,N,K)=({M},{N},{K}) vs minimum (m,k,n)=({gm},{gk},{gn})"
        )
    if M % (gm * n_aie_rows) or K % gk or N % (gn * n_aie_cols):
        reasons.append(
            f"not tileable across the {n_aie_rows}x{n_aie_cols} herd even at the "
            f"minimum tile ({gm},{gk},{gn}): need M%{gm * n_aie_rows}==0, "
            f"K%{gk}==0, N%{gn * n_aie_cols}==0"
        )
    if N // (gn * n_aie_cols) > 1 and (
        M * gn * n_aie_cols > DMA_MAX_STRIDE or gn * n_aie_cols * K > DMA_MAX_STRIDE
    ):
        reasons.append(
            f"column-major shim-DMA stride exceeds {DMA_MAX_STRIDE} even at the "
            f"minimum n={gn}: M*n*cols={M * gn * n_aie_cols}, "
            f"n*cols*K={gn * n_aie_cols * K}"
        )
    if working_set(gm, gk, gn) > L1_TILE_BUDGET_BYTES:
        reasons.append(
            f"minimum-tile working set {working_set(gm, gk, gn)} B exceeds the "
            f"per-core L1 budget {L1_TILE_BUDGET_BYTES} B"
        )
    if not reasons:
        reasons.append(
            "no (m,k,n) satisfied every constraint simultaneously, though the "
            "minimum tile satisfies each one individually"
        )
    msg = (
        f"No valid GEMM tile for {dev} MxNxK={M}x{N}x{K} "
        f"({np.dtype(dtype_in).name}->{np.dtype(dtype_out).name}): "
        + "; ".join(reasons)
    )
    raise ValueError(msg)


def main():
    """CLI entry point: parse arguments and print the generated GEMM MLIR."""
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Whole Array)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=512)
    argparser.add_argument("-N", type=int, default=512)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    # Whether to use the scalar kernel; this is low, but can be useful for debugging smaller sizes
    argparser.add_argument("--scalar", type=bool, choices=[0, 1], default=0)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--generate-taps",
        action="store_true",
        help="Generate TensorAccessPatterns, a Python object to represent each data transfer"
        "of the input/output matrices. These objects can be used for visualization.",
    )
    args = argparser.parse_args()
    with mlir_mod_ctx():
        maybe_taps = my_matmul(
            args.dev,
            args.M,
            args.K,
            args.N,
            args.m,
            args.k,
            args.n,
            args.n_aie_cols,
            args.dtype_in,
            args.dtype_out,
            args.b_col_maj,
            args.c_col_maj,
            args.scalar,
            args.emulate_bf16_mmul_with_bfp16,
            args.trace_size,
            f"matmul_{dtype_to_str(args.dtype_in)}_{dtype_to_str(args.dtype_out)}",
            f"zero_{dtype_to_str(args.dtype_out)}",
            f"mm_{args.m}x{args.k}x{args.n}.o",
            args.generate_taps,
        )
        # print(ctx.module.operation.verify())
        print(ctx.module)

    if args.generate_taps:
        return maybe_taps
    return None


def ceildiv(a, b):
    """Return the ceiling of integer division a/b.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        The smallest integer >= a / b.
    """
    return (a + b - 1) // b


def _core_c_dma(core_tile, bufs, locks, length):
    """Static core MM2S chain: send each finished ping/pong C buffer to the mem tile."""

    @mem(core_tile)
    def _(block):
        dma_start(
            DMAChannelDir.MM2S, C_CORE_MM2S_CHANNEL, dest=block[1], chain=block[3]
        )
        for b in range(2):
            with block[1 + b]:
                use_lock(locks[b][1], LockAction.AcquireGreaterEqual, value=1)
                dma_bd(bufs[b], transfer_len=length)
                use_lock(locks[b][0], LockAction.Release, value=1)
                next_bd(block[2 - b])
        with block[3]:
            EndOp()


def _mem_c_join(mem_tile, bufs, locks, m, n, r, t, n_aie_rows):
    """Static mem-tile S2MM join of the n_aie_rows core blocks of one object.

    matmul_vectorized_* with C_COL_MAJ leaves element (i = z*r + ri, j = jb*t + tj) at
    jb*t*m + z*r*t + tj*r + ri. This writes it as plain column-major m x n (element (i, j) at
    j*m + i), core `row` at offset row*m*n, so every read the runtime sequence issues is a <=3-D
    slice (gemm_c_plan.mem_read_bds). Runtime-task mem-tile BDs have only 3 real dims; this
    static BD may use 4. Each channel cycles over all of `bufs` (gemm_c_plan.mem_buffers).
    """
    nbuf = len(bufs)

    @memtile_dma(mem_tile)
    def _(block):
        nb = 1
        for row in range(n_aie_rows):
            channel = C_MEM_S2MM_CHANNEL_BASE + row
            if row == 0:
                dma_start(
                    DMAChannelDir.S2MM, channel, dest=block[nb], chain=block[nb + nbuf]
                )
            else:
                with block[nb - 1]:
                    dma_start(
                        DMAChannelDir.S2MM,
                        channel,
                        dest=block[nb],
                        chain=block[nb + nbuf],
                    )
            for b in range(nbuf):
                with block[nb + b]:
                    use_lock(locks[b][0], LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(
                        bufs[b],
                        offset=row * m * n,
                        transfer_len=m * n,
                        sizes=[n // t, m // r, t, r],
                        strides=[t * m, r, m, 1],
                    )
                    use_lock(locks[b][1], LockAction.Release, value=1)
                    next_bd(block[nb + (b + 1) % nbuf])
            nb += nbuf + 1
        with block[nb - 1]:
            EndOp()


def _start_mem_c_task(mem_tile, bufs, locks, token, task, n_aie_rows):
    """Queue one mem-tile MM2S task (gemm_c_plan.MemTask) reading valid C to the shim.

    A runtime-task BD takes 0 or 2 lock ops. An object read by several BDs passes a token lock
    along the chain: first acquire cons/release token, then acquire token/release prod.
    """
    task_op = dma_configure_task(
        mem_tile, DMAChannelDir.MM2S, C_MEM_MM2S_CHANNEL, repeat_count=task.repeat - 1
    )
    total = sum(len(obj_bds) for obj_bds in task.bds)
    with bds(task_op) as blk:
        i = 0
        for buf_idx, obj_bds in zip(task.buffers, task.bds, strict=True):
            prod_lock, cons_lock = locks[buf_idx]
            for j, bd in enumerate(obj_bds):
                first, last = j == 0, j == len(obj_bds) - 1
                with blk[i]:
                    use_lock(
                        cons_lock if first else token,
                        LockAction.AcquireGreaterEqual,
                        value=n_aie_rows if first else 1,
                    )
                    layout = (
                        {}
                        if len(bd.sizes) == 1
                        else {"sizes": list(bd.sizes), "strides": list(bd.strides)}
                    )
                    dma_bd(
                        bufs[buf_idx],
                        offset=bd.offset,
                        transfer_len=bd.length,
                        **layout,
                    )
                    use_lock(
                        prod_lock if last else token,
                        LockAction.Release,
                        value=n_aie_rows if last else 1,
                    )
                    if i == total - 1:
                        EndOp()
                    else:
                        next_bd(blk[i + 1])
                i += 1
    dma_start_task(task_op)


def _start_shim_c_task(shim_tile, C, chunk, bd_ids):
    """Start one shim S2MM task writing a chunk of gemm_c_plan.ShimBd into the dense C.

    A BD's iteration dimension advances once per execution, so an iterated BD (alone in its
    chunk, see gemm_c_plan.ShimBd) is run `iterations` times through the task's repeat count.
    """
    repeat = chunk[0].iterations
    assert repeat == 1 or len(chunk) == 1, (
        "an iterated shim BD must be alone in its task"
    )
    task_op = dma_configure_task(
        shim_tile,
        DMAChannelDir.S2MM,
        C_SHIM_S2MM_CHANNEL,
        repeat_count=repeat - 1,
        issue_token=True,
    )
    with bds(task_op) as blk:
        for i, (b, bd_id) in enumerate(zip(chunk, bd_ids[: len(chunk)], strict=True)):
            with blk[i]:
                dma_bd(
                    C,
                    offset=b.offset,
                    sizes=[b.iterations, *b.sizes],
                    strides=[b.iteration_stride, *b.strides],
                    transfer_len=b.sizes[0] * b.sizes[1] * b.sizes[2],
                    bd_id=bd_id,
                )
                if i == len(chunk) - 1:
                    EndOp()
                else:
                    next_bd(blk[i + 1])
    dma_start_task(task_op)
    return task_op


# A is tiled into (m, k) blocks broadcast across columns and distributed across rows; B into
# (k, n) blocks broadcast across rows and distributed across columns. Each core accumulates C
# tiles over the K dimension.
def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    use_scalar,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    zero_fn,
    matmul_fn,
    object_file,
    generate_taps=False,
    M_dense=None,
    N_dense=None,
    dtype_acc_str=None,
    narrow_fn=None,
):
    """Generate MLIR for tiled GEMM across an AIE array (C = A @ B).

    Args:
        dev: Target device ("npu" or "npu2").
        M: Number of rows in A / C.
        K: Number of columns in A / rows in B.
        N: Number of columns in B / C.
        m: Per-core M tile size.
        k: Per-core K tile size.
        n: Per-core N tile size.
        n_aie_cols: Number of AIE columns to use.
        dtype_in_str: Input dtype name (e.g. "bf16", "i8", "i16").
        dtype_out_str: Output dtype name (e.g. "bf16", "i8", "i16", "f32", "i32").
        b_col_maj: Whether B is stored column-major.
        c_col_maj: Whether C is stored column-major.
        use_scalar: Whether to use the scalar (non-vectorized) kernel.
        emulate_bf16_mmul_with_bfp16: Whether npu2 bf16 mmul is emulated via bfp16.
        trace_size: Trace buffer size (0 to disable tracing).
        zero_fn: Name of the external zero-init kernel function.
        matmul_fn: Name of the external matmul kernel function.
        object_file: Path to the compiled kernel object file to link.
        generate_taps: Whether to also return TensorAccessSequences for A/B/C.
        M_dense: Rows of the dense C written (defaults to M).
        N_dense: Columns of the dense C written (defaults to N).
        dtype_acc_str: Accumulator dtype name (defaults to dtype_out_str).
        narrow_fn: Name of the acc->out narrowing core function, required when the
            accumulator and output dtypes differ.

    Returns:
        A tuple of (A, B, C) TensorAccessSequences if generate_taps, else None.
    """
    n_aie_rows = 4

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    if np.issubdtype(dtype_in, np.integer) != np.issubdtype(dtype_out, np.integer):
        msg = f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
        raise ValueError(msg)
    if np.dtype(dtype_out).itemsize < np.dtype(dtype_in).itemsize:
        msg = f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"
        raise ValueError(msg)

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    # Resolved through the same helper select_gemm_tile uses, so the tile selector
    # and this design generator can never disagree on the microkernel shape.
    r, s, t = resolve_mac_dims(dev, dtype_in_str, emulate_bf16_mmul_with_bfp16)

    # npu is a 4 row x 4 col array
    if dev == "npu" and n_aie_cols > 4:
        msg = "Invalid configuration: NPU (Phoenix/Hawk) has 4 columns"
        raise AssertionError(msg)
    # npu2 is a 4 row x 8 col array
    if dev == "npu2" and n_aie_cols > 8:
        msg = "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        raise AssertionError(msg)

    # Input matrix A:
    # Conceptually, we divide input A into (m * n_rows, k)-sized blocks. These
    # blocks are _broadcast_ across AIE core columns, then _distributed_ across
    # rows, s.t. each of the n_rows compute cores in a column receives a
    # contiguous (m, k)-sized block of A.
    assert M % (m * n_aie_rows) == 0, (
        """A must be tileable into (m * n_aie_rows, k)-sized blocks"""
    )

    # Both A and B are tiled in the K dimension into size k.
    assert K % k == 0

    # Input matrix B:
    # Conceptually, we do the same as with A, but instead of broadcasting
    # across columns we broadcast across rows and distribute across columns.
    assert N % (n * n_aie_cols) == 0, (
        """B must be tileable into (k, n * n_aie_cols)-sized blocks"""
    )

    if not use_scalar:
        assert m % r == 0
        assert k % s == 0
        assert n % t == 0

    # If you get errors during CDO generation due to running out of program
    # memory, it may be because too much code is generated due to ObjectFIFO
    # loop unrollings. Reducing the depth to 1 here will work around that at
    # a big performance cost.
    fifo_depth = 2

    # When using more AIE columns than n_aie_rows (4) (applicable to NPU2),
    # restrict the number of shim/mem tiles to n_aie_rows,
    # since we have only n_aie_rows row tiles for matrix A
    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    # When using n_aie_rows (4) or less AIE columns (both NPU and NPU2),
    # the number of shim/mem tiles are equal to n_aie_cols.
    # We use the distribute pattern in object FIFO (see linking for A below),
    # since we have n_aie_rows (4) row tiles for matrix A
    else:
        n_shim_mem_A = n_aie_cols

    # Integer division when n_aie_cols < 4, otherwise set to 1
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    if dev == "npu":
        if n_aie_cols == 1:
            dev_ty = AIEDevice.npu1_1col
        elif n_aie_cols == 2:
            dev_ty = AIEDevice.npu1_2col
        elif n_aie_cols == 4:
            dev_ty = AIEDevice.npu1
    else:
        dev_ty = AIEDevice.npu2

    # These will hold TensorAccessPattern objects that represent the runtime
    # npu_dma_memcpy_nd operations of this design. They are only used if generate_taps is true
    A_taps = []
    B_taps = []
    C_taps = []

    if not c_col_maj or use_scalar:
        msg = (
            "the hand-written C path supports the vectorized column-major C kernel only"
        )
        raise AssertionError(msg)
    dtype_acc = str_to_dtype(dtype_acc_str) if dtype_acc_str else dtype_out
    narrowing = np.dtype(dtype_acc) != np.dtype(dtype_out)
    if narrowing and narrow_fn is None:
        msg = "a narrowing C path needs narrow_fn"
        raise ValueError(msg)
    grid = make_grid(
        M if M_dense is None else M_dense,
        N if N_dense is None else N_dense,
        M,
        N,
        m,
        n,
        n_aie_rows,
        n_aie_cols,
    )
    validate(grid)

    @device(dev_ty)
    def device_body():
        A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
        B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
        C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
        A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
        B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
        acc_l1_ty = np.ndarray[(m, n), np.dtype[dtype_acc]]
        send_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

        # AIE Core Function declarations
        zero = external_func(zero_fn, inputs=[acc_l1_ty], link_with=object_file)
        matmul = external_func(
            matmul_fn, inputs=[A_l1_ty, B_l1_ty, acc_l1_ty], link_with=object_file
        )
        narrow = (
            external_func(
                narrow_fn, inputs=[acc_l1_ty, send_l1_ty], link_with=object_file
            )
            if narrowing
            else None
        )

        # Tile declarations as tile[row][col]
        tiles = [[tile(col, row) for col in range(n_aie_cols)] for row in range(6)]
        shim_tiles = tiles[0]
        mem_tiles = tiles[1]
        core_tiles = tiles[2:]

        # AIE-array data movement with object fifos
        A_l3l2_fifos = [None] * n_shim_mem_A
        A_l2l1_fifos = [None] * n_aie_rows

        B_l3l2_fifos = [None] * n_aie_cols
        B_l2l1_fifos = [None] * n_aie_cols

        # Input A
        # L3 -> L2 data movement
        for i in range(n_shim_mem_A):
            A_l3l2_fifos[i] = object_fifo(
                f"A_L3L2_{i}",
                (
                    shim_tiles[2 * i]
                    if n_aie_cols == 8
                    else shim_tiles[i]  # alternate columns in full 4x8 NPU2 case
                ),
                mem_tiles[2 * i] if n_aie_cols == 8 else mem_tiles[i],
                fifo_depth,
                A_l2_ty,
            )

        # L2 -> L1 data movement
        for row in range(n_aie_rows):
            A_l2l1_fifos[row] = object_fifo(
                f"A_L2L1_{row}",
                (
                    mem_tiles[2 * row]
                    if n_aie_cols == 8
                    else mem_tiles[row // n_A_tiles_per_shim]
                ),
                core_tiles[row][0:n_aie_cols],  # broadcast along one row
                fifo_depth,
                A_l1_ty,
                (
                    [
                        (m // r, r * k),
                        (k // s, s),
                        (r, k),
                        (s, 1),
                    ]
                    if not use_scalar
                    else []
                ),
            )

        # A_l3_l2 and A_l2_l1 object FIFO linking
        for i in range(n_shim_mem_A):
            # If n_shim_mem_A == n_rows, n_A_tiles_per_shim is 1 and
            # this simply links a_l3l2_fifos[i] to a_l2l1_fifos[i] directly,
            # If n_shim_mem_A < n_rows, each column receives multiple rows of
            # tiles; distribute it along rows of AIE cores.
            start_row = i * n_A_tiles_per_shim
            stop_row = start_row + n_A_tiles_per_shim
            if stop_row - start_row > 1:
                of_offsets = [m * k * j for j in range(stop_row - start_row)]
            else:
                of_offsets = []
            object_fifo_link(
                A_l3l2_fifos[i],
                [A_l2l1_fifos[j] for j in range(start_row, stop_row)],
                [],
                of_offsets,
            )

        # Input B
        for col in range(n_aie_cols):
            # L3 -> L2 data movement
            B_l3l2_fifos[col] = object_fifo(
                f"B_L3L2_{col}",
                shim_tiles[col],
                mem_tiles[col],
                fifo_depth,
                B_l2_ty,
            )
            # L2 -> L1 data movement
            B_l2l1_fifos[col] = object_fifo(
                f"B_L2L1_{col}",
                mem_tiles[col],
                [
                    core_tiles[j][col] for j in range(n_aie_rows)
                ],  # broadcast along one column
                fifo_depth,
                B_l1_ty,
                (
                    (
                        [
                            (k // s, s * n),
                            (n // t, t),
                            (s, n),
                            (t, 1),
                        ]
                        if not b_col_maj
                        else [
                            (n // t, t * k),
                            (k // s, s),
                            (t, k),
                            (s, 1),
                        ]
                    )
                    if not use_scalar
                    else []
                ),
            )
            # B_l3_l2 and B_l2_l1 object FIFO linking
            object_fifo_link(B_l3l2_fifos[col], B_l2l1_fifos[col])

        # Output C, written by hand rather than with ObjectFifos, so the runtime sequence can read
        # only the valid part of each block (see gemm_c_plan.py). Core -> mem tile is a static
        # DMA chain per core; mem tile -> shim is issued from the runtime sequence.
        c_acc, c_send, c_core_lk = {}, {}, {}
        c_mem_buf, c_mem_lk, c_mem_tok = {}, {}, {}
        for col in range(n_aie_cols):
            mt = mem_tiles[col]
            # One buffer when the column fills an odd number per dispatch, so the static join
            # and the runtime sequence stay in step across dispatches (gemm_c_plan.mem_buffers).
            nbuf = mem_buffers(grid, col)
            c_mem_buf[col] = [
                buffer(mt, C_l2_ty, name=f"C_mem_{col}_{b}") for b in range(nbuf)
            ]
            c_mem_lk[col] = [
                (
                    lock(mt, init=n_aie_rows, sym_name=f"C_mem_prod_{col}_{b}"),
                    lock(mt, init=0, sym_name=f"C_mem_cons_{col}_{b}"),
                )
                for b in range(nbuf)
            ]
            c_mem_tok[col] = lock(mt, init=0, sym_name=f"C_mem_tok_{col}")
            flow(
                mt,
                WireBundle.DMA,
                C_MEM_MM2S_CHANNEL,
                shim_tiles[col],
                WireBundle.DMA,
                C_SHIM_S2MM_CHANNEL,
            )
            for row in range(n_aie_rows):
                ct = core_tiles[row][col]
                c_send[row, col] = [
                    buffer(ct, send_l1_ty, name=f"C_send_{col}_{row}_{b}")
                    for b in range(2)
                ]
                # Narrowing keeps one f32 accumulator and double-buffers the narrowed copy; the
                # L1 cost equals double-buffering the accumulator (select_gemm_tile budgets it).
                c_acc[row, col] = (
                    [buffer(ct, acc_l1_ty, name=f"C_acc_{col}_{row}")]
                    if narrowing
                    else c_send[row, col]
                )
                c_core_lk[row, col] = [
                    (
                        lock(ct, init=1, sym_name=f"C_core_prod_{col}_{row}_{b}"),
                        lock(ct, init=0, sym_name=f"C_core_cons_{col}_{row}_{b}"),
                    )
                    for b in range(2)
                ]
                flow(
                    ct,
                    WireBundle.DMA,
                    C_CORE_MM2S_CHANNEL,
                    mt,
                    WireBundle.DMA,
                    C_MEM_S2MM_CHANNEL_BASE + row,
                )
                _core_c_dma(ct, c_send[row, col], c_core_lk[row, col], m * n)
            _mem_c_join(mt, c_mem_buf[col], c_mem_lk[col], m, n, r, t, n_aie_rows)

        # Set up compute tiles. Each core follows its static schedule (gemm_c_plan.core_schedule).
        # Ping/pong parity must survive across dispatches: when a dispatch sends an odd number of
        # blocks, the forever-loop body covers two dispatches, the second starting on the other
        # buffer.
        def core_program(row, col):
            acc, send, lks = c_acc[row, col], c_send[row, col], c_core_lk[row, col]
            sched = core_schedule(grid, row, col)

            def k_loop(acc_buf):
                for _ in range_(K // k):
                    elem_in_a = A_l2l1_fifos[row].acquire(ObjectFifoPort.Consume, 1)
                    elem_in_b = B_l2l1_fifos[col].acquire(ObjectFifoPort.Consume, 1)
                    if acc_buf is not None:
                        matmul(elem_in_a, elem_in_b, acc_buf)
                    A_l2l1_fifos[row].release(ObjectFifoPort.Consume, 1)
                    B_l2l1_fifos[col].release(ObjectFifoPort.Consume, 1)

            def tile_(kind, b):
                if kind == "consume":
                    k_loop(None)
                    return
                use_lock(lks[b][0], LockAction.AcquireGreaterEqual, value=1)
                if kind == "compute":
                    acc_buf = acc[0] if narrowing else acc[b]
                    zero(acc_buf)
                    k_loop(acc_buf)
                    if narrowing:
                        narrow(acc_buf, send[b])
                else:  # "send": all rows are padding; the mem tile never reads this block
                    k_loop(None)
                use_lock(lks[b][1], LockAction.Release, value=1)

            def repeat(count, body):
                if count == 1:  # range_(1) trips issue #1547
                    body()
                elif count > 1:
                    for _ in range_(count):
                        body()

            def emit_runs(runs, p):
                for kind, count in runs:
                    if kind == "consume":
                        repeat(count, lambda kind=kind: tile_(kind, 0))
                        continue
                    repeat(
                        count // 2,
                        lambda kind=kind, p=p: (tile_(kind, p), tile_(kind, 1 - p)),
                    )
                    if count % 2:
                        tile_(kind, p)
                        p ^= 1
                return p

            def emit_dispatch(p):
                for rep, runs in sched:
                    if rep > 1 and sends(runs) % 2:
                        repeat(
                            rep // 2,
                            lambda runs=runs, p=p: emit_runs(runs, emit_runs(runs, p)),
                        )
                        if rep % 2:
                            p = emit_runs(runs, p)
                    else:
                        repeat(rep, lambda runs=runs, p=p: emit_runs(runs, p))
                        p ^= (sends(runs) * rep) % 2
                return p

            # The stack size choice is a workaround explained here:
            # https://github.com/Xilinx/mlir-aie/pull/2391#issuecomment-2967432485
            # In summary, the Peano compiler uses a stack size greater than the default one used by this kernel
            # (default is 0x400, chess' stack size is smaller). This is only necessary for bf16 through bfp16 emulation on npu2.
            # Exceding the stack size leads to wrong results from the kernel, but no error is triggered.
            # Stack usage can be checked as explained here:
            # https://github.com/Xilinx/llvm-aie/issues/487#issuecomment-2969438585
            @core(core_tiles[row][col], stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    if emit_dispatch(0):
                        emit_dispatch(1)

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                core_program(row, col)

        # To/from AIE-array data movement
        @runtime_sequence(
            np.ndarray[(M * K,), np.dtype[dtype_in]],
            np.ndarray[(K * N,), np.dtype[dtype_in]],
            np.ndarray[(grid.M * grid.N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            # Mem-tile MM2S for the whole dispatch, queued up front (gemm_c_plan.mem_tasks keeps
            # each channel within its task-queue depth). The tasks are lock-gated, so they run as
            # the cores deliver.
            for col in range(n_aie_cols):
                for task in mem_tasks(grid, col):
                    _start_mem_c_task(
                        mem_tiles[col],
                        c_mem_buf[col],
                        c_mem_lk[col],
                        c_mem_tok[col],
                        task,
                        n_aie_rows,
                    )
            # Per column, oldest first: started shim C tasks, or -- for a row block in which the
            # column writes no C -- the A/B fifos whose transfers were issued with a token.
            outstanding = [[] for _ in range(n_aie_cols)]

            def await_all(col, keep=0):
                # Await all but the newest `keep` entries.
                n_wait = len(outstanding[col]) - keep
                for entry in outstanding[col][:n_wait]:
                    if isinstance(entry, list):
                        dma_wait(*entry)
                    else:
                        dma_await_task(entry)
                del outstanding[col][:n_wait]

            # We are limited in the number of BDs. After synchronizing, we can reuse BDs.
            # We only transfer 4 rows of tiles at once before starting a new transfer block.
            # tb = transfer block; block of transfers before sync call
            tb_max_n_rows = 4 if not c_col_maj else 2
            for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    M // m // n_aie_rows // tb_max_n_rows
                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    bd_id_base = 8 * pingpong
                    tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                    )
                    if tb_n_rows <= 0:
                        # for small input sizes, we may not even need a "pong" iteration
                        break
                    assert tb_n_rows == 1
                    rest_chunks = {}
                    for col in range(n_aie_cols):
                        # C Output Transfer:
                        # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                        # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                        # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                        # Each shim will start at a different column offset, transferring interleaved
                        # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                        # may transfer the blocks marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # M   |0011    0011    |
                        #     |                |
                        #     |                |
                        #     |                |
                        #     |                |
                        #      ----------------
                        # C output for this row block: shim writes of the valid data only.
                        # tb_n_rows is 1 with c_col_maj, so row_base is the row block.
                        chunks = shim_rb_chunks(grid, col, row_base)
                        if chunks:
                            outstanding[col].append(
                                _start_shim_c_task(
                                    shim_tiles[col],
                                    C,
                                    chunks[0],
                                    SHIM_C_BD_IDS[pingpong],
                                )
                            )
                        rest_chunks[col] = chunks[1:]
                        # A column that writes no C in this row block has no C task whose
                        # completion proves its A/B transfers done before their BD ids are
                        # reused, so those transfers carry a token and are awaited directly.
                        ab_token = True if not chunks else None
                        if not chunks:
                            outstanding[col].append(
                                [A_l3l2_fifos[col], B_l3l2_fifos[col]]
                                if col < n_aie_rows
                                else [B_l3l2_fifos[col]]
                            )
                        if generate_taps:
                            C_taps.extend(
                                TensorAccessPattern(
                                    (grid.M * grid.N,),
                                    offset=b.offset,
                                    sizes=[b.iterations, *b.sizes],
                                    strides=[b.iteration_stride, *b.strides],
                                )
                                for chunk in chunks
                                for b in chunk
                            )

                        for tile_row in range(tb_n_rows):
                            # A input transfer:
                            #
                            # The smallest transfer unit is a (m*n_A_tiles_per_shim)-sized sub-tile of the input matrix.
                            # Transfer one such tile for every column, contiguously.
                            # Repeat this transfer with identical tiles a total of (N//n//n_aie_cols) times.
                            # Each shim transfers the tiles for separate rows. For example, shim 0 may transfer the
                            # tiles marked 0 below, and shim 1 may transfer the tiles marked 1.
                            #             K
                            #      ----------------
                            #     |0000000000000000|    (repeated N//n//n_aie_cols times)
                            #     |0000000000000000|
                            #     |1111111111111111|
                            # M   |1111111111111111|
                            #     |                |
                            #     |                |
                            #     |                |
                            #     |                |
                            #      ----------------
                            A_block_offset = (
                                (row_base + tile_row) * n_aie_rows * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                col * n_A_tiles_per_shim * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset
                            A_sizes = [
                                N // n // n_aie_cols,
                                K // k,
                                m * n_A_tiles_per_shim,
                                k,
                            ]
                            A_strides = [0, k, K, 1]

                            # always equal to n_aie_rows since we have n_aie_rows row tiles for matrix A
                            if col < n_aie_rows:
                                npu_dma_memcpy_nd(
                                    metadata=A_l3l2_fifos[col],
                                    bd_id=bd_id_base + 2 * tile_row + 1,
                                    mem=A,
                                    offsets=[0, 0, 0, A_offset],
                                    sizes=A_sizes,
                                    strides=A_strides,
                                    issue_token=ab_token,
                                )
                            # # Use the calculated sizes/strides/offsets to record the data movement
                            # # caused by the above call to npu_dma_memcpy_nd.
                            # # This line does not change MLIR output at all.
                            if generate_taps:
                                A_taps.append(
                                    TensorAccessPattern(
                                        (M, K),
                                        offset=A_offset,
                                        sizes=A_sizes,
                                        strides=A_strides,
                                    )
                                )

                            # B input transfer:
                            # Transfer the first a (n)-wide block of columns of B,
                            # Then transfer the (n_aie_columns)-th such block, and so on.
                            # Each shim will start at a different column offset.
                            # For example, shim 0 may transfer the tiles marked 0 below,
                            # and shim 1 may transfer the tiles marked 1.
                            #
                            #             N
                            #      ----------------
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            # K   |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #      ----------------
                            B_col_offset = col * n if not b_col_maj else col * n * K
                            if not b_col_maj:
                                B_sizes = [N // n // n_aie_cols, K // k, k, n]
                                B_strides = [n * n_aie_cols, k * N, N, 1]
                            else:
                                B_sizes = [N // n // n_aie_cols, K // k, n, k]
                                B_strides = [n * n_aie_cols * K, k, K, 1]

                            npu_dma_memcpy_nd(
                                metadata=B_l3l2_fifos[col],
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset],
                                sizes=B_sizes,
                                strides=B_strides,
                                issue_token=ab_token,
                            )
                            # # Use the calculated sizes/strides/offsets to record the data movement
                            # # caused by the above call to npu_dma_memcpy_nd.
                            # # This line does not change MLIR output at all.
                            if generate_taps:
                                B_taps.append(
                                    TensorAccessPattern(
                                        (K, N),
                                        offset=B_col_offset,
                                        sizes=B_sizes,
                                        strides=B_strides,
                                    )
                                )
                    # Row blocks with more shim BDs than one half's ids: await the previous chunk
                    # (it completes, because this row block's A and B are already issued) and
                    # reuse the ids.
                    for col in range(n_aie_cols):
                        for chunk in rest_chunks[col]:
                            await_all(col)
                            outstanding[col].append(
                                _start_shim_c_task(
                                    shim_tiles[col], C, chunk, SHIM_C_BD_IDS[pingpong]
                                )
                            )
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        # Keep this row block's C in flight while the next one's A and B are
                        # issued; awaiting it too would drain the pipeline every row block. Its
                        # BD ids are not reused before the next wait, which awaits it.
                        for col in range(n_aie_cols):
                            await_all(col, keep=1)
            for col in range(n_aie_cols):
                await_all(col)

    if generate_taps:
        # If generate_taps is true, return a representation of tensor tiles
        # representing all the npu_dma_memcpy_nd runtime sequence operations per input/ouput tensor.
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return None


if __name__ == "__main__":
    main()


def create_mat_mul_external_functions(
    arch: str,
    input_tensors: list,
    output_tensor,
):
    """Create the zero-init and matmul ExternalFunctions for GEMM.

    Args:
        arch: Target architecture ("aie2" or "aie2p").
        input_tensors: List of input tensors [A, B].
        output_tensor: Output tensor C.

    Returns:
        (m, n, k, use_scalar, num_cols, zero_fn, matmul_fn, narrow_fn, dtype_acc).

    Raises:
        ValueError: If the architecture is unsupported.
    """
    use_scalar = False
    scalar_suffix = "_scalar" if use_scalar else ""

    # Pick the largest valid per-core tile for this problem size instead of a fixed
    # small tile: the previous fixed tiles (8x8x8 on aie2, 16x16x16 on aie2p) left
    # the herd DMA/sync-bound, and on aie2 the 8x8x8 tile also violated the bf16 and
    # i8 wrappers' rowA/colB divisibility contracts. select_gemm_tile honors the
    # per-dtype microkernel expansion, array tiling, even core distribution, the L1
    # budget and the shim-DMA stride range, so awkward shapes still get a valid
    # (small) tile while large GEMMs get much bigger ones. Shapes with no valid tile
    # raise ValueError, which the JIT reports as an unsupported kernel (CPU fallback).
    dev = {"aie2": "npu", "aie2p": "npu2"}.get(arch)
    if dev is None:
        msg = f"Unsupported architecture: {arch}"
        raise ValueError(msg)
    num_cols = 8 if dev == "npu2" else 4

    dtype_in = input_tensors[0].dtype
    dtype_out = output_tensor.dtype
    dtype_in_str = dtype_to_str(dtype_in)
    # The mmul reduces in f32 even when the destination is bf16: narrowing per K step would lose
    # precision. A bf16 C therefore accumulates in f32 and narrows once per tile on the core.
    narrowing = dtype_in_str == "bf16" and dtype_to_str(dtype_out) == "bf16"
    dtype_acc = np.dtype(np.float32) if narrowing else np.dtype(dtype_out)
    r, s, t = resolve_mac_dims(dev, dtype_in_str)
    row_expand, col_expand = resolve_expansion(dev, dtype_in_str)
    # GGML shape convention (innermost first): A is [K, M], B is [K, N].
    M = input_tensors[0].shape[1]
    K = input_tensors[0].shape[0]
    N = input_tensors[1].shape[1]
    # Budget L1 for the accumulator dtype: a narrowing core holds one f32 accumulator plus two
    # bf16 send buffers, the same bytes as two f32 buffers.
    m, k, n = select_gemm_tile(
        dev, M, N, K, dtype_in, dtype_acc, r, s, t, row_expand, col_expand
    )

    current_dir = Path(__file__).resolve().parent
    source_file = str(current_dir / arch / "mm.cc")
    dtype_in_name = dtype_to_str(dtype_in)
    dtype_acc_name = dtype_to_str(dtype_acc)
    compile_args = [
        f"-DDIM_M={m}",
        f"-DDIM_N={n}",
        f"-DDIM_K={k}",
        f"-D{dtype_in_name}_{dtype_acc_name}_ONLY",
        "-DB_COL_MAJ",
        "-DC_COL_MAJ",
    ] + (["-DGEMM_NARROW_BF16"] if narrowing else [])
    # Name the object after the compile flags that vary. Isolation does not depend
    # on this today -- build_iron.py gives each kernel its own work_dir, keyed by a
    # name that already encodes the shapes -- but m/k/n stopped being arch constants
    # when select_gemm_tile started choosing them per shape, so the old fixed name
    # left correctness resting on that directory layout alone. compile_external_kernel
    # caches on object existence without ever comparing against the source or flags.
    #
    # Always a .o, not core_function_object(): this path passes the object to the low-level
    # dialect via link_with, which ld.lld cannot do with textual IR, so this kernel can't inline.
    object_file_name = (
        f"matmul_core_functions_{dtype_in_name}_{dtype_acc_name}"
        f"{'_narrow_bf16' if narrowing else ''}_{m}x{k}x{n}.o"
    )

    zero_fn = ExternalFunction(
        name=f"zero{scalar_suffix}_{dtype_acc_name}",
        object_file_name=object_file_name,
        source_file=source_file,
        arg_types=[np.ndarray[(m, n), np.dtype[dtype_acc]]],
        compile_flags=compile_args,
    )

    matmul_fn = ExternalFunction(
        name=f"matmul{scalar_suffix}_{dtype_in_name}_{dtype_acc_name}",
        object_file_name=object_file_name,
        source_file=source_file,
        arg_types=[
            np.ndarray[(m, k), np.dtype[dtype_in]],
            np.ndarray[(k, n), np.dtype[dtype_in]],
            np.ndarray[(m, n), np.dtype[dtype_acc]],
        ],
        compile_flags=compile_args,
    )

    narrow_fn = (
        ExternalFunction(
            name="narrow_f32_bf16",
            object_file_name=object_file_name,
            source_file=source_file,
            arg_types=[
                np.ndarray[(m, n), np.dtype[dtype_acc]],
                np.ndarray[(m, n), np.dtype[dtype_out]],
            ],
            compile_flags=compile_args,
        )
        if narrowing
        else None
    )

    return (m, n, k, use_scalar, num_cols, zero_fn, matmul_fn, narrow_fn, dtype_acc)


def gemm(arch: str, input_tensors: list, output_tensor):
    """Build the GEMM IRON program (C = A @ B).

    Args:
        arch: Target architecture ("aie2" or "aie2p").
        input_tensors: List of two input tensors [A, B].
        output_tensor: Output tensor C.

    Returns:
        The MLIR module for the GEMM design.

    Raises:
        ValueError: On invalid tensor count, contiguity, shape mismatch, or
            unsupported architecture.
    """
    if len(input_tensors) != 2:
        msg = "Requires two input tensors"
        raise ValueError(msg)

    A = input_tensors[0]  # MxK = A.shape(1) x A.shape(0)
    B = input_tensors[1]  # KxN = B.shape(0) x B.shape(1)
    C = output_tensor  # MxN = C.shape(0) x C.shape(1)

    if not A.contiguous or not B.contiguous or not C.contiguous:
        msg = "Tensors must be contiguous"
        raise ValueError(msg)

    # C is the dense destination; A and B may be zero-padded past it to the tile multiples. The
    # GEMM computes over the padded extent and its mem tiles read back only the dense part
    # (gemm_c_plan.py), so there is no separate de-pad.
    if not (0 < C.shape[0] <= A.shape[1]):
        msg = f"C rows {C.shape[0]} must be in (0, padded M {A.shape[1]}]"
        raise ValueError(msg)
    if not (0 < C.shape[1] <= B.shape[1]):
        msg = f"C columns {C.shape[1]} must be in (0, padded N {B.shape[1]}]"
        raise ValueError(msg)
    # DMA addresses 32-bit words: with an odd M every other bf16 column would start mid-word.
    if np.dtype(C.dtype).itemsize == 2 and C.shape[0] % 2:
        msg = f"a 16-bit C needs an even M for word-aligned columns; got odd M={C.shape[0]}"
        raise ValueError(msg)

    if A.shape[0] != B.shape[0]:
        msg = f"Incompatible K for A and B: {A.shape[0]} != {B.shape[0]}"
        raise ValueError(msg)

    if arch == "aie2":
        dev = "npu"
    elif arch == "aie2p":
        dev = "npu2"
    else:
        msg = f"Unsupported architecture: {arch}"
        raise ValueError(msg)

    (m, n, k, use_scalar, num_cols, zero_fn, matmul_fn, narrow_fn, dtype_acc) = (
        create_mat_mul_external_functions(
            arch=arch, input_tensors=input_tensors, output_tensor=output_tensor
        )
    )

    with mlir_mod_ctx() as ctx:
        my_matmul(
            dev=dev,
            M=A.shape[1],
            N=B.shape[1],
            K=A.shape[0],
            m=m,
            n=n,
            k=k,
            n_aie_cols=num_cols,
            dtype_in_str=dtype_to_str(A.dtype),
            dtype_out_str=dtype_to_str(C.dtype),
            b_col_maj=True,
            c_col_maj=True,
            use_scalar=use_scalar,
            emulate_bf16_mmul_with_bfp16=False,
            trace_size=0,
            zero_fn=zero_fn._name,
            matmul_fn=matmul_fn._name,
            object_file=matmul_fn.object_file_name,
            M_dense=C.shape[0],
            N_dense=C.shape[1],
            dtype_acc_str=dtype_to_str(dtype_acc),
            narrow_fn=narrow_fn._name if narrow_fn else None,
        )
        return ctx.module
