# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Static plan of the GEMM's C (output) data movement into a dense destination.

The GEMM runs over zero-padded operands, A = [Kpad, Mpad] and B = [Kpad, Npad], but writes only the
dense [M, N] result, so no separate de-pad dispatch is needed. The padding is dropped on the mem
tile's *read* side. A DMA write (S2MM) cannot discard stream data it receives, but a DMA read
(MM2S) skips data simply by not reading it.

Terms. The herd is n_aie_rows x n_aie_cols cores, each computing (m, n) C tiles. An *object* is
the (n_aie_rows * m) x n block that one AIE column's mem tile assembles from its cores for one
(row block, column group). Cores iterate row block by row block and, within one, column group by
column group -- the order the runtime sequence streams A and B -- so object i of a column is
(rb, cg) = divmod(i, CG).

Everything here is decided at JIT time from shapes and the tile, so it is pure Python and is
tested without an NPU (tests/ggml-hsa/python/test_gemm_c_plan.py).
"""

from dataclasses import dataclass
from math import prod

# Hardware/toolchain limits (AIE2 target model, mlir-aie 1.4.3).
MAX_QUEUED_TASKS = (
    4  # DMA channel task queue depth; a push onto a full queue is silently dropped
)
MAX_TASK_REPEAT = (
    256  # dma_configure_task repeat_count is 0..255, i.e. up to 256 executions
)
MAX_SHIM_ITERATIONS = 64  # shim BD iteration wrap is 6 bits
SHIM_MAX_STEP = 1 << 20  # shim BD step fields are 20 bits
# Shim BD ids C may use in each ping/pong half; the A/B transfers take 1, 2 and 9, 10.
SHIM_C_BD_IDS = ((0, 3, 4, 5, 6, 7), (8, 11, 12, 13, 14, 15))
MAX_MEM_CHAIN_BDS = 16  # mem-tile MM2S BDs one task chain may use


@dataclass(frozen=True)
class Grid:
    """Dense and padded C extents plus the herd tiling."""

    M: int
    N: int
    Mpad: int
    Npad: int
    m: int
    n: int
    n_aie_rows: int
    n_aie_cols: int

    @property
    def rows_per_block(self) -> int:
        """C rows one row block covers across the herd's rows."""
        return self.m * self.n_aie_rows

    @property
    def RB(self) -> int:  # noqa: N802
        """Number of row blocks."""
        return self.Mpad // self.rows_per_block

    @property
    def CG(self) -> int:  # noqa: N802
        """Number of column groups."""
        return self.Npad // (self.n * self.n_aie_cols)


def make_grid(M, N, Mpad, Npad, m, n, n_aie_rows, n_aie_cols) -> Grid:
    """Build a Grid, rejecting shapes the plan cannot express."""
    g = Grid(M, N, Mpad, Npad, m, n, n_aie_rows, n_aie_cols)
    if Mpad % g.rows_per_block or Npad % (n * n_aie_cols):
        msg = (
            f"padded C [{Mpad}, {Npad}] is not a multiple of the "
            f"({g.rows_per_block}, {n * n_aie_cols}) herd block"
        )
        raise ValueError(msg)
    if not (0 < M <= Mpad and 0 < N <= Npad):
        msg = f"dense C [{M}, {N}] must be non-empty and within the padded [{Mpad}, {Npad}]"
        raise ValueError(msg)
    if Mpad - M >= g.rows_per_block:
        msg = f"the last row block is entirely padding (M={M}, Mpad={Mpad})"
        raise ValueError(msg)
    if Npad - N >= n * n_aie_cols:
        msg = f"the last column group is entirely padding (N={N}, Npad={Npad})"
        raise ValueError(msg)
    if M > SHIM_MAX_STEP:
        msg = f"M={M} exceeds the shim DMA step range {SHIM_MAX_STEP}"
        raise ValueError(msg)
    return g


@dataclass(frozen=True)
class CObject:
    """One object: its (row block, column group) and how much of it is valid."""

    rb: int
    cg: int
    rows: int  # valid rows of the object's rows_per_block, >= 1
    cols: int  # valid columns of this AIE column's n; 0 = the column produces no object here


def column_objects(g: Grid, col: int) -> list[CObject]:
    """All objects of AIE column `col`, in the order its cores compute them."""
    objs = []
    for rb in range(g.RB):
        rows = min(g.rows_per_block, g.M - rb * g.rows_per_block)
        for cg in range(g.CG):
            cols = max(0, min(g.n, g.N - (cg * g.n_aie_cols + col) * g.n))
            objs.append(CObject(rb, cg, rows, cols))
    return objs


def core_kind(o: CObject, core_row: int, m: int) -> str:
    """What core `core_row` of the object's AIE column does for object `o`.

    - "consume": the column produces nothing here. The core only consumes A and B, which are
      broadcast and must be taken by every consumer.
    - "send": every row of this core is padding. It skips the math but still sends its block,
      because the mem tile's per-buffer semaphore counts all n_aie_rows cores. The mem tile never
      reads the block.
    - "compute": compute and send.
    """
    if o.cols == 0:
        return "consume"
    if o.rows <= core_row * m:
        return "send"
    return "compute"


Runs = tuple[tuple[str, int], ...]


def _rle(kinds) -> Runs:
    runs: list[list] = []
    for k in kinds:
        if runs and runs[-1][0] == k:
            runs[-1][1] += 1
        else:
            runs.append([k, 1])
    return tuple((k, c) for k, c in runs)


def sends(runs: Runs) -> int:
    """How many C objects a run list sends (flips the core's ping/pong parity per send)."""
    return sum(c for k, c in runs if k != "consume")


def core_schedule(g: Grid, core_row: int, col: int) -> list[tuple[int, Runs]]:
    """One dispatch of one core as [(repeat, runs)].

    Every row block but the last is fully valid, so they all share one period (one row block's
    column groups); the last row block gets its own entry.
    """
    kinds = [core_kind(o, core_row, g.m) for o in column_objects(g, col)]
    sched = []
    if g.RB > 1:
        period = kinds[: g.CG]
        assert kinds[: (g.RB - 1) * g.CG] == period * (g.RB - 1)
        sched.append((g.RB - 1, _rle(period)))
    sched.append((1, _rle(kinds[(g.RB - 1) * g.CG :])))
    return sched


@dataclass(frozen=True)
class Bd:
    """A DMA pattern: element offset plus (size, stride) pairs, outermost first."""

    offset: int
    sizes: tuple[int, ...]
    strides: tuple[int, ...]

    @property
    def length(self) -> int:
        """Number of elements the pattern visits."""
        return prod(self.sizes)


def _split_rows(o: CObject, g: Grid) -> tuple[int, int]:
    full = min(o.rows // g.m, g.n_aie_rows)
    return full, o.rows - full * g.m


def mem_read_bds(o: CObject, g: Grid) -> list[Bd]:
    """Mem-tile MM2S patterns that read exactly the valid part of object `o`, relative to its buffer.

    The S2MM join stores core c's block as plain column-major m x n at offset c*m*n. Runtime-task
    mem-tile BDs have 3 usable dims, which these patterns never exceed.
    """
    m, n = g.m, g.n
    full, part = _split_rows(o, g)
    bds = []
    if full == g.n_aie_rows and o.cols == n:
        bds.append(Bd(0, (full * m * n,), (1,)))
    elif full:
        bds.append(Bd(0, (full, o.cols, m), (m * n, m, 1)))
    if part:
        bds.append(Bd(full * m * n, (o.cols, part), (m, 1)))
    return bds


def shim_write_bds(o: CObject, g: Grid, col: int) -> list[Bd]:
    """Shim S2MM patterns writing the stream of mem_read_bds(o) into the dense destination.

    The destination is column-major with leading dimension g.M (ggml ne[0]).
    """
    m = g.m
    base = (o.cg * g.n_aie_cols + col) * g.n * g.M + o.rb * g.rows_per_block
    full, part = _split_rows(o, g)
    bds = []
    if full:
        bds.append(Bd(base, (full, o.cols, m), (m, g.M, 1)))
    if part:
        bds.append(Bd(base + full * m, (o.cols, part), (g.M, 1)))
    return bds


@dataclass(frozen=True)
class ShimBd:
    """A shim BD: a <=3-D pattern repeated `iterations` times, `iteration_stride` apart.

    The iteration dimension advances once per *execution* of the BD, not within one, so a BD
    with iterations > 1 only covers its whole pattern when its task repeats it that many times.
    A task repeat re-runs the whole BD chain, so such a BD must be alone in its task.
    """

    offset: int
    iterations: int
    iteration_stride: int
    sizes: tuple[int, int, int]
    strides: tuple[int, int, int]


def _to_shim(bd: Bd) -> ShimBd:
    pad = 3 - len(bd.sizes)
    return ShimBd(bd.offset, 1, 0, (1,) * pad + bd.sizes, (0,) * pad + bd.strides)


def _merge(bds: list[ShimBd]) -> list[ShimBd]:
    """Fold runs of adjacent, identically shaped, evenly spaced BDs into iterated ones.

    Only adjacent BDs merge, so the stream order is unchanged.
    """
    out: list[ShimBd] = []
    for b in bds:
        if out:
            p = out[-1]
            delta = b.offset - (p.offset + (p.iterations - 1) * p.iteration_stride)
            if (
                p.sizes == b.sizes
                and p.strides == b.strides
                and p.iterations < MAX_SHIM_ITERATIONS
                and 0 < delta <= SHIM_MAX_STEP
                and (p.iterations == 1 or delta == p.iteration_stride)
            ):
                out[-1] = ShimBd(p.offset, p.iterations + 1, delta, p.sizes, p.strides)
                continue
        out.append(b)
    return out


def shim_rb_chunks(g: Grid, col: int, rb: int) -> list[list[ShimBd]]:
    """Shim BDs for row block `rb` of AIE column `col`, split into task-sized chunks.

    A BD with iterations > 1 is a chunk of its own (see ShimBd); its task repeats it.
    """
    bds = []
    for o in column_objects(g, col)[rb * g.CG : (rb + 1) * g.CG]:
        if o.cols:
            bds.extend(_to_shim(b) for b in shim_write_bds(o, g, col))
    k = len(SHIM_C_BD_IDS[0])
    chunks: list[list[ShimBd]] = []
    for b in _merge(bds):
        if b.iterations > 1:
            chunks.append([b])
        elif chunks and len(chunks[-1]) < k and chunks[-1][0].iterations == 1:
            chunks[-1].append(b)
        else:
            chunks.append([b])
    return chunks


@dataclass(frozen=True)
class MemTask:
    """One mem-tile MM2S runtime task: a chain of objects, executed `repeat` times."""

    buffers: tuple[int, ...]  # ping/pong buffer of each object in the chain
    bds: tuple[tuple[Bd, ...], ...]  # read patterns of each object
    repeat: int  # executions of the whole chain, 1..MAX_TASK_REPEAT


def mem_buffers(g: Grid, col: int) -> int:
    """Number of mem-tile C buffers of AIE column `col`: 2 (ping/pong) or 1.

    The static S2MM join alternates over the buffers without end, across dispatches, while every
    dispatch's runtime sequence reads the same buffer sequence starting at buffer 0. That only
    agrees if each dispatch fills an even number of buffers, so a column producing an odd number
    of objects per dispatch gets a single buffer.
    """
    n_objs = sum(1 for o in column_objects(g, col) if o.cols)
    return 1 if n_objs % 2 else 2


def _mem_task(
    objs: list[CObject], first_buf: int, repeat: int, g: Grid, nbuf: int
) -> MemTask:
    return MemTask(
        tuple((first_buf + i) % nbuf for i in range(len(objs))),
        tuple(tuple(mem_read_bds(o, g)) for o in objs),
        repeat,
    )


def mem_tasks(g: Grid, col: int) -> list[MemTask]:
    """The mem-tile MM2S tasks for one whole dispatch of AIE column `col`, queued up front.

    Row blocks before the last are identical, so one chain (one row block's produced objects,
    doubled when that count is odd so the chain returns to the same ping/pong buffer) repeats
    across them. The last row block is its own task. Object i of the dispatch is read from buffer
    i % mem_buffers(g, col).
    """
    objs = column_objects(g, col)
    nbuf = mem_buffers(g, col)

    def produced(rb):
        return [o for o in objs[rb * g.CG : (rb + 1) * g.CG] if o.cols]

    tasks = []
    buf = 0
    if g.RB > 1:
        period = produced(0)
        if period:
            odd = len(period) % 2
            chain = period * 2 if odd else period
            runs = (g.RB - 1) // 2 if odd else g.RB - 1
            while runs:
                r = min(runs, MAX_TASK_REPEAT)
                tasks.append(_mem_task(chain, buf, r, g, nbuf))
                runs -= r
            if odd and (g.RB - 1) % 2:
                tasks.append(_mem_task(period, buf, 1, g, nbuf))
                buf = (buf + 1) % nbuf
    last = produced(g.RB - 1)
    if last:
        tasks.append(_mem_task(last, buf, 1, g, nbuf))

    if len(tasks) > MAX_QUEUED_TASKS:
        msg = f"column {col} needs {len(tasks)} mem-tile C tasks; the queue holds {MAX_QUEUED_TASKS}"
        raise ValueError(msg)
    for t in tasks:
        n_bds = sum(len(b) for b in t.bds)
        if n_bds > MAX_MEM_CHAIN_BDS:
            msg = f"column {col} needs a {n_bds}-BD mem-tile chain (max {MAX_MEM_CHAIN_BDS})"
            raise ValueError(msg)
    return tasks


def validate(g: Grid) -> None:
    """Raise ValueError if any column's plan exceeds a hardware limit."""
    for col in range(g.n_aie_cols):
        mem_tasks(g, col)
