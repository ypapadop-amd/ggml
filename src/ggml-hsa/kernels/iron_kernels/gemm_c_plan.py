# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Static plan of the GEMM's C (output) data movement into a dense destination.

The GEMM runs over zero-padded operands, A = [Kpad, Mpad] and B = [Kpad, Npad], but writes only the
dense [M, N] result, so no separate post-pass dispatch is needed. The padding is dropped on the mem
tile's *read* side. A DMA write (S2MM) cannot discard stream data it receives, but a DMA read
(MM2S) skips data simply by not reading it.

Terms. The herd is n_aie_rows x n_aie_cols cores, each computing (m, n) C tiles. An *object* is
the (n_aie_rows * m) x n block that one AIE column's mem tile assembles from its cores for one
(row block, column group). Cores iterate row block by row block and, within one, column group by
column group -- the order the runtime sequence streams A and B -- so object i of a column is
(rb, cg) = divmod(i, CG).

Two ways to end at the dense edge. By default the last row block and column group are *clipped*:
they run over the zero padding, and the mem tile reads back only their valid rows and columns. A
GEMM whose operands are not padded along a dimension instead *shifts* its last row block (column
group) back to end at M (N): every row (column) of it is valid, and it recomputes rows (columns)
the previous one already wrote. Those overlapping writes carry bit-identical values -- the same A
rows, B columns and K order -- so the order in which they land does not matter.

Everything here is decided at JIT time from shapes and the tile, so it is pure Python and is
tested without an NPU (tests/ggml-hsa/python/test_gemm_c_plan.py).
"""

from dataclasses import dataclass
from functools import cache
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
# Shim BD ids C may use in each ping/pong half; the A/B transfers take 1, 2 and 9, 10. A B split
# into two runs by a shifted last column group also takes 5 and 13 (see shim_c_bd_ids).
SHIM_C_BD_IDS = ((0, 3, 4, 5, 6, 7), (8, 11, 12, 13, 14, 15))
SHIM_B_SHIFTED_RUN_BD_OFFSET = 5  # per-half offset of the shifted B run's BD id
# Mem-tile BDs the runtime sequence may hold at once. A mem tile has 48 BDs, but even channels
# only reach BDs 0-23 and odd channels 24-47 (AIE2TargetModel::isBdChannelAccessible), and the
# runtime-sequence allocator draws every mem-tile task BD from the even half. Of those 24, the
# static DMA code takes 8: the S2MM join's even channels 2 and 4 (one BD per C buffer each, so 4
# with ping/pong) and the A/B ObjectFifos' even channels S2MM 0 and MM2S 0 (2 BDs each at depth
# 2), read from aiecc's input_with_addresses.mlir on aie2 and aie2p. 24 - 8 = 16. Checked at the
# boundary with aiecc: 16 live BDs compile (aie2 2048x50257x768), 18 fail (aie2p 2000x50257x4096).
MAX_LIVE_MEM_BDS = 16


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
    shift_rows: bool = False  # the last row block ends at M instead of being clipped
    shift_cols: bool = False  # the last column group ends at N instead of being clipped

    @property
    def rows_per_block(self) -> int:
        """C rows one row block covers across the herd's rows."""
        return self.m * self.n_aie_rows

    @property
    def group_width(self) -> int:
        """C columns one column group covers across the herd's columns."""
        return self.n * self.n_aie_cols

    @property
    def RB(self) -> int:  # noqa: N802
        """Number of row blocks."""
        return self.Mpad // self.rows_per_block

    @property
    def CG(self) -> int:  # noqa: N802
        """Number of column groups."""
        return self.Npad // (self.n * self.n_aie_cols)


def make_grid(
    M,
    N,
    Mpad,
    Npad,
    m,
    n,
    n_aie_rows,
    n_aie_cols,
    shift_rows=False,
    shift_cols=False,
) -> Grid:
    """Build a Grid, rejecting shapes the plan cannot express.

    shift_rows / shift_cols select the shifted edge (see the module docstring) along M / N; they
    are dropped when that dimension is not padded, where there is nothing to shift.
    """
    g = Grid(
        M,
        N,
        Mpad,
        Npad,
        m,
        n,
        n_aie_rows,
        n_aie_cols,
        shift_rows=shift_rows and Mpad > M,
        shift_cols=shift_cols and Npad > N,
    )
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
    if g.shift_rows and g.rows_per_block > M:
        msg = f"a shifted last row block needs M={M} >= {g.rows_per_block} rows"
        raise ValueError(msg)
    if g.shift_cols and g.group_width > N:
        msg = f"a shifted last column group needs N={N} >= {g.group_width} columns"
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


def row_origin(g: Grid, rb: int) -> int:
    """First dense C row of row block `rb`: M - rows_per_block for a shifted last one."""
    if g.shift_rows and rb == g.RB - 1:
        return g.M - g.rows_per_block
    return rb * g.rows_per_block


def col_origin(g: Grid, cg: int, col: int) -> int:
    """First dense C column of AIE column `col` in column group `cg`.

    A shifted last column group starts at N - group_width.
    """
    if g.shift_cols and cg == g.CG - 1:
        return g.N - g.group_width + col * g.n
    return (cg * g.n_aie_cols + col) * g.n


def column_objects(g: Grid, col: int, rbs: range | None = None) -> list[CObject]:
    """All objects of AIE column `col` (of row blocks `rbs`), in the order its cores compute them."""
    objs = []
    for rb in range(g.RB) if rbs is None else rbs:
        rows = (
            g.rows_per_block
            if g.shift_rows
            else min(g.rows_per_block, g.M - rb * g.rows_per_block)
        )
        for cg in range(g.CG):
            cols = (
                g.n
                if g.shift_cols
                else max(0, min(g.n, g.N - (cg * g.n_aie_cols + col) * g.n))
            )
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

    The destination is column-major with leading dimension g.M (ggml ne[0]). A shifted object
    starts at its shifted origin (row_origin, col_origin).
    """
    m = g.m
    base = col_origin(g, o.cg, col) * g.M + row_origin(g, o.rb)
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


def b_group_runs(g: Grid) -> list[tuple[int, int]]:
    """B's column groups in transfer order, as (group count, first column) runs.

    One run from column 0, or, with a shifted last column group, a second run of that group from
    N - group_width. Each run is one shim BD for B.
    """
    if not g.shift_cols:
        return [(g.CG, 0)]
    return [(g.CG - 1, 0), (1, g.N - g.group_width)]


def shim_c_bd_ids(g: Grid) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shim BD ids C may use in each ping/pong half: SHIM_C_BD_IDS less the shifted B run's."""
    if len(b_group_runs(g)) == 1:
        return SHIM_C_BD_IDS
    return tuple(
        tuple(i for i in ids if i != 8 * half + SHIM_B_SHIFTED_RUN_BD_OFFSET)
        for half, ids in enumerate(SHIM_C_BD_IDS)
    )


def shim_rb_chunks(g: Grid, col: int, rb: int) -> list[list[ShimBd]]:
    """Shim BDs for row block `rb` of AIE column `col`, split into task-sized chunks.

    A BD with iterations > 1 is a chunk of its own (see ShimBd); its task repeats it.
    """
    bds = []
    for o in column_objects(g, col, range(rb, rb + 1)):
        if o.cols:
            bds.extend(_to_shim(b) for b in shim_write_bds(o, g, col))
    k = len(shim_c_bd_ids(g)[0])
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

    @property
    def n_bds(self) -> int:
        """Mem-tile BDs the task's chain holds while it is live."""
        return sum(len(b) for b in self.bds)


def mem_buffers(g: Grid, col: int) -> int:
    """Number of mem-tile C buffers of AIE column `col`: 2 (ping/pong) or 1.

    The static S2MM join alternates over the buffers without end, across dispatches, while every
    dispatch's runtime sequence reads the same buffer sequence starting at buffer 0. That only
    agrees if each dispatch fills an even number of buffers, so a column producing an odd number
    of objects per dispatch gets a single buffer.
    """
    n_objs = sum(1 for o in column_objects(g, col) if o.cols)
    return 1 if n_objs % 2 else 2


def _compress(objs: list[CObject], first_buf: int, g: Grid, nbuf: int) -> list[MemTask]:
    """Mem tasks reading `objs` in order, object i from buffer (first_buf + i) % nbuf.

    Each run of objects with the same read geometry repeats its smallest unit -- nbuf objects,
    one per buffer -- through the task repeat count. When the run has more units than one repeat
    count holds, the chain gets more copies of the unit rather than the run more tasks. Objects
    left over (a run too short to repeat, or the remainder of one) are chained, in order, into a
    single repeat-1 task together with any leftovers that directly follow.
    """
    tasks: list[MemTask] = []
    pending: list[CObject] = []
    buf = first_buf

    def task(chain, repeat):
        nonlocal buf
        tasks.append(
            MemTask(
                tuple((buf + i) % nbuf for i in range(len(chain))),
                tuple(tuple(mem_read_bds(o, g)) for o in chain),
                repeat,
            )
        )
        buf = (buf + len(chain) * repeat) % nbuf

    def flush():
        if pending:
            task(list(pending), 1)
            pending.clear()

    geoms = [(o.rows, o.cols) for o in objs]  # mem_read_bds depends on these only
    i = 0
    while i < len(objs):
        j = i
        while j < len(objs) and geoms[j] == geoms[i]:
            j += 1
        units = (j - i) // nbuf
        if units >= 2:
            # The fewest copies that fit the repeat count, lengthened while that shrinks
            # copies + leftover units (both cost BDs).
            k0 = -(-units // MAX_TASK_REPEAT)
            k = min(range(k0, 2 * k0 + 1), key=lambda c: c + units % c)
            flush()
            task(objs[i : i + k * nbuf], units // k)
            pending.extend(objs[i + (units // k) * k * nbuf : j])
        else:
            pending.extend(objs[i:j])
        i = j
    flush()
    return tasks


MEM_LIVE_ROW_BLOCKS = 2  # per-row-block groups live at once (see MemPlan)


@dataclass(frozen=True)
class MemPlan:
    """How the runtime sequence issues one AIE column's mem-tile MM2S tasks.

    Up front (per_row_block False): groups[0] holds every task of the dispatch, queued before the
    first row block and never awaited, so all of them are live for the whole sequence.

    Per row block (per_row_block True): groups[rb] is queued at the start of row block rb. The
    sequence awaits row block rb's shim C tasks before it starts row block rb + 2 (gemm.py's
    await_all(col, keep=1)), and the shim receiving all of rb's C proves the mem tile finished
    reading it; only then are rb's tasks freed and their BDs reused. So at most two consecutive
    groups are live at once (MEM_LIVE_ROW_BLOCKS).
    """

    per_row_block: bool
    groups: tuple[tuple[MemTask, ...], ...]

    def live(self) -> tuple[int, int]:
        """The most (tasks, BDs) live on the mem tile's MM2S channel at once."""
        if not self.per_row_block:
            ts = self.groups[0] if self.groups else ()
            return len(ts), sum(t.n_bds for t in ts)
        tasks = bds = 0
        for rb in range(len(self.groups)):
            window = self.groups[max(0, rb - MEM_LIVE_ROW_BLOCKS + 1) : rb + 1]
            tasks = max(tasks, sum(len(g) for g in window))
            bds = max(bds, sum(t.n_bds for g in window for t in g))
        return tasks, bds


def _fits(plan: MemPlan) -> bool:
    tasks, bds = plan.live()
    return tasks <= MAX_QUEUED_TASKS and bds <= MAX_LIVE_MEM_BDS


@cache
def mem_plan(g: Grid, col: int) -> MemPlan:
    """The mem-tile MM2S plan of AIE column `col` for one dispatch.

    Object i of the dispatch is read from buffer i % mem_buffers(g, col). A column whose row
    blocks each read one repeating geometry (every row block but the last is identical) queues
    its compressed tasks up front. A column whose row block mixes geometries -- the one that
    straddles N when there are several column groups: full objects then a clipped one -- would
    need tasks per row block, so it issues them per row block (see MemPlan). An up-front plan
    that exceeds the task queue or the live-BD budget (many row blocks and column groups) falls
    back to per row block too.
    Raises ValueError when neither fits.
    """
    nbuf = mem_buffers(g, col)
    objs = column_objects(g, col)
    per_rb = [
        [o for o in objs[rb * g.CG : (rb + 1) * g.CG] if o.cols] for rb in range(g.RB)
    ]
    uniform = len({tuple(mem_read_bds(o, g)) for o in per_rb[0]}) <= 1

    if uniform:
        everything = [o for r in per_rb for o in r]
        plan = MemPlan(
            per_row_block=False, groups=(tuple(_compress(everything, 0, g, nbuf)),)
        )
        if _fits(plan):
            return plan
    groups, buf = [], 0
    for r in per_rb:
        groups.append(tuple(_compress(r, buf, g, nbuf)))
        buf = (buf + len(r)) % nbuf
    plan = MemPlan(per_row_block=True, groups=tuple(groups))
    if not _fits(plan):
        tasks, bds = plan.live()
        msg = (
            f"column {col} needs {tasks} live mem-tile C tasks with {bds} BDs; the queue holds "
            f"{MAX_QUEUED_TASKS} and the budget is {MAX_LIVE_MEM_BDS} BDs"
        )
        raise ValueError(msg)
    return plan


def validate(g: Grid) -> None:
    """Raise ValueError if any column's plan exceeds a hardware limit."""
    for col in range(g.n_aie_cols):
        mem_plan(g, col)
