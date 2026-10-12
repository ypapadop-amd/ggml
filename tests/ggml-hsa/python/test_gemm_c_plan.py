# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for the GEMM's dense-destination C plan (``gemm_c_plan``).

The simulation test replays the plan at the address level. Every element the mem tiles read is
mapped to the dense destination index it belongs to. That sequence must equal the sequence of
indices the shim BDs write, and the union over all columns must be exactly [0, M*N). This catches
wrong offsets, strides, stream order, ping/pong parity and coverage without an NPU.

A shifted grid (aie2's unpadded f32 B, see gemm_c_plan) maps each element by its *shifted*
destination: the last row block starts at M - rows_per_block and the last column group at
N - group_width, both computed here independently of the plan. Its writes cover [0, M*N) with
duplicates exactly in the overlap the shift recomputes.
"""

import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

KERNELS_DIR = Path(__file__).resolve().parents[3] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

from iron_kernels import gemm_c_plan as P
from iron_kernels.gemm import (
    create_mat_mul_external_functions,
    resolve_expansion,
    resolve_mac_dims,
    select_gemm_tile,
)

# (arch device name, gm, gk, gn, n_aie_cols): the backend's padding granularity, which
# test_gemm_tiling.py pins against ggml-hsa.cpp.
DEVS = {"npu2": (8, 8, 16, 8), "npu": (16, 8, 16, 4)}

# Dense (M, N, K) shapes. Each exercises one mechanism; see the spec's shape matrix.
SHAPES = [
    (512, 512, 512),  # no clip
    (500, 500, 784),  # MNIST fc1: partial core + partial last AIE column
    (10, 500, 500),  # two column groups, clip in the last one
    (10, 500, 784),  # cores 1-3 all padding
    (392000, 8, 9),  # conv1: AIE columns 1-7 all padding, long M
    (98000, 16, 72),  # conv2: all-padding columns + row clip, long M
    (499, 500, 784),  # odd M (f32 destination only)
    (33, 130, 64),  # small, awkward
    (1000, 2000, 1000),  # straddling column issued per row block
    (1000, 260, 512),  # CG 3, straddling column 0
    (1500, 2436, 64),  # many row blocks, straddling column
    (256, 50257, 64),  # LM-head-like N: odd CG > 16
]


def _pad(x, g):
    return -(-x // g) * g


def _grid(dev, M, N, K):
    gm, gk, gn, cols = DEVS[dev]
    Mpad, Npad, Kpad = _pad(M, gm * 4), _pad(N, gn * cols), _pad(K, gk)
    r, s, t = resolve_mac_dims(dev, "bf16")
    re, ce = resolve_expansion(dev, "bf16")
    m, _k, n = select_gemm_tile(
        dev,
        Mpad,
        Npad,
        Kpad,
        np.dtype(ml_dtypes.bfloat16),
        np.dtype(np.float32),
        r,
        s,
        t,
        re,
        ce,
    )
    return P.make_grid(M, N, Mpad, Npad, m, n, 4, cols)


class _TD:
    """The tensor-descriptor fields create_mat_mul_external_functions reads."""

    def __init__(self, dtype, ne) -> None:  # noqa: D107
        self.dtype, self.shape = np.dtype(dtype), tuple(ne)


def _f32_b_grid(M, N, K):
    """aie2 grid of an f32 x f32 MUL_MAT as the backend hands it to the GEMM.

    Mirrors ggml_hsa_prepare_mul_mat_f32: A is converted and padded; an f32 B at least one
    column group wide is read unpadded (else padded, still f32); C is the dense [M, N]. The
    shift decisions and tile come from gemm.py itself.
    """
    from aie.iron import ExternalFunction

    gm, gk, gn, cols = DEVS["npu"]
    Mpad, Npad, Kpad = _pad(M, gm * 4), _pad(N, gn * cols), _pad(K, gk)
    b = (K, N) if N >= gn * cols else (Kpad, Npad)
    ops = [
        _TD(ml_dtypes.bfloat16, (Kpad, Mpad, 1, 1)),
        _TD(np.float32, (*b, 1, 1)),
    ]
    got = create_mat_mul_external_functions("aie2", ops, _TD(np.float32, (M, N, 1, 1)))
    ExternalFunction._instances.clear()
    m, n, _k = got[0], got[1], got[2]
    n_pad, shift_rows, shift_cols = got[8], got[9], got[10]
    assert n_pad == Npad
    return P.make_grid(
        M, N, Mpad, Npad, m, n, 4, cols, shift_rows=shift_rows, shift_cols=shift_cols
    )


# Dense (M, N, K) f32 x f32 shapes on aie2's f32-B path: test-mul-mat-f32-hsa's cases, MNIST,
# and larger ones. Each is shifted, clipped or mixed per dimension.
SHIFT_SHAPES = [
    (512, 512, 512),  # aligned: nothing to shift or clip
    (500, 500, 784),  # MNIST fc1: rows and columns shifted
    (100, 64, 256),  # rows shifted
    (129, 300, 1000),  # rows and columns shifted
    (1000, 70, 40),  # many row blocks, columns shifted
    (64, 100, 500),  # K tail, columns shifted
    (64, 70, 257),
    (10, 500, 500),  # MNIST fc2: rows clipped (below a row block), columns shifted
    (10, 500, 784),
    (33, 257, 129),
    (300, 8, 256),  # N below a column group: B padded, both clipped
    (4000, 500, 4000),
    (4096, 512, 4096),
    (3000, 1000, 64),
    (98000, 64, 72),  # long M, one column group: rows shifted
    (8000, 100, 72),
]


def _row_start(g, rb):
    """First destination row of row block rb, shifted or not (independent of the plan)."""
    if g.shift_rows and rb == g.RB - 1:
        return g.M - g.rows_per_block
    return rb * g.rows_per_block


def _col_start(g, cg, col):
    """First destination column of AIE column col in group cg (independent of the plan)."""
    gw = g.n * g.n_aie_cols
    if g.shift_cols and cg == g.CG - 1:
        return g.N - gw + col * g.n
    return cg * gw + col * g.n


def _expected_writes(g):
    """How many times each dense C element is written: 2 per shifted overlap it lies in."""
    rows = np.ones(g.M, dtype=np.int64)
    if g.shift_rows:
        rows[g.M - g.rows_per_block : (g.RB - 1) * g.rows_per_block] = 2
    cols = np.ones(g.N, dtype=np.int64)
    if g.shift_cols:
        gw = g.n * g.n_aie_cols
        cols[g.N - gw : (g.CG - 1) * gw] = 2
    return (cols[:, None] * rows[None, :]).reshape(-1)  # column-major, index j*M + i


def _addresses(offset, sizes, strides):
    """Element addresses a DMA pattern visits, in order (sizes/strides outermost first)."""
    idx = np.array(offset, dtype=np.int64)
    for s, st in zip(sizes, strides):
        idx = idx[..., None] + np.arange(s, dtype=np.int64) * st
    return idx.reshape(-1)


def _mem_tasks_in_order(g, col):
    """Every mem task of one dispatch in the order the runtime sequence queues them.

    An up-front plan queues its single group before the first row block; a per-row-block plan
    queues group rb at the start of row block rb. Either way the channel runs them in this order.
    """
    plan = P.mem_plan(g, col)
    if plan.per_row_block:
        assert len(plan.groups) == g.RB
    else:
        assert len(plan.groups) == 1
    return [t for group in plan.groups for t in group]


def _mem_stream(g, col):
    """Dense destination index of every element the mem tile streams to the shim, in order."""
    produced = [o for o in P.column_objects(g, col) if o.cols]
    out, i = [], 0
    for task in _mem_tasks_in_order(g, col):
        for _ in range(task.repeat):
            for buf, obj_bds in zip(task.buffers, task.bds):
                o = produced[i]
                assert buf == i % P.mem_buffers(g, col), f"object {i} uses buffer {buf}"
                assert tuple(obj_bds) == tuple(P.mem_read_bds(o, g)), (
                    f"object {i} geometry"
                )
                for bd in obj_bds:
                    off = _addresses(bd.offset, bd.sizes, bd.strides)
                    core, rem = np.divmod(off, g.m * g.n)
                    jj, ii = np.divmod(rem, g.m)
                    row = _row_start(g, o.rb) + core * g.m + ii
                    colg = _col_start(g, o.cg, col) + jj
                    out.append(colg * g.M + row)
                i += 1
    assert i == len(produced), f"tasks cover {i} of {len(produced)} objects"
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _shim_stream(g, col):
    out = []
    for rb in range(g.RB):
        for chunk in P.shim_rb_chunks(g, col, rb):
            assert len(chunk) <= len(P.shim_c_bd_ids(g)[0])
            # The iteration dim advances per BD execution: an iterated BD needs its own task.
            assert len(chunk) == 1 or all(b.iterations == 1 for b in chunk)
            for b in chunk:
                out.append(
                    _addresses(
                        b.offset,
                        (b.iterations, *b.sizes),
                        (b.iteration_stride, *b.strides),
                    )
                )
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _check_streams(g):
    written = []
    for col in range(g.n_aie_cols):
        mem, shim = _mem_stream(g, col), _shim_stream(g, col)
        assert np.array_equal(mem, shim), (
            f"column {col}: mem read order != shim write order"
        )
        written.append(shim)
    allw = np.concatenate(written)
    assert allw.min() >= 0
    assert allw.max() < g.M * g.N, "a write lands past the dense C"
    assert np.array_equal(
        np.bincount(allw, minlength=g.M * g.N), _expected_writes(g)
    ), "dense C not covered once, plus once more in each shifted overlap"


@pytest.mark.parametrize("dev", list(DEVS))
@pytest.mark.parametrize("shape", SHAPES)
def test_streams_pair_up_and_cover_dense_c_exactly_once(dev, shape):
    try:
        g = _grid(dev, *shape)
    except ValueError as e:
        pytest.skip(f"not tileable on {dev}: {e}")
    assert not (g.shift_rows or g.shift_cols)
    _check_streams(g)


@pytest.mark.parametrize("shape", SHIFT_SHAPES)
def test_shifted_streams_pair_up_and_cover_dense_c(shape):
    """aie2 f32 B: shifted destinations pair up, and only the shifted overlap is rewritten."""
    _check_streams(_f32_b_grid(*shape))


@pytest.mark.parametrize(
    ("shape", "shift"),
    [
        ((500, 500, 784), (True, True)),
        ((10, 500, 500), (False, True)),
        ((100, 64, 256), (True, False)),
        ((300, 8, 256), (False, False)),
        ((512, 512, 512), (False, False)),
    ],
)
def test_f32_b_shift_decisions(shape, shift):
    """Rows shift once C holds a row block; columns once B is read unpadded and ragged."""
    g = _f32_b_grid(*shape)
    assert (g.shift_rows, g.shift_cols) == shift


@pytest.mark.parametrize("shape", SHIFT_SHAPES)
def test_operands_follow_the_shifted_c(shape):
    """A's rows and B's columns come from where C's are written, so overlaps recompute equal values.

    gemm.py offsets A's transfer by row_origin and splits B's into b_group_runs; both must place
    each row block / column group where the shim writes its C.
    """
    g = _f32_b_grid(*shape)
    for rb in range(g.RB):
        assert P.row_origin(g, rb) == _row_start(g, rb)
    b_starts = [
        start + i * g.n * g.n_aie_cols
        for count, start in P.b_group_runs(g)
        for i in range(count)
    ]
    assert len(b_starts) == g.CG
    for cg, start in enumerate(b_starts):
        for col in range(g.n_aie_cols):
            assert start + col * g.n == _col_start(g, cg, col)
    # every operand row/column read exists
    assert b_starts[-1] + g.n * g.n_aie_cols <= (g.N if g.shift_cols else g.Npad)
    assert _row_start(g, g.RB - 1) + g.rows_per_block <= (
        g.M if g.shift_rows else g.Mpad
    )


def test_shifted_b_run_takes_a_c_bd_id():
    """A split B's second run uses BD 5 / 13, so C gives it up in both halves."""
    g = _f32_b_grid(500, 500, 784)
    assert len(P.b_group_runs(g)) == 2
    ids = P.shim_c_bd_ids(g)
    assert ids == ((0, 3, 4, 6, 7), (8, 11, 12, 14, 15))
    b_ids = {1, 2, P.SHIM_B_SHIFTED_RUN_BD_OFFSET}
    for half in range(2):
        assert not ({8 * half + i for i in b_ids} & set(ids[half]))


def _check_core_sends(g):
    for col in range(g.n_aie_cols):
        produced = sum(1 for o in P.column_objects(g, col) if o.cols)
        for row in range(g.n_aie_rows):
            sched = P.core_schedule(g, row, col)
            kinds = [
                k
                for rep, runs in sched
                for _ in range(rep)
                for k, c in runs
                for _ in range(c)
            ]
            assert len(kinds) == g.RB * g.CG
            assert sum(k != "consume" for k in kinds) == produced
            # A shifted row block has no all-padding core; a shifted column group no idle column.
            if g.shift_rows:
                assert "send" not in kinds
            if g.shift_cols:
                assert "consume" not in kinds


@pytest.mark.parametrize("dev", list(DEVS))
@pytest.mark.parametrize("shape", SHAPES)
def test_every_core_in_a_column_sends_once_per_produced_object(dev, shape):
    try:
        g = _grid(dev, *shape)
    except ValueError as e:
        pytest.skip(f"not tileable on {dev}: {e}")
    _check_core_sends(g)


@pytest.mark.parametrize("shape", SHIFT_SHAPES)
def test_every_core_sends_once_per_object_when_shifted(shape):
    _check_core_sends(_f32_b_grid(*shape))


def _mem_buffer_sequence(g, col):
    """Mem-tile buffer each object of one dispatch is read from, in order."""
    return [
        buf
        for task in _mem_tasks_in_order(g, col)
        for _ in range(task.repeat)
        for buf in task.buffers
    ]


def _parity_grids():
    for dev in DEVS:
        for shape in SHAPES:
            try:
                yield f"{dev}-{shape}", _grid(dev, *shape)
            except ValueError:
                continue
    # dense == padded (Task 2's C shape): conv2 produces an odd 1021 objects per column,
    # M=32 x N=512 one object per column.
    for shape in SHIFT_SHAPES:
        yield f"f32b-{shape}", _f32_b_grid(*shape)
    yield "conv2-padded", P.make_grid(98016, 128, 98016, 128, 24, 16, 4, 8)
    yield "m32-padded", P.make_grid(32, 512, 32, 512, 8, 64, 4, 8)


@pytest.mark.parametrize(("name", "g"), list(_parity_grids()))
def test_mem_buffers_agree_across_dispatches(name, g):
    # The static join fills buffer j % nbuf for the column's j-th object ever received, across
    # dispatches (every core sends each produced object once per dispatch). Each dispatch's
    # runtime sequence is identical, so its reads must match the join in both dispatches.
    for col in range(g.n_aie_cols):
        count = sum(1 for o in P.column_objects(g, col) if o.cols)
        nbuf = P.mem_buffers(g, col)
        for row in range(g.n_aie_rows):
            sends = sum(
                rep * P.sends(runs) for rep, runs in P.core_schedule(g, row, col)
            )
            assert sends == count
        join = [j % nbuf for j in range(2 * count)]
        reads = _mem_buffer_sequence(g, col)
        for d in range(2):
            assert reads == join[d * count : (d + 1) * count], (
                f"{name} column {col}: dispatch {d} reads buffers out of step with the join"
            )


def test_fc1_geometry():
    g = P.make_grid(500, 500, 512, 512, 32, 64, 4, 8)
    last = P.column_objects(g, 7)[-1]
    assert (last.rb, last.cg, last.rows, last.cols) == (3, 0, 116, 52)
    assert P.mem_read_bds(last, g) == [
        P.Bd(0, (3, 52, 32), (2048, 32, 1)),
        P.Bd(3 * 2048, (52, 20), (32, 1)),
    ]
    full = P.column_objects(g, 0)[0]
    assert P.mem_read_bds(full, g) == [P.Bd(0, (4 * 32 * 64,), (1,))]


def test_conv1_idle_columns_produce_nothing():
    g = P.make_grid(392000, 8, 392000, 128, 200, 16, 4, 8)
    assert P.mem_plan(g, 1) == P.MemPlan(per_row_block=False, groups=((),))
    assert P.core_schedule(g, 0, 1) == [
        (489, (("consume", 1),)),
        (1, (("consume", 1),)),
    ]
    assert P.core_schedule(g, 0, 0) == [
        (489, (("compute", 1),)),
        (1, (("compute", 1),)),
    ]


def test_conv2_long_m_fits_the_task_queue():
    g = P.make_grid(98000, 16, 98016, 128, 24, 16, 4, 8)
    tasks = _mem_tasks_in_order(g, 0)
    assert len(tasks) <= P.MAX_QUEUED_TASKS
    assert all(1 <= t.repeat <= P.MAX_TASK_REPEAT for t in tasks)


def test_iterated_shim_bd_gets_its_own_task():
    # Column 7: column groups 0 and 1 are full (merged into one iterated BD), group 2 is
    # partial. The iteration dim advances per BD execution, so the iterated BD must not share
    # a task (whose repeat re-runs the whole chain) with the partial one.
    g = P.make_grid(32, 760, 32, 768, 8, 32, 4, 8)
    chunks = P.shim_rb_chunks(g, 7, 0)
    assert [[b.iterations for b in c] for c in chunks] == [[2], [1]]


def test_m10_cores_1_to_3_only_send():
    g = P.make_grid(10, 500, 32, 512, 8, 64, 4, 8)
    assert [P.core_kind(P.column_objects(g, 0)[0], r, 8) for r in range(4)] == [
        "compute",
        "compute",
        "send",
        "send",
    ]


@pytest.mark.parametrize(
    "args",
    [
        (600, 512, 512, 512, 32, 64, 4, 8),  # M > Mpad
        (300, 512, 512, 512, 32, 64, 4, 8),  # last row block entirely padding
        (512, 512, 500, 512, 32, 64, 4, 8),  # Mpad not a herd multiple
    ],
)
def test_make_grid_rejects_bad_shapes(args):
    with pytest.raises(ValueError):
        P.make_grid(*args)


@pytest.mark.parametrize(
    ("args", "kw"),
    [
        ((100, 512, 128, 512, 32, 64, 4, 8), {"shift_rows": True}),  # M < row block
        ((512, 500, 512, 512, 32, 128, 4, 4), {"shift_cols": True}),  # N < group
    ],
)
def test_make_grid_rejects_a_shift_wider_than_c(args, kw):
    with pytest.raises(ValueError, match="shifted"):
        P.make_grid(*args, **kw)


def test_make_grid_drops_a_shift_along_an_unpadded_dimension():
    g = P.make_grid(512, 512, 512, 512, 32, 64, 4, 8, shift_rows=True, shift_cols=True)
    assert not (g.shift_rows or g.shift_cols)


def _simulate_issue(g, col):
    """Replay gemm.py's issue/await order for one column's mem-tile MM2S channel.

    Mirrors the runtime sequence: an up-front plan queues its tasks before row block 0 and never
    frees them. Per row block rb, a per-row-block plan queues group rb; the shim C chunks follow
    (every chunk after the first awaits all outstanding entries first); the group is attached to
    the row block's last entry; from rb 1 on, all entries but the newest are awaited
    (await_all(col, keep=1)). Awaiting an entry frees the groups attached to it.

    Asserts a group is freed only once every shim entry of its row block was awaited -- the shim
    having received all of that row block's C is what proves the mem tile finished reading it.
    Returns the peak (live tasks, live BDs).
    """
    plan = P.mem_plan(g, col)
    live = {}
    peak = (0, 0)

    def note():
        nonlocal peak
        ts = [t for group in live.values() for t in group]
        peak = max(peak, (len(ts), sum(t.n_bds for t in ts)))

    if not plan.per_row_block:
        live["up-front"] = plan.groups[0]
        note()
    n_entries, awaited, outstanding = {}, set(), []

    def await_all(keep=0):
        n = len(outstanding) - keep
        for rb, i, frees in outstanding[:n]:
            awaited.add((rb, i))
            for key in frees:
                assert all((key, j) in awaited for j in range(n_entries[key])), (
                    f"column {col}: row block {key}'s mem BDs freed before its shim C completed"
                )
                del live[key]
        del outstanding[:n]

    for rb in range(g.RB):
        if plan.per_row_block:
            live[rb] = plan.groups[rb]
            note()
        chunks = P.shim_rb_chunks(g, col, rb)
        # A row block without C awaits its A/B instead.
        n_entries[rb] = max(1, len(chunks))
        outstanding.append([rb, 0, []])
        for i in range(1, len(chunks)):
            await_all()
            outstanding.append([rb, i, []])
        if plan.per_row_block:
            outstanding[-1][2].append(rb)
        if rb > 0:
            await_all(keep=1)
    await_all()
    return peak


def _check_accepted(g):
    for col in range(g.n_aie_cols):
        plan = P.mem_plan(g, col)
        tasks, bds = plan.live()
        assert tasks <= P.MAX_QUEUED_TASKS
        assert bds <= P.MAX_LIVE_MEM_BDS
        assert all(
            1 <= t.repeat <= P.MAX_TASK_REPEAT for grp in plan.groups for t in grp
        )
        sim_tasks, sim_bds = _simulate_issue(g, col)
        assert sim_tasks <= tasks
        assert sim_bds <= bds


# Plan-level sweep: large N (LM heads), large M, many and odd column groups, aligned squares.
SWEEP_M = [1, 10, 100, 500, 1000, 1500, 2000, 2048, 2436, 3000, 4096]
SWEEP_N = [8, 100, 260, 500, 1000, 2000, 2436, 4096, 5000, 8192, 32000, 50257]
SWEEP_K = [64, 768, 4096]
SWEEP_SQUARES = [(s, s, s) for s in range(512, 4097, 512)]
# Long M with narrow N (conv layers): row blocks in the thousands.
SWEEP_LONG_M = [
    (M, N, K)
    for M in (49999, 98000, 150000, 392000)
    for N in (8, 16, 100, 260)
    for K in (9, 72)
]


@pytest.mark.parametrize("dev", list(DEVS))
def test_sweep_plans_fit_the_mem_tile_or_are_rejected(dev):
    """A plan either fits the mem tile's queue and live-BD budget or raises ValueError."""
    shapes = [(M, N, K) for M in SWEEP_M for N in SWEEP_N for K in SWEEP_K]
    shapes += SWEEP_SQUARES + SWEEP_LONG_M + SHAPES
    accepted = rejected = untileable = 0
    per_rb_seen = cg_gt16_seen = odd_cg_seen = long_rb_seen = 0
    for shape in shapes:
        try:
            g = _grid(dev, *shape)
        except ValueError:
            untileable += 1
            continue
        try:
            P.validate(g)
        except ValueError:
            rejected += 1
            continue
        _check_accepted(g)
        accepted += 1
        per_rb_seen += any(P.mem_plan(g, c).per_row_block for c in range(g.n_aie_cols))
        cg_gt16_seen += g.CG > 16
        odd_cg_seen += g.CG % 2 == 1 and g.CG > 8
        long_rb_seen += g.RB >= 1500
    print(f"{dev}: {accepted} accepted, {rejected} rejected, {untileable} not tileable")
    # The sweep must reach the regimes it is for.
    assert per_rb_seen
    assert cg_gt16_seen
    assert odd_cg_seen
    assert long_rb_seen
    assert accepted >= 0.9 * (accepted + rejected)


def test_sweep_shifted_plans_fit_the_mem_tile_or_are_rejected():
    """aie2 f32 B: a shifted plan fits the mem tile's queue and live-BD budget or raises."""
    shapes = [(M, N, K) for M in SWEEP_M for N in SWEEP_N for K in SWEEP_K]
    shapes += SWEEP_SQUARES + SWEEP_LONG_M + SHIFT_SHAPES
    accepted = rejected = untileable = 0
    rows_seen = cols_seen = both_seen = 0
    for shape in shapes:
        try:
            g = _f32_b_grid(*shape)
        except ValueError:
            untileable += 1
            continue
        try:
            P.validate(g)
        except ValueError:
            rejected += 1
            continue
        _check_accepted(g)
        accepted += 1
        rows_seen += g.shift_rows
        cols_seen += g.shift_cols
        both_seen += g.shift_rows and g.shift_cols
    print(
        f"npu f32 B: {accepted} accepted, {rejected} rejected, {untileable} not tileable"
    )
    assert rows_seen
    assert cols_seen
    assert both_seen
    assert accepted >= 0.9 * (accepted + rejected)


@pytest.mark.parametrize(
    ("dev", "shape"),
    [
        ("npu2", (4096, 4096, 4096)),
        ("npu2", (1000, 2000, 1000)),
        ("npu2", (1024, 4096, 1024)),
        ("npu", (1024, 4096, 1024)),
        ("npu2", (2048, 2048, 2048)),
        ("npu", (2048, 2048, 2048)),
    ],
)
def test_large_shapes_are_accepted(dev, shape):
    """Shapes base 2053fe89 ran on the NPU are accepted."""
    # One BD per object in an up-front chain once exceeded the mem-tile BD budget for these.
    g = _grid(dev, *shape)
    P.validate(g)
    _check_accepted(g)


def test_straddling_column_issues_per_row_block():
    """The column straddling N issues its mem tasks per row block."""
    # aie2p 1000x2000: column 6 straddles N (full objects, then one clipped by N) in every row
    # block, so it cannot repeat one chain across row blocks.
    g = _grid("npu2", 1000, 2000, 1000)
    plan = P.mem_plan(g, 6)
    assert plan.per_row_block
    assert len(plan.groups) == g.RB
    assert not P.mem_plan(g, 0).per_row_block


def test_uniform_run_is_one_short_chain():
    """A run of identical objects repeats one ping/pong unit."""
    # 4096^3 aie2p: 256 identical full objects per column, ping/pong -> a 2-BD chain x 128.
    g = _grid("npu2", 4096, 4096, 4096)
    (group,) = P.mem_plan(g, 0).groups
    assert [(t.buffers, t.repeat) for t in group] == [((0, 1), 128)]


def test_rejects_what_does_not_fit(monkeypatch):
    """A plan over the live-BD budget raises ValueError (CPU fallback)."""
    g = _grid("npu2", 1000, 2000, 1000)
    monkeypatch.setattr(P, "MAX_LIVE_MEM_BDS", 4)
    P.mem_plan.cache_clear()  # plans are cached per grid
    try:
        with pytest.raises(ValueError, match="live mem-tile"):
            P.validate(g)
    finally:
        P.mem_plan.cache_clear()
