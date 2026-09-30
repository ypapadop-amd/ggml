# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for the GEMM's dense-destination C plan (``gemm_c_plan``).

The simulation test replays the plan at the address level. Every element the mem tiles read is
mapped to the dense destination index it belongs to. That sequence must equal the sequence of
indices the shim BDs write, and the union over all columns must be exactly [0, M*N). This catches
wrong offsets, strides, stream order, ping/pong parity and coverage without an NPU.
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
                    row = o.rb * g.rows_per_block + core * g.m + ii
                    colg = (o.cg * g.n_aie_cols + col) * g.n + jj
                    out.append(colg * g.M + row)
                i += 1
    assert i == len(produced), f"tasks cover {i} of {len(produced)} objects"
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _shim_stream(g, col):
    out = []
    for rb in range(g.RB):
        for chunk in P.shim_rb_chunks(g, col, rb):
            assert len(chunk) <= len(P.SHIM_C_BD_IDS[0])
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


@pytest.mark.parametrize("dev", list(DEVS))
@pytest.mark.parametrize("shape", SHAPES)
def test_streams_pair_up_and_cover_dense_c_exactly_once(dev, shape):
    try:
        g = _grid(dev, *shape)
    except ValueError as e:
        pytest.skip(f"not tileable on {dev}: {e}")
    written = []
    for col in range(g.n_aie_cols):
        mem, shim = _mem_stream(g, col), _shim_stream(g, col)
        assert np.array_equal(mem, shim), (
            f"column {col}: mem read order != shim write order"
        )
        written.append(shim)
    allw = np.sort(np.concatenate(written))
    assert np.array_equal(allw, np.arange(g.M * g.N)), (
        "dense C not covered exactly once"
    )


@pytest.mark.parametrize("dev", list(DEVS))
@pytest.mark.parametrize("shape", SHAPES)
def test_every_core_in_a_column_sends_once_per_produced_object(dev, shape):
    try:
        g = _grid(dev, *shape)
    except ValueError as e:
        pytest.skip(f"not tileable on {dev}: {e}")
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
