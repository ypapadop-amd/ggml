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


def _mem_stream(g, col):
    """Dense destination index of every element the mem tile streams to the shim, in order."""
    produced = [o for o in P.column_objects(g, col) if o.cols]
    out, i = [], 0
    for task in P.mem_tasks(g, col):
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
        for task in P.mem_tasks(g, col)
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
    assert P.mem_tasks(g, 1) == []
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
    tasks = P.mem_tasks(g, 0)
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
