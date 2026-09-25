# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Model the shim buffer descriptors a de-pad-free GEMM would emit.

Fusing the de-pad away means the GEMM's C output descriptors write only the
real [M, N] sub-block of the padded (Mpad, Npad) tile grid. That is expressible
in a buffer descriptor or it is not, and the answer is decided entirely by
arithmetic that can be checked here rather than discovered as a wrong edge on
hardware.

What a descriptor can say (``npu_dma_memcpy_nd``): a 4-D box with *uniform*
sizes and strides. gemm.py emits one per (shim column, transfer block) for a
column-major C:

    sizes   = [col_blocks, n_aie_rows, n, m]
    strides = [M*n*n_aie_cols, m, M, 1]

Uniformity is the whole problem: the real boundary M can fall in the middle of
a row-block, and then one descriptor cannot describe both the full row-tiles
and the partial one. The saving grace is ordering -- row-tiles within a block
ascend in i, so the full ones are always a *prefix*, followed by at most one
partial, followed by tiles entirely in the pad region. Same in the column
direction. So each dimension splits at most once, giving at most 4 descriptors
where 1 is used today.

BD budget: the sequence uses bd_id_base+0 (C), +1 (A), +2 (B) of 8 per
pingpong, so 5 ids are free.

These tests check the generated descriptor set actually tiles the real output
exactly once, stays inside the stride range, and fits the budget.
"""

from __future__ import annotations

import numpy as np
import pytest

N_AIE_ROWS = 4
N_AIE_COLS = 8
DMA_MAX_STRIDE = 1 << 20
FREE_BD_IDS_PER_PINGPONG = 5


def _runs(count: int, tile: int, limit: int) -> tuple[int, int]:
    """Split ``count`` tiles of size ``tile`` against a real bound.

    Returns (n_full, partial) -- how many whole tiles fit entirely below
    ``limit``, and the size of the one partial tile (0 if none). Tiles beyond
    that are entirely in the pad region and are simply not transferred.
    """
    n_full = 0
    partial = 0
    for idx in range(count):
        start = idx * tile
        if start + tile <= limit:
            n_full += 1
        elif start < limit:
            partial = limit - start
            break
        else:
            break
    return n_full, partial


def clipped_c_descriptors(M, N, Mpad, Npad, m, n, row_block, col):
    """Descriptors writing one row-block's share of the real [M, N] output.

    Column-major C: element (i, j) lives at j*M + i. Yields dicts with the same
    shape as the npu_dma_memcpy_nd arguments.
    """
    col_blocks = Npad // (n * N_AIE_COLS)

    # Row tiles this (row_block) owns, in ascending i.
    i_of = lambda r: (row_block * N_AIE_ROWS + r) * m  # noqa: E731
    # Column tiles this shim column owns, in ascending j.
    j_of = lambda cb: (cb * N_AIE_COLS + col) * n  # noqa: E731

    # How far into this row-block the real output reaches.
    r_full, r_part = _runs(
        N_AIE_ROWS, m, max(0, M - i_of(0)) if M > i_of(0) else 0
    )
    # Column direction: tiles for this shim column are strided by n*N_AIE_COLS,
    # so evaluate each cb against N directly.
    cb_full = sum(1 for cb in range(col_blocks) if j_of(cb) + n <= N)
    cb_part = 0
    for cb in range(col_blocks):
        if j_of(cb) < N < j_of(cb) + n:
            cb_part = N - j_of(cb)
            break

    descriptors = []
    row_groups = []
    if r_full:
        row_groups.append((0, r_full, m))
    if r_part:
        row_groups.append((r_full, 1, r_part))
    col_groups = []
    if cb_full:
        col_groups.append((0, cb_full, n))
    if cb_part:
        col_groups.append((cb_full, 1, cb_part))

    for cb0, nb_cb, ncols in col_groups:
        for r0, nb_r, mrows in row_groups:
            descriptors.append(
                {
                    "offset": j_of(cb0) * M + i_of(r0),
                    "sizes": [nb_cb, nb_r, ncols, mrows],
                    "strides": [M * n * N_AIE_COLS, m, M, 1],
                }
            )
    return descriptors


def paint(desc, canvas):
    """Apply a descriptor to a flat canvas, counting writes (coverage)."""
    s0, s1, s2, s3 = desc["sizes"]
    d0, d1, d2, d3 = desc["strides"]
    base = desc["offset"]
    for a in range(s0):
        for b in range(s1):
            for c in range(s2):
                for d in range(s3):
                    canvas[base + a * d0 + b * d1 + c * d2 + d * d3] += 1


def pad_to(v, mult):
    return ((v + mult - 1) // mult) * mult


# (M, N, m, n): M/N real; Mpad/Npad derived with the aie2p bf16 granularity.
CASES = [
    (500, 500, 32, 64),    # mnist fc1: boundary lands mid row-block and mid column tile
    (500, 500, 8, 16),     # same shape, smallest tile
    (512, 512, 32, 64),    # already aligned: must degenerate to today's single descriptor
    (10, 500, 8, 16),      # mnist fc2: M smaller than one row-block
    (400, 300, 8, 16),     # both boundaries interior
    (129, 130, 8, 16),     # awkward primes-ish
]


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_clipped_descriptors_cover_real_output_exactly_once(M, N, m, n):
    """The descriptor set must tile [0,M)x[0,N) once: no gap, no overlap."""
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    row_blocks = Mpad // (m * N_AIE_ROWS)
    canvas = np.zeros(M * N, dtype=np.int64)

    for rb in range(row_blocks):
        for col in range(N_AIE_COLS):
            for desc in clipped_c_descriptors(M, N, Mpad, Npad, m, n, rb, col):
                paint(desc, canvas)

    written = canvas.reshape(N, M).T  # column-major: j*M + i
    np.testing.assert_array_equal(
        written, np.ones((M, N), dtype=np.int64)
    )


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_descriptor_count_fits_the_bd_budget(M, N, m, n):
    """At most 4 descriptors per (row-block, column); 5 ids are free."""
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    row_blocks = Mpad // (m * N_AIE_ROWS)
    worst = 0
    for rb in range(row_blocks):
        for col in range(N_AIE_COLS):
            worst = max(
                worst, len(clipped_c_descriptors(M, N, Mpad, Npad, m, n, rb, col))
            )
    assert worst <= FREE_BD_IDS_PER_PINGPONG, (
        f"{worst} descriptors needed, only {FREE_BD_IDS_PER_PINGPONG} BD ids free"
    )


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_clipped_strides_stay_in_range(M, N, m, n):
    """Clipping only shrinks strides (M_real <= Mpad), but assert it."""
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    row_blocks = Mpad // (m * N_AIE_ROWS)
    for rb in range(row_blocks):
        for col in range(N_AIE_COLS):
            for desc in clipped_c_descriptors(M, N, Mpad, Npad, m, n, rb, col):
                for stride in desc["strides"]:
                    assert 1 <= stride <= DMA_MAX_STRIDE, desc


def test_aligned_shape_needs_only_one_descriptor():
    """When nothing is padded the scheme must collapse to today's descriptor."""
    descs = clipped_c_descriptors(512, 512, 512, 512, m=32, n=64, row_block=0, col=0)
    assert len(descs) == 1
    assert descs[0]["sizes"] == [1, N_AIE_ROWS, 64, 32]
