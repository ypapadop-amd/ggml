# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Shim descriptors for a de-pad-free GEMM -- correct arithmetic, and why it is
not sufficient on hardware.

The idea: fusing the de-pad away by having the GEMM's C descriptors write only
the real [M, N] sub-block of the padded (Mpad, Npad) tile grid, so a padded
MUL_MAT stops being three dispatches. The arithmetic below is right, and these
tests prove it -- the descriptor set tiles the real output exactly once, fits
the buffer-descriptor budget, and collapses to today's single descriptor when
nothing is padded.

WHAT THIS MODEL MISSES, measured on aie2p 2026-09-25
----------------------------------------------------
Wiring these descriptors into gemm.py and running them produced ~6.5-7.4% wrong
elements on every shape that actually clips (M=496/500/504/508, N=500), while
every shape needing no clipping (M=256/480, 512^3) stayed bit-exact. The wrong
elements read as zero at the partial tile and the corruption spread past the
boundary, bounded per transfer block.

The reason is an invariant this file does not model: **the shim must drain
exactly the objects the herd produced.** Production volume is fixed by the
padded tile grid -- every core produces its full (m, n) tiles whether or not
they fall inside [M, N] -- so a descriptor set that transfers less leaves
objects undrained and the rest of that transfer block lands shifted. The
``dma_wait`` between transfer blocks is what keeps the damage bounded instead
of total.

That is a source-side invariant; everything checked here is destination-side
coverage. A model that only asks "does the write cover the output exactly
once?" cannot see it, which is precisely why this passed and hardware did not.

So de-padding is NOT expressible as clipped descriptors alone. Making it work
needs the pad-region objects drained somewhere harmless -- a scratch buffer
passed as an extra kernel argument, with its own descriptors -- which keeps the
single dispatch but costs an ABI change and more BD ids. The arithmetic here is
the part that would be reused.

SECOND ATTEMPT, ALSO REVERTED: phased de-pad in one dispatch
------------------------------------------------------------
The follow-up avoided the drain invariant entirely. The herd still filled the
padded destination (so the shim drained exactly what the cores produced), and a
second phase of the *same* runtime sequence copied the real sub-block out of it
-- a pure strided copy, N_out runs of M_out contiguous elements, needing no
compute tile. The caller's unpadded buffer was passed as a third kernel source;
gemm.py grew shim -> memtile -> shim forward fifos per column.

aiecc rejected it:

    'aie.tile' op number of output DMA channel exceeded!

Each shim tile already carries A (columns 0-3), B and C. Two more fifos per
column exceeds the shim's DMA channel budget, and unlike program memory there
is no slack to find -- the channels are the resource.

This is why upstream's multi-phase examples (ml/scale_shift,
ml/mm_activation_epilogue) reuse *the same* ObjectFifos across phases and vary
only the RTP word and the DMA addressing. Any workable version has to route
phase 2 over the fifos that already exist rather than adding any, which means
matching their object types (B carries bf16, C carries f32) or accepting a
compute tile in the path.

Both attempts were blocked by a resource or invariant that destination-side
coverage cannot see. That is the durable lesson: for this dataflow, simulate
what the *hardware* must supply -- drain volume, DMA channels, program memory --
not just whether the output ends up in the right place.
"""

from __future__ import annotations

import numpy as np
import pytest

from presim import pad_to

N_AIE_ROWS = 4
N_AIE_COLS = 8
DMA_MAX_STRIDE = 1 << 20
# C keeps bd_id_base and can draw base+3..base+7; A and B take base+1 and +2.
FREE_BD_IDS_PER_PINGPONG = 6


def clipped_c_descriptors(
    m_out, n_out, m, n, n_aie_rows, n_aie_cols, col_blocks, row_block, col
):
    """Descriptors writing one (row_block, col)'s share of an unpadded C.

    A descriptor is a 4-D box with *uniform* sizes, so a boundary falling inside
    a row-block cannot be described by one. It does not have to be: row tiles
    within a block ascend in i, so the whole ones form a prefix, then at most one
    partial, then tiles lying entirely in the pad region. The column direction
    behaves the same way, so each dimension splits at most once.

    C is column-major: element (i, j) lives at j * m_out + i.

    Returns:
        A list of (offset, sizes, strides) tuples shaped for npu_dma_memcpy_nd.
    """

    def row_start(r):
        return (row_block * n_aie_rows + r) * m

    def col_start(cb):
        return (cb * n_aie_cols + col) * n

    r_full = sum(1 for r in range(n_aie_rows) if row_start(r) + m <= m_out)
    r_part = next(
        (
            m_out - row_start(r)
            for r in range(n_aie_rows)
            if row_start(r) < m_out < row_start(r) + m
        ),
        0,
    )
    cb_full = sum(1 for cb in range(col_blocks) if col_start(cb) + n <= n_out)
    cb_part = next(
        (
            n_out - col_start(cb)
            for cb in range(col_blocks)
            if col_start(cb) < n_out < col_start(cb) + n
        ),
        0,
    )

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

    return [
        (
            col_start(cb0) * m_out + row_start(r0),
            [nb_cb, nb_r, n_cols, n_rows],
            [m_out * n * n_aie_cols, m, m_out, 1],
        )
        for cb0, nb_cb, n_cols in col_groups
        for r0, nb_r, n_rows in row_groups
    ]


def paint(desc, canvas):
    """Apply a descriptor to a flat canvas, counting writes (coverage)."""
    base, (s0, s1, s2, s3), (d0, d1, d2, d3) = desc
    for a in range(s0):
        for b in range(s1):
            for c in range(s2):
                for d in range(s3):
                    canvas[base + a * d0 + b * d1 + c * d2 + d * d3] += 1


def drained_elements(descs):
    """Elements a descriptor set transfers -- the source-side quantity."""
    return sum(np.prod(sizes) for _, sizes, _ in descs)


# (M, N, m, n): M/N real; Mpad/Npad derived with the aie2p bf16 granularity.
CASES = [
    (500, 500, 32, 64),    # mnist fc1: boundary mid row-block and mid column tile
    (500, 500, 8, 16),     # same shape, smallest tile
    (512, 512, 32, 64),    # already aligned: must degenerate to one descriptor
    (10, 500, 8, 16),      # mnist fc2: M smaller than one row-block
    (400, 300, 8, 16),     # both boundaries interior
    (129, 130, 8, 16),     # awkward
]


def _descs_for(M, N, m, n, rb, col, Npad):
    return clipped_c_descriptors(
        M, N, m, n, N_AIE_ROWS, N_AIE_COLS, Npad // (n * N_AIE_COLS), rb, col
    )


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_clipped_descriptors_cover_real_output_exactly_once(M, N, m, n):
    """The descriptor set tiles [0,M)x[0,N) once: no gap, no overlap."""
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    canvas = np.zeros(M * N, dtype=np.int64)
    for rb in range(Mpad // (m * N_AIE_ROWS)):
        for col in range(N_AIE_COLS):
            for desc in _descs_for(M, N, m, n, rb, col, Npad):
                paint(desc, canvas)
    written = canvas.reshape(N, M).T  # column-major: j*M + i
    np.testing.assert_array_equal(written, np.ones((M, N), dtype=np.int64))


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_descriptor_count_fits_the_bd_budget(M, N, m, n):
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    worst = max(
        len(_descs_for(M, N, m, n, rb, col, Npad))
        for rb in range(Mpad // (m * N_AIE_ROWS))
        for col in range(N_AIE_COLS)
    )
    assert worst <= FREE_BD_IDS_PER_PINGPONG


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_clipped_strides_stay_in_range(M, N, m, n):
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    for rb in range(Mpad // (m * N_AIE_ROWS)):
        for col in range(N_AIE_COLS):
            for desc in _descs_for(M, N, m, n, rb, col, Npad):
                for stride in desc[2]:
                    assert 1 <= stride <= DMA_MAX_STRIDE, desc


def test_aligned_shape_needs_only_one_descriptor():
    """With nothing padded the scheme collapses to today's descriptor."""
    descs = clipped_c_descriptors(512, 512, 32, 64, N_AIE_ROWS, N_AIE_COLS, 1, 0, 0)
    assert len(descs) == 1
    assert descs[0][1] == [1, N_AIE_ROWS, 64, 32]


@pytest.mark.parametrize(("M", "N", "m", "n"), CASES)
def test_clipping_under_drains_the_object_fifo(M, N, m, n):
    """The invariant hardware enforced and destination coverage cannot see.

    The herd produces the full padded tile grid regardless of [M, N], so a
    clipped descriptor set transfers fewer elements than were produced. Exactly
    the shapes where this deficit is non-zero are the shapes that miscomputed on
    device; where it is zero, the design was bit-exact.

    Asserting the deficit here keeps the finding attached to the code that
    caused it: any future scheme must drain the difference somewhere (a scratch
    buffer) rather than simply transferring less.
    """
    Mpad, Npad = pad_to(M, m * N_AIE_ROWS), pad_to(N, n * N_AIE_COLS)
    produced = drained = 0
    for rb in range(Mpad // (m * N_AIE_ROWS)):
        for col in range(N_AIE_COLS):
            produced += (Npad // (n * N_AIE_COLS)) * N_AIE_ROWS * n * m
            drained += drained_elements(_descs_for(M, N, m, n, rb, col, Npad))

    assert drained <= produced
    aligned = (M, N) == (Mpad, Npad)
    if aligned:
        assert drained == produced, "aligned shapes must drain the grid exactly"
    else:
        assert drained < produced, (
            "a clipped shape under-drains; the deficit is what must go to scratch"
        )
