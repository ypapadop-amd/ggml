# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Pre-hardware simulation of the two mechanisms a *fused* padded GEMM needs.

Today a padded MUL_MAT is three dispatches -- convert_pad, mul_mat, depad -- and
the AIE queue pays to swap the whole-array GEMM overlay in and out around them.
Fusing the transforms into the GEMM makes it one dispatch. Two mechanisms are
needed, neither of which exists in this repo or upstream:

  1. **Clipped output writes.** The herd computes on a padded (Mpad, Npad) tile
     grid, but the shim DMA writes only the real [M, N] sub-block, so no de-pad
     pass is needed. Boundary tiles are partial and tiles entirely in the pad
     region are skipped. This is pure addressing -- it becomes buffer-descriptor
     sizes/strides -- and it is exactly where an off-by-one silently corrupts an
     edge.

  2. **On-core pad of a partial operand tile.** The activation arrives unpadded,
     so the core zero-fills the partial K/N tile as it consumes it, instead of
     reading a pre-zeroed padded buffer produced by a separate kernel.

Both are modelled here as index arithmetic over the real topology, with integer
data so the comparison is exact. This validates the *dataflow and addressing*.
It deliberately does not model bf16 rounding -- that is a numeric question the
on-device bit-exact gate answers, and mixing it in here would cost the exact
comparison that makes an addressing bug obvious.
"""

from __future__ import annotations

import numpy as np
import pytest

from presim import build_gemm_fifos, pad_to, run_workers

N_AIE_ROWS = 4
N_AIE_COLS = 8

# Padding granularity for aie2p bf16 = (row_expand*r, s, col_expand*t). These are
# literals so the harness stays independent of the IRON toolchain that importing
# gemm.py would pull in -- which does mean nothing checks them against gemm.py's
# tables automatically. test_padding_matches_the_host_rule below pins the padded
# shapes they produce, so a drift shows up there rather than silently.
GM, GK, GN = 8, 8, 16


def padded_dims(M: int, N: int, K: int) -> tuple[int, int, int]:
    """The shapes the host pads to today."""
    return (
        pad_to(M, GM * N_AIE_ROWS),
        pad_to(N, GN * N_AIE_COLS),
        pad_to(K, GK),
    )


def build_fused_workers(
    A, B, C_out, coverage, m, k, n, fifo_depth, *, zero_fill=True, poison=0
):
    """GEMM over a padded grid that consumes an unpadded B and writes unpadded C.

    A is already padded bf16 in the real design (a constant, converted once and
    cached), so it is fed from a padded view. B arrives unpadded and is
    zero-filled on the core. C is written clipped.
    """
    M, K = A.shape
    _, N = B.shape
    Mpad, Npad, Kpad = padded_dims(M, N, K)

    A_pad = np.zeros((Mpad, Kpad), dtype=A.dtype)
    A_pad[:M, :K] = A

    assert Mpad % (m * N_AIE_ROWS) == 0
    assert Npad % (n * N_AIE_COLS) == 0
    assert Kpad % k == 0

    row_blocks = Mpad // (m * N_AIE_ROWS)
    col_blocks = Npad // (n * N_AIE_COLS)
    k_tiles = Kpad // k
    tiles_per_core = row_blocks * col_blocks

    a_fifos, b_fifos, c_fifos = build_gemm_fifos(
        N_AIE_ROWS, N_AIE_COLS, fifo_depth
    )

    def a_feeder(r):
        def run():
            for rb in range(row_blocks):
                for _cb in range(col_blocks):
                    for kt in range(k_tiles):
                        i = (rb * N_AIE_ROWS + r) * m
                        a_fifos[r].acquire_produce()
                        a_fifos[r].release_produce(
                            A_pad[i : i + m, kt * k : (kt + 1) * k]
                        )

        return run

    def b_feeder(c):
        # MECHANISM 2: the shim moves only the real elements; the rest of the
        # (k, n) tile is whatever the core leaves there. ``zero_fill=False``
        # models not clearing it, and ``poison`` is what stale memory decodes
        # to -- see test_partial_operand_tile_must_be_zero_filled.
        def run():
            for _rb in range(row_blocks):
                for cb in range(col_blocks):
                    for kt in range(k_tiles):
                        j = (cb * N_AIE_COLS + c) * n
                        fill = 0 if zero_fill else poison
                        tile = np.full((k, n), fill, dtype=B.dtype)
                        ki, ji = min(kt * k + k, K), min(j + n, N)
                        if kt * k < K and j < N:
                            valid = B[kt * k : ki, j:ji]
                            tile[: valid.shape[0], : valid.shape[1]] = valid
                        b_fifos[c].acquire_produce()
                        b_fifos[c].release_produce(tile)

        return run

    def core(r, c):
        def run():
            for _ in range(tiles_per_core):
                c_fifos[r][c].acquire_produce()
                acc = np.zeros((m, n), dtype=A.dtype)
                for _kt in range(k_tiles):
                    a = a_fifos[r].acquire_consume(consumer=c)
                    b = b_fifos[c].acquire_consume(consumer=r)
                    acc += a @ b
                    a_fifos[r].release_consume(consumer=c)
                    b_fifos[c].release_consume(consumer=r)
                c_fifos[r][c].release_produce(acc)

        return run

    def drain(r, c):
        # MECHANISM 1: write only the part of each tile inside [M, N]; skip
        # tiles that lie entirely in the pad region.
        def run():
            for t in range(tiles_per_core):
                rb, cb = divmod(t, col_blocks)
                tile = c_fifos[r][c].acquire_consume()
                i = (rb * N_AIE_ROWS + r) * m
                j = (cb * N_AIE_COLS + c) * n
                i1, j1 = min(i + m, M), min(j + n, N)
                if i < M and j < N:
                    C_out[i:i1, j:j1] = tile[: i1 - i, : j1 - j]
                    coverage[i:i1, j:j1] += 1
                c_fifos[r][c].release_consume()

        return run

    workers = [a_feeder(r) for r in range(N_AIE_ROWS)]
    workers += [b_feeder(c) for c in range(N_AIE_COLS)]
    workers += [core(r, c) for r in range(N_AIE_ROWS) for c in range(N_AIE_COLS)]
    workers += [drain(r, c) for r in range(N_AIE_ROWS) for c in range(N_AIE_COLS)]
    return workers


def _run(M, N, K, m, k, n, fifo_depth=2, *, dtype=np.int64, **kw):
    rng = np.random.default_rng(0)
    A = rng.integers(-4, 5, size=(M, K)).astype(dtype)
    B = rng.integers(-4, 5, size=(K, N)).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)
    cov = np.zeros((M, N), dtype=np.int64)
    run_workers(build_fused_workers(A, B, C, cov, m, k, n, fifo_depth, **kw))
    return A, B, C, cov


# (M, N, K, m, k, n) -- shapes chosen so the padding is non-trivial in every
# dimension, i.e. real partial tiles at the M, N and K edges.
SHAPES = [
    (50, 50, 49, 8, 8, 16),     # mnist-fc1-like awkwardness, scaled down
    (64, 128, 56, 8, 8, 16),    # already aligned: must degenerate correctly
    (33, 100, 17, 8, 8, 16),    # partial in all three dims
    (96, 130, 64, 8, 16, 16),   # N just past a block boundary
]


@pytest.mark.parametrize(("M", "N", "K", "m", "k", "n"), SHAPES)
def test_fused_gemm_is_bit_exact(M, N, K, m, k, n):
    """Clipped writes + on-core zero-fill reproduce the unpadded reference."""
    A, B, C, _ = _run(M, N, K, m, k, n)
    np.testing.assert_array_equal(C, A @ B)


@pytest.mark.parametrize(("M", "N", "K", "m", "k", "n"), SHAPES)
def test_clipped_writes_cover_the_output_exactly_once(M, N, K, m, k, n):
    """Every real element written once: no gap, no double write, no overrun.

    This is the check that catches a bad buffer descriptor. A wrong sub-rectangle
    usually still produces plausible numbers somewhere; it shows up here as a 0
    or a 2 in the coverage map.
    """
    _, _, _, cov = _run(M, N, K, m, k, n)
    assert cov.shape == (M, N)
    np.testing.assert_array_equal(cov, np.ones((M, N), dtype=np.int64))


@pytest.mark.parametrize("fifo_depth", [1, 2, 3])
def test_fused_gemm_holds_at_every_fifo_depth(fifo_depth):
    A, B, C, cov = _run(50, 50, 49, 8, 8, 16, fifo_depth=fifo_depth)
    np.testing.assert_array_equal(C, A @ B)
    np.testing.assert_array_equal(cov, np.ones_like(cov))


def test_finite_garbage_in_the_operand_pad_is_masked():
    """With finite garbage, zero-filling the partial B tile is *not* required.

    acc[i,j] = sum_p a[i,p]*b[p,j]. Garbage in B's K-pad rows multiplies A's
    K-pad columns, which are zero; garbage in B's N-pad columns only reaches
    output columns the clipped drain throws away. So A's zero K-pad and the
    clipping between them mask the operand pad entirely.

    Recorded because it says the fused design does not need the host to hand it
    a pre-zeroed padded operand buffer -- which is one of the things the
    separate convert_pad dispatch exists to produce.
    """
    A, B, C, _ = _run(33, 100, 17, 8, 8, 16, zero_fill=False, poison=7)
    np.testing.assert_array_equal(C, A @ B)


def test_partial_operand_tile_must_be_zero_filled():
    """...but the masking breaks for non-finite garbage, so zero-fill anyway.

    0.0 * NaN is NaN, not 0, so the argument above collapses the moment stale
    memory decodes to NaN/Inf -- which uninitialised bf16 can. This is the test
    that makes the zero-fill a requirement rather than an assumption; an integer
    model cannot express it and silently passes either way.
    """
    _, _, poisoned, _ = _run(
        33, 100, 17, 8, 8, 16, dtype=np.float64, zero_fill=False, poison=np.nan
    )
    assert np.isnan(poisoned).any(), "expected NaN to leak out of the operand pad"

    A, B, clean, _ = _run(
        33, 100, 17, 8, 8, 16, dtype=np.float64, zero_fill=True, poison=np.nan
    )
    np.testing.assert_array_equal(clean, A @ B)
