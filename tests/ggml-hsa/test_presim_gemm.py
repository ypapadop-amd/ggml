# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Pre-hardware simulation of the whole-array GEMM dataflow.

Models the ObjectFifo topology and core body that ``iron_kernels/gemm.py``
actually builds, and checks it against numpy. The point is not to re-test the
matmul -- hardware does that -- but to have a place where a dataflow change
(a different FIFO depth, a new pipeline stage, a reordered feed) can be shown
to neither deadlock nor miscompute *before* anything is compiled.

Validating today's working topology first is what makes the harness
trustworthy: a mock that cannot reproduce the design that already works on
hardware is not evidence about a design that does not exist yet.

Topology mirrored from gemm.py:
  * A is tiled (m, k); each block is broadcast across the columns of a row and
    distributed across rows, so A_l2l1[row] has n_aie_cols consumers.
  * B is tiled (k, n); broadcast across rows, distributed across columns, so
    B_l2l1[col] has n_aie_rows consumers.
  * Each core accumulates one (m, n) output tile over K//k steps, then releases
    it; the feed order is (row_block, col_block, k_tile), matching the
    runtime_sequence's transfer blocks with their stride-0 outer A repeat.
"""

from __future__ import annotations

import numpy as np
import pytest

from presim import DeadlockError, ObjectFifo, run_workers

N_AIE_ROWS = 4
N_AIE_COLS = 8


def build_gemm_workers(A, B, C_out, m, k, n, fifo_depth, *, starve_a=False):
    """Wire the GEMM topology; return the worker callables.

    ``starve_a`` deliberately drops the last A tile, to prove the harness
    detects a starved FIFO instead of hanging.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M % (m * N_AIE_ROWS) == 0
    assert N % (n * N_AIE_COLS) == 0
    assert K % k == 0

    row_blocks = M // (m * N_AIE_ROWS)
    col_blocks = N // (n * N_AIE_COLS)
    k_tiles = K // k
    tiles_per_core = row_blocks * col_blocks

    a_fifos = [
        ObjectFifo(f"A_l2l1[{r}]", fifo_depth, n_consumers=N_AIE_COLS)
        for r in range(N_AIE_ROWS)
    ]
    b_fifos = [
        ObjectFifo(f"B_l2l1[{c}]", fifo_depth, n_consumers=N_AIE_ROWS)
        for c in range(N_AIE_COLS)
    ]
    c_fifos = [
        [ObjectFifo(f"C_l1l2[{r}][{c}]", fifo_depth) for c in range(N_AIE_COLS)]
        for r in range(N_AIE_ROWS)
    ]

    workers = []

    def a_feeder(r):
        def run():
            emitted = 0
            total = row_blocks * col_blocks * k_tiles
            for rb in range(row_blocks):
                for _cb in range(col_blocks):
                    for kt in range(k_tiles):
                        if starve_a and emitted == total - 1:
                            return
                        i = (rb * N_AIE_ROWS + r) * m
                        a_fifos[r].acquire_produce()
                        a_fifos[r].release_produce(A[i : i + m, kt * k : (kt + 1) * k])
                        emitted += 1

        return run

    def b_feeder(c):
        def run():
            for _rb in range(row_blocks):
                for cb in range(col_blocks):
                    for kt in range(k_tiles):
                        j = (cb * N_AIE_COLS + c) * n
                        b_fifos[c].acquire_produce()
                        b_fifos[c].release_produce(B[kt * k : (kt + 1) * k, j : j + n])

        return run

    def core(r, c):
        # Mirrors gemm.py's @core body: acquire C, zero it, accumulate over K,
        # release C.
        def run():
            for _ in range(tiles_per_core):
                c_fifos[r][c].acquire_produce()
                acc = np.zeros((m, n), dtype=np.int64)
                for _kt in range(k_tiles):
                    a = a_fifos[r].acquire_consume(consumer=c)
                    b = b_fifos[c].acquire_consume(consumer=r)
                    acc += a.astype(np.int64) @ b.astype(np.int64)
                    a_fifos[r].release_consume(consumer=c)
                    b_fifos[c].release_consume(consumer=r)
                c_fifos[r][c].release_produce(acc)

        return run

    def drain(r, c):
        def run():
            for t in range(tiles_per_core):
                rb, cb = divmod(t, col_blocks)
                tile = c_fifos[r][c].acquire_consume()
                i = (rb * N_AIE_ROWS + r) * m
                j = (cb * N_AIE_COLS + c) * n
                C_out[i : i + m, j : j + n] = tile
                c_fifos[r][c].release_consume()

        return run

    workers += [a_feeder(r) for r in range(N_AIE_ROWS)]
    workers += [b_feeder(c) for c in range(N_AIE_COLS)]
    workers += [core(r, c) for r in range(N_AIE_ROWS) for c in range(N_AIE_COLS)]
    workers += [drain(r, c) for r in range(N_AIE_ROWS) for c in range(N_AIE_COLS)]
    return workers


def _problem(M=64, N=128, K=32, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, size=(M, K), dtype=np.int64)
    B = rng.integers(-4, 5, size=(K, N), dtype=np.int64)
    return A, B


@pytest.mark.parametrize("fifo_depth", [1, 2, 3])
def test_gemm_topology_is_bit_exact(fifo_depth):
    """Today's topology computes A@B exactly, at every FIFO depth."""
    A, B = _problem()
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    run_workers(build_gemm_workers(A, B, C, m=8, k=8, n=16, fifo_depth=fifo_depth))
    np.testing.assert_array_equal(C, A @ B)


@pytest.mark.parametrize(("m", "k", "n"), [(8, 8, 16), (16, 8, 16), (8, 16, 16)])
def test_gemm_topology_across_tile_shapes(m, k, n):
    """The selector may pick any valid tile; the dataflow must hold for each."""
    A, B = _problem(M=128, N=128, K=32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    run_workers(build_gemm_workers(A, B, C, m=m, k=k, n=n, fifo_depth=2))
    np.testing.assert_array_equal(C, A @ B)


def test_starved_fifo_is_reported_not_hung():
    """A missing tile must fail fast and name the FIFO, not hang the suite."""
    A, B = _problem()
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    workers = build_gemm_workers(
        A, B, C, m=8, k=8, n=16, fifo_depth=2, starve_a=True
    )
    with pytest.raises(DeadlockError) as excinfo:
        run_workers(workers, timeout=15.0)
    assert "A_l2l1" in str(excinfo.value) or "C_l1l2" in str(excinfo.value)
