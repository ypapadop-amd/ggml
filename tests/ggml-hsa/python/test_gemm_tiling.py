# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Unit tests for the GEMM per-core tile selector (``select_gemm_tile``).

These are pure-Python and need no NPU: they check the invariants the AIE
whole-array GEMM design relies on, so a bad tile is caught here instead of as a
miscompute or an ``aiecc`` failure on device.
"""

import math
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

KERNELS_DIR = Path(__file__).resolve().parents[3] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

from iron_kernels.gemm import (  # noqa: E402
    DMA_MAX_STRIDE,
    L1_TILE_BUDGET_BYTES,
    microkernel_expansion_map,
    microkernel_mac_dim_map,
    resolve_expansion,
    resolve_mac_dims,
    select_gemm_tile,
)

BF16 = np.dtype(ml_dtypes.bfloat16)
F32 = np.dtype(np.float32)

# Input dtype name -> (numpy input dtype, numpy output dtype) as the GEMM uses them.
DTYPES = {
    "bf16": (BF16, F32),
    "i8": (np.dtype(np.int8), np.dtype(np.int8)),
    "i16": (np.dtype(np.int16), np.dtype(np.int16)),
}

# GEMM output dtypes. The microkernel always consumes bf16, but the destination
# varies: ggml's native MUL_MAT output is f32 (what create_mat_mul_external_functions
# actually passes), while graph_optimize can retype it to bf16 so the de-pad narrows
# in one pass. The L1 budget depends on the output element size, so both must be
# covered -- testing only bf16->bf16 would exercise a combination production never uses.
OUT_DTYPES = [BF16, F32]

# Upper bound on a single tile dimension. Passed explicitly to select_gemm_tile so
# the maximality oracle below and the implementation always search the same space.
MAX_TILE = 256

# Devices under test, with the herd width select_gemm_tile derives internally.
DEVICES = [("npu", 4), ("npu2", 8)]
DEVICE_NAMES = [dev for dev, _ in DEVICES]
N_AIE_ROWS = 4

# Square shapes the matmul benchmark drives, plus non-square and awkward ones.
SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (512, 1024, 2048),
    (1024, 512, 512),
    (2048, 512, 1024),
]


def _tile(dev, M, N, K, dtype_out=F32, dtype_in_str="bf16"):
    """Return ``(r, s, t, m, k, n)`` for a device/shape: MAC dims plus selected tile."""
    r, s, t = resolve_mac_dims(dev, dtype_in_str)
    row_expand, col_expand = resolve_expansion(dev, dtype_in_str)
    dtype_in = DTYPES[dtype_in_str][0]
    m, k, n = select_gemm_tile(
        dev,
        M,
        N,
        K,
        dtype_in,
        dtype_out,
        r,
        s,
        t,
        row_expand,
        col_expand,
        max_tile=MAX_TILE,
    )
    return r, s, t, m, k, n


def _granular_minimum(dev, dtype_in_str="bf16"):
    """Return the smallest microkernel-granular tile the search can consider."""
    r, s, t = resolve_mac_dims(dev, dtype_in_str)
    row_expand, col_expand = resolve_expansion(dev, dtype_in_str)
    return row_expand * r, s, col_expand * t


def _working_set(m, k, n, dtype_out=F32, fifo_depth=2):
    """Bytes the double-buffered A/B/C object-FIFO tiles occupy in one core's L1."""
    return fifo_depth * (
        BF16.itemsize * m * k + BF16.itemsize * k * n + dtype_out.itemsize * m * n
    )


def _is_valid(n_aie_cols, M, N, K, m, k, n, dtype_out=F32):
    """True if (m, k, n) satisfies every constraint ``select_gemm_tile`` enforces."""
    if M % (m * N_AIE_ROWS) or K % k or N % (n * n_aie_cols):
        return False
    if ((M // m) * (N // n)) % (N_AIE_ROWS * n_aie_cols):
        return False
    if M * n * n_aie_cols > DMA_MAX_STRIDE or n * n_aie_cols * K > DMA_MAX_STRIDE:
        return False
    return _working_set(m, k, n, dtype_out) <= L1_TILE_BUDGET_BYTES


@pytest.mark.parametrize("dev", DEVICE_NAMES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tile_satisfies_microkernel_granularity(dev, M, N, K):
    """The tile must be a whole number of 2x2-expanded mmul instructions."""
    r, s, t, m, k, n = _tile(dev, M, N, K)
    assert m % (2 * r) == 0, f"m={m} not a multiple of 2*r={2 * r}"
    assert k % s == 0, f"k={k} not a multiple of s={s}"
    assert n % (2 * t) == 0, f"n={n} not a multiple of 2*t={2 * t}"


@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tile_tiles_the_array_evenly(dev, n_aie_cols, M, N, K):
    """A/B/C must decompose across the herd with every core getting equal work."""
    _, _, _, m, k, n = _tile(dev, M, N, K)
    assert M % (m * N_AIE_ROWS) == 0
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0
    assert ((M // m) * (N // n)) % (N_AIE_ROWS * n_aie_cols) == 0


@pytest.mark.parametrize("dtype_out", OUT_DTYPES, ids=["out_bf16", "out_f32"])
@pytest.mark.parametrize("dev", DEVICE_NAMES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tile_fits_l1_budget(dev, M, N, K, dtype_out):
    """Double-buffered A/B/C tiles must fit the per-core L1 data-memory budget."""
    _, _, _, m, k, n = _tile(dev, M, N, K, dtype_out)
    ws = _working_set(m, k, n, dtype_out)
    assert ws <= L1_TILE_BUDGET_BYTES, (
        f"working set {ws} exceeds {L1_TILE_BUDGET_BYTES}"
    )


@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tile_respects_dma_stride_limit(dev, n_aie_cols, M, N, K):
    """Column-major B/C shim transfers must stay inside the BD stride range.

    Exceeding it is not a miscompute but a hard aiecc failure
    ("Stride N exceeds the [1:1048576] range"), so the selector must exclude it.
    """
    _, _, _, m, k, n = _tile(dev, M, N, K)
    assert M * n * n_aie_cols <= DMA_MAX_STRIDE
    assert n * n_aie_cols * K <= DMA_MAX_STRIDE


@pytest.mark.parametrize("dtype_out", OUT_DTYPES, ids=["out_bf16", "out_f32"])
@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_tile_is_volume_maximal(dev, n_aie_cols, M, N, K, dtype_out):
    """No larger-volume valid tile exists than the one returned.

    Guards the search itself: a selector that silently returned the *first*
    valid tile rather than the best one would still pass every other test here.
    """
    r, s, t, m, k, n = _tile(dev, M, N, K, dtype_out)
    chosen_key = (m * k * n, m * n, k)

    gm, gk, gn = 2 * r, s, 2 * t
    best_key = None
    for mm in range(gm, min(M, MAX_TILE) + 1, gm):
        for kk in range(gk, min(K, MAX_TILE) + 1, gk):
            for nn in range(gn, min(N, MAX_TILE) + 1, gn):
                if not _is_valid(n_aie_cols, M, N, K, mm, kk, nn, dtype_out):
                    continue
                key = (mm * kk * nn, mm * nn, kk)
                if best_key is None or key > best_key:
                    best_key = key
    assert best_key is not None, "reference search found no valid tile"
    assert chosen_key == best_key, f"selector returned {chosen_key}, best is {best_key}"


@pytest.mark.parametrize("dev", DEVICE_NAMES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_real_tile_found_for_every_benchmarked_shape(dev, M, N, K):
    """Every shape we care about must get a real tile, never the granular minimum.

    This is the invariant that protects the measured speedup: the selector going
    conservative and falling back to the smallest microkernel-granular tile would
    silently undo it. Asserting a strict increase over that floor is falsifiable
    on both devices, unlike comparing against the superseded fixed 8^3 / 16^3.
    """
    _, _, _, m, k, n = _tile(dev, M, N, K)
    floor = _granular_minimum(dev)
    assert m * k * n > floor[0] * floor[1] * floor[2], (
        f"{dev} {M}x{N}x{K}: selector returned {(m, k, n)}, no better than the "
        f"granular minimum {floor}"
    )


@pytest.mark.parametrize("dev", DEVICE_NAMES)
def test_tile_selection_is_deterministic(dev):
    """Repeated calls must agree; the JIT cache keys on shape, not on the tile."""
    first = _tile(dev, 1024, 1024, 1024)
    for _ in range(5):
        assert _tile(dev, 1024, 1024, 1024) == first


def _stride_cliff_m(dev, n_aie_cols):
    """Smallest square M=N=K with no valid tile, where only the stride bound fails.

    Rounded up to the LCM of every array-tiling granularity so M, N and K are all
    tileable at the minimum tile -- otherwise the shape would be rejected for two
    reasons at once and the test could not attribute the failure to the stride.
    """
    gm, gk, gn = _granular_minimum(dev)
    step = math.lcm(gm * N_AIE_ROWS, gk, gn * n_aie_cols)
    m_cliff = DMA_MAX_STRIDE // (gn * n_aie_cols) + 1
    return -(-m_cliff // step) * step


@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
def test_raises_past_the_dma_stride_cliff(dev, n_aie_cols):
    """A shape with no valid tile must raise, not return an invalid one.

    The shim stride bound M * n * n_aie_cols caps M once n is at its granular
    minimum, so a large enough shape excludes every candidate. Returning the
    minimum tile anyway would be silently wrong: my_matmul re-checks only
    microkernel granularity and array tiling -- which that tile satisfies -- so
    it would reach aiecc and fail there with "Stride N exceeds the [1:1048576]
    range" instead of here, where the reason is still known.
    """
    m_cliff = _stride_cliff_m(dev, n_aie_cols)
    with pytest.raises(ValueError, match="shim-DMA stride exceeds"):
        select_gemm_tile(
            dev,
            m_cliff,
            m_cliff,
            m_cliff,
            BF16,
            F32,
            *resolve_mac_dims(dev, "bf16"),
            *resolve_expansion(dev, "bf16"),
            max_tile=MAX_TILE,
        )


@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
def test_stride_cliff_minimum_tile_would_pass_my_matmul_asserts(dev, n_aie_cols):
    """The rejected shape is exactly the dangerous kind: the old fallback was silent.

    Pins the reason the selector must raise. At the stride cliff the granular
    minimum still satisfies every constraint my_matmul asserts, so the previous
    behaviour of returning it produced a design that looked valid right up until
    aiecc rejected the stride.
    """
    gm, gk, gn = _granular_minimum(dev)
    m_cliff = _stride_cliff_m(dev, n_aie_cols)
    # my_matmul's re-checks: microkernel granularity and array tiling only.
    assert m_cliff % (gm * N_AIE_ROWS) == 0
    assert m_cliff % gk == 0
    assert m_cliff % (gn * n_aie_cols) == 0
    # ...yet the stride the design would emit is over the hardware limit.
    assert m_cliff * gn * n_aie_cols > DMA_MAX_STRIDE


@pytest.mark.parametrize("dev", DEVICE_NAMES)
def test_raises_when_problem_smaller_than_granularity(dev):
    """A problem below the microkernel granularity has no tile and must raise."""
    with pytest.raises(ValueError, match="smaller than the microkernel granularity"):
        select_gemm_tile(
            dev,
            1,
            1,
            1,
            BF16,
            F32,
            *resolve_mac_dims(dev, "bf16"),
            *resolve_expansion(dev, "bf16"),
            max_tile=MAX_TILE,
        )


def test_resolve_mac_dims_bf16_emulation_variants():
    """npu2 bf16 has two MAC shapes; the bfp16-emulated one must be selectable."""
    assert resolve_mac_dims("npu2", "bf16", False) == (4, 8, 8)
    assert resolve_mac_dims("npu2", "bf16", True) == (8, 8, 8)
    # npu has a single (non-dict) entry, so the flag must be ignored there.
    assert resolve_mac_dims("npu", "bf16") == microkernel_mac_dim_map["npu"]["bf16"]


def test_resolve_mac_dims_unknown_dtype_raises():
    with pytest.raises(KeyError):
        resolve_mac_dims("npu2", "f64")


@pytest.mark.parametrize("dtype_in_str", sorted(DTYPES))
@pytest.mark.parametrize("dev", DEVICE_NAMES)
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_selected_tile_satisfies_wrapper_contract(dev, M, N, K, dtype_in_str):
    """The tile must satisfy the rowA/colB divisibility its mm.cc wrapper documents.

    rowA = m/r and colB = n/t, and each wrapper steps those loops by its expansion
    factor. A blanket 2x2 assumption under-constrains the aie2 bf16 (4x4) and i8
    (4x2) wrappers, which is how the old fixed 8x8x8 aie2 tile came to violate the
    bf16 contract (colB = 8/4 = 2, but matmul_vectorized_4x4 needs colB % 4 == 0).
    """
    row_expand, col_expand = resolve_expansion(dev, dtype_in_str)
    dtype_out = DTYPES[dtype_in_str][1]
    r, s, t, m, k, n = _tile(dev, M, N, K, dtype_out, dtype_in_str)
    assert (m // r) % row_expand == 0, f"rowA={m // r} not divisible by {row_expand}"
    assert (n // t) % col_expand == 0, f"colB={n // t} not divisible by {col_expand}"
    assert k % s == 0


def test_aie2_i16_small_square_shapes_remain_supported():
    """aie2 i16 32^3/64^3 tiled with 8s before and must keep working.

    Regression guard: a fixed 32x32x32 aie2 tile silently dropped these, because
    my_matmul then needs M % (32*4) == 0. i16 uses the 2x2 wrapper, so its minimum
    tile is 8x4x8 and these shapes are legal.
    """
    for S in (32, 64, 128):
        _, _, _, m, k, n = _tile("npu", S, S, S, DTYPES["i16"][1], "i16")
        assert S % (m * N_AIE_ROWS) == 0
        assert S % (n * 4) == 0
        assert S % k == 0


def test_expansion_map_covers_every_mac_dim_entry():
    """Both per-dtype maps must agree on which (device, dtype) pairs exist."""
    for dev, dtypes in microkernel_mac_dim_map.items():
        assert set(microkernel_expansion_map[dev]) == set(dtypes), dev


@pytest.mark.parametrize("dev", DEVICE_NAMES)
@pytest.mark.parametrize("dtype_in_str", sorted(DTYPES))
def test_expansion_is_at_least_two(dev, dtype_in_str):
    """Every wrapper consumes at least 2 mmul subtiles per rowA/colB step."""
    row_expand, col_expand = resolve_expansion(dev, dtype_in_str)
    assert row_expand >= 2
    assert col_expand >= 2
