# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Unit tests for the GEMM per-core tile selector (``select_gemm_tile``).

These are pure-Python and need no NPU: they check the invariants the AIE
whole-array GEMM design relies on, so a bad tile is caught here instead of as a
miscompute or an ``aiecc`` failure on device.
"""

import math
import re
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

KERNELS_DIR = Path(__file__).resolve().parents[3] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

from iron_kernels.gemm import (  # noqa: E402
    DMA_MAX_SHIM_ITERATIONS,
    DMA_MAX_STRIDE,
    L1_TILE_BUDGET_BYTES,
    ceildiv,
    microkernel_expansion_map,
    microkernel_mac_dim_map,
    resolve_expansion,
    resolve_mac_dims,
    select_gemm_tile,
)

import iron_kernels.gemm as gemm_module  # noqa: E402

BF16 = np.dtype(ml_dtypes.bfloat16)
F32 = np.dtype(np.float32)

# Input dtype name -> (numpy input dtype, numpy output dtype) as the GEMM uses them.
DTYPES = {
    "bf16": (BF16, F32),
    "i8": (np.dtype(np.int8), np.dtype(np.int8)),
    "i16": (np.dtype(np.int16), np.dtype(np.int16)),
}

# GEMM output dtypes. The microkernel always consumes bf16, but the destination
# varies: the backend requests ggml's native f32 MUL_MAT output (what
# create_mat_mul_external_functions passes), while bf16 remains a valid gemm.py output.
# The L1 budget depends on the output element size, so both are covered.
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

# im2col-shaped GEMMs (M = batch*OH*OW, N = out channels, K = IC*KH*KW) that the
# herd covers in a single column group. Their M * n * n_aie_cols is far past
# DMA_MAX_STRIDE, but the dimension that stride steps has size 1, so it is never
# applied. Both are rejected outright on both devices by a bound that ignores
# the dimension size.
ONE_COLUMN_GROUP_SHAPES = [
    (32768, 128, 288),
    (65536, 256, 64),
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
    if N // (n * n_aie_cols) > DMA_MAX_SHIM_ITERATIONS:
        return False
    if N // (n * n_aie_cols) > 1 and (
        M * n * n_aie_cols > DMA_MAX_STRIDE or n * n_aie_cols * K > DMA_MAX_STRIDE
    ):
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

    Both strides step the column-group dimension, so the bound only binds while
    that dimension has size > 1; see test_single_column_group_ignores_the_bound
    and test_single_column_group_ignores_the_stride_bound.
    """
    _, _, _, m, k, n = _tile(dev, M, N, K)
    if N // (n * n_aie_cols) > 1:
        assert M * n * n_aie_cols <= DMA_MAX_STRIDE
        assert n * n_aie_cols * K <= DMA_MAX_STRIDE


@pytest.mark.parametrize(
    "dev,n_aie_cols,M,N,K",
    [
        # Hung on aie2p: the largest-volume tile gives 65 and 125 column groups.
        ("npu2", 8, 32, 16640, 512),
        ("npu2", 8, 1504, 32000, 768),
        # aie2: the largest-volume tile 16x128x48 gives 65 column groups.
        ("npu", 4, 64, 12480, 128),
    ],
)
def test_column_groups_fit_the_shim_iteration_limit(dev, n_aie_cols, M, N, K, monkeypatch):
    """The column-group count, the outermost A/B shim dimension, must stay <= 64.

    A shim BD iterates at most 64 times. aiecc splits a longer transfer over BDs whose
    ids it does not check, so the GEMM hangs or miscomputes instead of failing to
    compile. Pins both halves: without the bound the selector picks a tile over it.
    """
    *_, n = _tile(dev, M, N, K)
    assert N // (n * n_aie_cols) <= DMA_MAX_SHIM_ITERATIONS
    monkeypatch.setattr(gemm_module, "DMA_MAX_SHIM_ITERATIONS", 1 << 30)
    *_, unbounded_n = _tile(dev, M, N, K)
    assert N // (unbounded_n * n_aie_cols) > DMA_MAX_SHIM_ITERATIONS


def test_raises_past_the_shim_iteration_limit():
    """No tile with <= 64 column groups exists for N = 128 * 67 at max_tile 256.

    67 is prime, so a column group is 128 or 8576 columns wide; the second needs
    n = 1072. Returning the 67-group tile would hang the device.
    """
    with pytest.raises(ValueError, match="column groups"):
        select_gemm_tile(
            "npu2",
            32,
            128 * 67,
            64,
            BF16,
            F32,
            *resolve_mac_dims("npu2", "bf16"),
            *resolve_expansion("npu2", "bf16"),
            max_tile=MAX_TILE,
        )


@pytest.mark.parametrize("dev,n_aie_cols", DEVICES)
@pytest.mark.parametrize("M,N,K", ONE_COLUMN_GROUP_SHAPES)
def test_single_column_group_ignores_the_bound(dev, n_aie_cols, M, N, K):
    """A one-column-group GEMM is supportable only because the bound is size-aware.

    The im2col GEMMs behind CONV_2D are one column group wide, so the outermost
    B/C dimension has size 1 and its stride is never applied. Pins both halves:
    the dimension really is size 1, and the stride it carries really would be
    over the limit -- so dropping the size guard puts these shapes back on the
    CPU, and dropping the bound entirely lets a multi-group shape reach aiecc
    and fail there (covered by test_tile_respects_dma_stride_limit).
    """
    *_, n = _tile(dev, M, N, K)
    assert N // (n * n_aie_cols) == 1
    assert M * n * n_aie_cols > DMA_MAX_STRIDE


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


# ---------------------------------------------------------------------------
# Coupling guard for the host-side padding in gemm.cpp
# ---------------------------------------------------------------------------
#
# ggml_hsa_prepare_mul_mat_f32 pads a MUL_MAT's operands *before* any of this
# Python runs, so it cannot call select_gemm_tile and has to carry the
# granularity itself. These tests pin the C++ literals against the tables here:
# if a gemm.py change moves the granularity, the C++ is now wrong and this
# fails, naming the value it has to become.
#
# Both operands are converted to bf16 by that function, so only bf16 applies.

# The constants are read from gemm.cpp itself, so a change on either side alone
# fails here; a change to how gemm.cpp spells them fails the parse below.
GEMM_CPP = Path(__file__).resolve().parents[3] / "src" / "ggml-hsa" / "gemm.cpp"


def _read_cpp_padding_constants():
    """((gm, gk, gn, n_aie_cols) per device, n_aie_rows) from ggml_hsa_prepare_mul_mat_f32."""
    src = GEMM_CPP.read_text()

    def constant(name):
        pattern = rf"constexpr std::int64_t {name} = (\d+);"
        match = re.search(pattern, src)
        assert match, f"gemm.cpp: no match for {pattern!r}; update this parser"
        return int(match[1])

    def per_device(name):
        pattern = rf"const std::int64_t {name} = aie2p \? (\d+) : (\d+);"
        match = re.search(pattern, src)
        assert match, f"gemm.cpp: no match for {pattern!r}; update this parser"
        return {"npu2": int(match[1]), "npu": int(match[2])}

    gk, gn = constant("gk"), constant("gn")
    gm, n_aie_cols = per_device("gm"), per_device("n_aie_cols")
    granularity = {dev: (gm[dev], gk, gn, n_aie_cols[dev]) for dev in ("npu", "npu2")}
    return granularity, constant("n_aie_rows")


# (gm, gk, gn, n_aie_cols) per device, as ggml_hsa_prepare_mul_mat_f32 sets them.
CPP_PADDING_GRANULARITY, CPP_N_AIE_ROWS = _read_cpp_padding_constants()


def _cpp_padded_shape(dev, M, N, K):
    """(Mpad, Npad, Kpad) as ggml_hsa_prepare_mul_mat_f32 pads an M x N x K GEMM."""
    gm, gk, gn, n_aie_cols = CPP_PADDING_GRANULARITY[dev]
    return (
        ceildiv(M, gm * CPP_N_AIE_ROWS) * gm * CPP_N_AIE_ROWS,
        ceildiv(N, gn * n_aie_cols) * gn * n_aie_cols,
        ceildiv(K, gk) * gk,
    )


@pytest.mark.parametrize("dev", ["npu", "npu2"])
def test_cpp_padding_granularity_matches_gemm_py(dev):
    """The C++ padding granularity must equal (row_expand*r, s, col_expand*t)."""
    expected = _granular_minimum(dev)
    gm, gk, gn, _ = CPP_PADDING_GRANULARITY[dev]
    assert (gm, gk, gn) == expected, (
        f"ggml_hsa_prepare_mul_mat_f32 pads {dev} bf16 to {(gm, gk, gn)}, but "
        f"gemm.py's microkernel contract now requires {expected}; update the "
        f"literals in gemm.cpp"
    )


@pytest.mark.parametrize("dev", ["npu", "npu2"])
def test_cpp_n_aie_cols_matches_gemm_py(dev):
    """The C++ column count must match the one select_gemm_tile assumes."""
    _, _, _, n_aie_cols = CPP_PADDING_GRANULARITY[dev]
    assert n_aie_cols == dict(DEVICES)[dev]


@pytest.mark.parametrize("dev", ["npu", "npu2"])
@pytest.mark.parametrize(
    "mnk",
    [
        (1, 1, 1),
        (10, 500, 500),
        (500, 500, 784),
        (512, 512, 512),
        # The im2col GEMMs behind MNIST-CNN's two CONV_2D layers: M is the whole
        # batch*OH*OW extent, N is a single column group wide.
        (392000, 8, 9),
        (98000, 16, 72),
    ],
)
def test_cpp_padding_always_admits_a_tile(dev, mnk):
    """Padding as the C++ does must leave select_gemm_tile a valid tile.

    This is the property that matters: the host pads without consulting the
    selector, so if the two disagree the kernel build raises and the op
    silently falls back to the CPU.
    """
    M, N, K = mnk
    n_aie_cols = CPP_PADDING_GRANULARITY[dev][3]
    m_pad, n_pad, k_pad = _cpp_padded_shape(dev, M, N, K)

    r, s, t = resolve_mac_dims(dev, "bf16")
    row_expand, col_expand = resolve_expansion(dev, "bf16")
    tile = select_gemm_tile(
        dev, m_pad, n_pad, k_pad, BF16, F32, r, s, t, row_expand, col_expand
    )
    m, k, n = tile
    assert m_pad % (m * CPP_N_AIE_ROWS) == 0
    assert k_pad % k == 0
    assert n_pad % (n * n_aie_cols) == 0


@pytest.mark.parametrize("dev", ["npu", "npu2"])
@pytest.mark.parametrize("mnk", [(392000, 8, 9), (98000, 16, 72)])
def test_single_column_group_ignores_the_stride_bound(dev, mnk):
    """A one-column-group GEMM is supportable only because the bound is size-aware.

    The im2col GEMMs behind CONV_2D pad to a single column group, so the
    outermost B/C dimension has size 1 and its stride is never applied. Pins
    both halves: the dimension really is size 1, and the stride it carries
    really would be over the limit -- so dropping the size guard would put these
    shapes back on the CPU, and dropping the bound entirely would let a
    multi-group shape reach aiecc and fail there.
    """
    M, N, K = mnk
    n_aie_cols = CPP_PADDING_GRANULARITY[dev][3]
    m_pad, n_pad, k_pad = _cpp_padded_shape(dev, M, N, K)

    r, s, t = resolve_mac_dims(dev, "bf16")
    row_expand, col_expand = resolve_expansion(dev, "bf16")
    _, _, n = select_gemm_tile(
        dev, m_pad, n_pad, k_pad, BF16, F32, r, s, t, row_expand, col_expand
    )

    assert n_pad // (n * n_aie_cols) == 1
    assert m_pad * n * n_aie_cols > DMA_MAX_STRIDE


def _f32_b_tile(M, N, K, n_limit=None, m_limit=None, dev="npu"):
    """Tile for the GEMM that streams an f32 B and converts it on the core."""
    r, s, t = resolve_mac_dims(dev, "bf16")
    row_expand, col_expand = resolve_expansion(dev, "bf16")
    return select_gemm_tile(
        dev,
        M,
        N,
        K,
        BF16,
        F32,
        r,
        s,
        t,
        row_expand,
        col_expand,
        max_tile=MAX_TILE,
        dtype_b=F32,
        n_limit=n_limit,
        m_limit=m_limit,
    )


# Shapes whose f32-B tile was timed on aie2 over every equal-volume candidate (and, for the
# last, the six leading candidates of any volume), with the fastest balanced tile.
@pytest.mark.parametrize(
    "M,N,K,n_limit,m_limit,expected",
    [
        (4096, 512, 4096, None, None, (64, 64, 16)),
        (4096, 128, 4096, None, None, (64, 64, 16)),
        (2048, 512, 2048, None, None, (64, 64, 16)),
        # 4000x500x4000 as the backend pads it: M 4032, N 512, K 4000.
        (4032, 512, 4000, 500, 4000, (48, 80, 16)),
    ],
)
def test_f32_b_tile_balances_m_and_k(M, N, K, n_limit, m_limit, expected):
    """An f32 B ranks tiles by min(m, k) before volume.

    Each K-tile call reloads the f32 C tile (cost falling with k) and streams and converts
    the B tile (cost falling with m). The volume-first rule picked 64x16x64 or 128x16x32 on
    the first three shapes and 144x40x16 on the last, up to 1.9x slower than these.
    """
    assert _f32_b_tile(M, N, K, n_limit, m_limit) == expected


# Shapes as the backend pads them for each device (M to 4 * gm, N to n_aie_cols * gn, K to 8).
@pytest.mark.parametrize(
    "dev,M,N,K,n_limit,m_limit",
    [
        ("npu", 512, 512, 784, 500, 500),  # MNIST fc1
        ("npu", 64, 512, 504, 500, None),  # MNIST fc2
        ("npu", 1024, 1024, 1000, 1000, 1000),
        ("npu", 128, 128, 264, 70, 100),
        ("npu", 4032, 512, 4000, 500, 4000),
        ("npu2", 512, 512, 784, 500, 500),  # MNIST fc1
        ("npu2", 32, 512, 504, 500, None),  # MNIST fc2
        ("npu2", 1024, 1024, 1000, 1000, 1000),
        ("npu2", 128, 256, 264, 140, 100),
        ("npu2", 4000, 512, 4000, 500, None),
    ],
)
def test_f32_b_tile_fits_the_shift_limits_and_l1(dev, M, N, K, n_limit, m_limit):
    """A shifted last column group / row block must fit inside the real B / C.

    The f32 B tile is double-buffered at 4 bytes and has a single bf16 scratch copy, so
    the L1 working set is larger than the bf16-B one ``_working_set`` models.
    """
    m, k, n = _f32_b_tile(M, N, K, n_limit, m_limit, dev)
    assert n * dict(DEVICES)[dev] <= n_limit
    if m_limit is not None:
        assert m * N_AIE_ROWS <= m_limit
    working_set = 2 * (
        BF16.itemsize * m * k + F32.itemsize * k * n + F32.itemsize * m * n
    )
    working_set += BF16.itemsize * k * n
    assert working_set <= L1_TILE_BUDGET_BYTES


@pytest.mark.parametrize("n_limit,m_limit", [(63, None), (None, 63)])
def test_shift_limits_below_one_group_raise(n_limit, m_limit):
    """No tile exists when B or C is narrower than one minimum column group / row block."""
    with pytest.raises(ValueError, match="minimum"):
        _f32_b_tile(512, 512, 512, n_limit, m_limit)
