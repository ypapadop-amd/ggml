# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Codegen tests for the GEMM writing a dense destination from padded operands.

Every case builds the MLIR module and verifies it. With RUN_AIECC_TESTS=1 (and the IRON
environment exported) the aiecc cases also compile it to a PDI, which is the only check the
runtime-sequence passes -- BD ids, task queue, lock rules -- run without hardware.
"""

import os
import sys
import tempfile
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

KERNELS_DIR = Path(__file__).resolve().parents[3] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

from iron_kernels.gemm import gemm  # noqa: E402

BF16 = ml_dtypes.bfloat16
F32 = np.float32
# arch -> (gm, gk, gn, n_aie_cols): the backend's padding granularity
PAD = {"aie2p": (8, 8, 16, 8), "aie2": (16, 8, 16, 4)}
SHAPES = [
    (512, 512, 512),
    (500, 500, 784),
    (10, 500, 500),
    (10, 500, 784),
    (392000, 8, 9),
    (98000, 16, 72),
    (499, 500, 784),
]


class TD:
    """The tensor-descriptor fields gemm() reads."""

    def __init__(self, dtype, ne) -> None:  # noqa: D107
        self.dtype, self.shape, self.contiguous = dtype, tuple(ne), True


def _pad(x, g):
    return -(-x // g) * g


def _operands(arch, M, N, K):
    gm, gk, gn, cols = PAD[arch]
    Mp, Np, Kp = _pad(M, gm * 4), _pad(N, gn * cols), _pad(K, gk)
    return [TD(BF16, (Kp, Mp, 1, 1)), TD(BF16, (Kp, Np, 1, 1))], Mp, Np


def _module(arch, shape, out):
    from aie.iron import ExternalFunction

    # The registry is process-global; aie2 and aie2p register the same object name from
    # different mm.cc sources, which would collide across cases.
    ExternalFunction._instances.clear()
    M, N, K = shape
    ops, Mp, Np = _operands(arch, M, N, K)
    try:
        gemm(arch, ops, TD(F32, (Mp, Np, 1, 1)))
    except ValueError as e:
        pytest.skip(f"padded shape unsupported on {arch} before this change: {e}")
    return gemm(arch, ops, TD(out, (M, N, 1, 1)))


@pytest.mark.parametrize("arch", list(PAD))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("out", [F32, BF16], ids=["f32", "bf16"])
def test_dense_c_module_verifies(arch, shape, out):
    """The dense-C module builds and verifies."""
    if out is BF16 and shape[0] % 2:
        pytest.skip("graph_optimize never gives the GEMM a bf16 C with odd M")
    assert _module(arch, shape, out).operation.verify()


def test_bf16_c_with_odd_m_is_rejected():
    """A bf16 C with odd M would start every other column mid-word."""
    ops, _, _ = _operands("aie2p", 499, 500, 784)
    with pytest.raises(ValueError, match="odd"):
        gemm("aie2p", ops, TD(BF16, (499, 500, 1, 1)))


def test_c_larger_than_padded_operands_is_rejected():
    """C may not extend past the padded operands."""
    ops, Mp, Np = _operands("aie2p", 500, 500, 784)
    with pytest.raises(ValueError, match="padded M"):
        gemm("aie2p", ops, TD(F32, (Mp + 1, Np, 1, 1)))


@pytest.mark.skipif(
    os.environ.get("RUN_AIECC_TESTS") != "1", reason="set RUN_AIECC_TESTS=1"
)
@pytest.mark.parametrize("arch", list(PAD))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("out", [F32, BF16], ids=["f32", "bf16"])
def test_dense_c_compiles_with_aiecc(arch, shape, out):
    """The dense-C module compiles to a PDI through aiecc."""
    if out is BF16 and shape[0] % 2:
        pytest.skip("graph_optimize never gives the GEMM a bf16 C with odd M")
    from aie.iron import ExternalFunction
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    ExternalFunction._instances.clear()
    mod = _module(arch, shape, out)
    with tempfile.TemporaryDirectory() as work:
        for f in ExternalFunction._instances:
            compile_external_kernel(f, work, arch)
        ExternalFunction._instances.clear()
        compile_mlir_module(
            mlir_module=mod,
            insts_path=f"{work}/insts.bin",
            pdi_path=f"{work}/gemm.pdi",
            verbose=False,
            work_dir=work,
        )
        assert Path(f"{work}/gemm.pdi").stat().st_size > 0


# (arch, shape): shapes whose mem-tile C tasks once exceeded the mem tile's live-BD budget, and
# base 2053fe89 ran on the NPU. aiecc is the check that counts the live BDs.
LARGE = [
    ("aie2p", (4096, 4096, 4096)),
    ("aie2p", (1000, 2000, 1000)),
    ("aie2", (1024, 4096, 1024)),
]


# aie2's f32-B path (dense M, N, K): B streamed as f32 and converted on the core, read unpadded
# with a K tail once it is a column group wide, and C written through the shifted last column
# group and (once M is a row block) row block. 4096x512x4096 is upstream's timed f32-B shape.
F32B_SHAPES = [
    (500, 500, 784),  # MNIST fc1: rows and columns shifted
    (10, 500, 500),  # MNIST fc2: rows clipped, columns shifted, K tail
    (4096, 512, 4096),  # aligned
    (4000, 500, 4000),  # shifted, many row blocks
    (129, 300, 1000),
    (64, 70, 257),  # K tail with odd K
    (300, 8, 256),  # N below a column group: B padded, clipped
]


def _f32_b_module(shape):
    from aie.iron import ExternalFunction

    ExternalFunction._instances.clear()
    M, N, K = shape
    gm, gk, gn, cols = PAD["aie2"]
    Mp, Np, Kp = _pad(M, gm * 4), _pad(N, gn * cols), _pad(K, gk)
    b = (K, N) if N >= gn * cols else (Kp, Np)
    ops = [TD(BF16, (Kp, Mp, 1, 1)), TD(F32, (*b, 1, 1))]
    return gemm("aie2", ops, TD(F32, (M, N, 1, 1)))


@pytest.mark.parametrize("shape", F32B_SHAPES)
def test_f32_b_module_verifies(shape):
    """aie2's f32-B GEMM (shifted or clipped C) builds and verifies."""
    assert _f32_b_module(shape).operation.verify()


def test_unpadded_b_needs_the_f32_b_path():
    """Only an f32 B converted on aie2's cores may be read unpadded."""
    ops = [TD(BF16, (784, 512, 1, 1)), TD(BF16, (784, 500, 1, 1))]
    with pytest.raises(ValueError, match="f32 B"):
        gemm("aie2", ops, TD(F32, (500, 500, 1, 1)))


def _compile(mod, arch):
    from aie.iron import ExternalFunction
    from aie.utils.compile import compile_external_kernel, compile_mlir_module

    with tempfile.TemporaryDirectory() as work:
        for f in ExternalFunction._instances:
            compile_external_kernel(f, work, arch)
        ExternalFunction._instances.clear()
        compile_mlir_module(
            mlir_module=mod,
            insts_path=f"{work}/insts.bin",
            pdi_path=f"{work}/gemm.pdi",
            verbose=False,
            work_dir=work,
        )
        assert Path(f"{work}/gemm.pdi").stat().st_size > 0


@pytest.mark.skipif(
    os.environ.get("RUN_AIECC_TESTS") != "1", reason="set RUN_AIECC_TESTS=1"
)
@pytest.mark.parametrize("shape", F32B_SHAPES)
def test_f32_b_compiles_with_aiecc(shape):
    """aie2's f32-B GEMM compiles to a PDI: BD ids, task queue and lock rules hold."""
    _compile(_f32_b_module(shape), "aie2")


@pytest.mark.parametrize(("arch", "shape"), LARGE)
def test_large_dense_c_module_verifies(arch, shape):
    """The large-shape dense-C modules build and verify."""
    assert _module(arch, shape, F32).operation.verify()


@pytest.mark.skipif(
    os.environ.get("RUN_AIECC_TESTS") != "1", reason="set RUN_AIECC_TESTS=1"
)
@pytest.mark.parametrize(("arch", "shape"), LARGE)
def test_large_dense_c_compiles_with_aiecc(arch, shape):
    """The large-shape dense-C modules fit the mem tile's BDs and compile to a PDI."""
    test_dense_c_compiles_with_aiecc(arch, shape, F32)
