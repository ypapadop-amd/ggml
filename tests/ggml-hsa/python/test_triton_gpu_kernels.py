# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tests for the Triton GPU (gfx) kernels behind the MNIST layers: MUL_MAT, ADD, RELU.

No GPU is required: numerics run in the Triton interpreter on CPU tensors (in a subprocess, see
triton_interpret_run.py), and the compile test only ahead-of-time compiles to a .hsaco.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import KERNELS_DIR

# build_triton.py uses flat imports (from kernel import ...); see test_triton_matmul_compile.py.
sys.path.insert(0, str(KERNELS_DIR))

pytest.importorskip("torch")
pytest.importorskip("triton")

ARCH = "gfx1150"


@pytest.fixture(scope="module")
def mods(import_kernel_module):
    """Load the dispatch modules and TensorDesc."""
    return {
        "Backend": import_kernel_module("kernel").Backend,
        "TensorDesc": import_kernel_module("tensor_desc").TensorDesc,
        "mul_mat": import_kernel_module("mul_mat"),
        "binary_ops": import_kernel_module("binary_ops"),
        "unary_ops": import_kernel_module("unary_ops"),
    }


def _td(mods, shape, dtype="f32"):
    return mods["TensorDesc"](dtype=dtype, shape=(*shape, *(1,) * (4 - len(shape))))


def _run(tmp_path, module, fn, inputs, output, *arrays):
    """Run a GPU spec's kernel in the Triton interpreter; see triton_interpret_run.py.

    Returns:
        The flat f32 output.
    """
    input_files = []
    for i, array in enumerate(arrays):
        input_files.append(str(tmp_path / f"in{i}.npy"))
        np.save(input_files[-1], np.ascontiguousarray(array).reshape(-1))
    output_file = tmp_path / "out.npy"
    args = {
        "module": module,
        "fn": fn,
        "arch": ARCH,
        "inputs": inputs,
        "output": output,
        "input_files": input_files,
        "output_file": str(output_file),
    }
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "triton_interpret_run.py"),
            json.dumps(args),
        ],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        check=True,
    )
    return np.load(output_file)


# GGML shapes, innermost first: (K, M) x (K, N) -> (M, N). MNIST FC is 784 -> 500 -> 10 at a
# 500-image batch; the small ones leave partial tiles in every dimension.
@pytest.mark.parametrize(
    ("k", "m", "n"),
    [(784, 500, 500), (500, 10, 500), (33, 17, 5), (16, 64, 64)],
)
def test_mul_mat_matches_reference(tmp_path, mods, k, m, n):
    """MUL_MAT GPU kernel matches a numpy reference."""
    specs = mods["mul_mat"].ggml_op_mul_mat(
        ARCH, [_td(mods, (k, m)), _td(mods, (k, n))], _td(mods, (m, n)), bytearray()
    )
    assert [s.backend for s in specs] == [mods["Backend"].TRITON]

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k), dtype=np.float32)  # row m holds K contiguous values
    b = rng.standard_normal((n, k), dtype=np.float32)
    c = _run(tmp_path, "mul_mat", "ggml_op_mul_mat", [(k, m), (k, n)], (m, n), a, b)

    expected = (b @ a.T).reshape(-1)  # C[n * M + m]
    np.testing.assert_allclose(c, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("dst", "src1"),
    [
        ((500, 500), (500,)),  # MNIST bias add
        ((10, 7, 3, 2), (10, 1, 3, 1)),
        ((6, 4), (3, 2)),  # repeated, not just broadcast from 1
    ],
)
def test_add_broadcast_matches_reference(tmp_path, mods, dst, src1):
    """Broadcast ADD GPU kernel matches a numpy reference."""
    specs = mods["binary_ops"].ggml_op_add(
        ARCH, [_td(mods, dst), _td(mods, src1)], _td(mods, dst), bytearray()
    )
    assert len(specs) == 1
    assert specs[0].config["triton_fn"].fn.__name__ == "add_bcast_gpu"

    rng = np.random.default_rng(0)
    # numpy shapes are GGML shapes reversed (outermost first)
    a = rng.standard_normal(dst[::-1], dtype=np.float32)
    b = rng.standard_normal(src1[::-1], dtype=np.float32)
    c = _run(tmp_path, "binary_ops", "ggml_op_add", [dst, src1], dst, a, b)

    reps = tuple(
        x // y for x, y in zip(a.shape, (1,) * (a.ndim - b.ndim) + b.shape, strict=True)
    )
    expected = a + np.tile(b.reshape((1,) * (a.ndim - b.ndim) + b.shape), reps)
    np.testing.assert_array_equal(c, expected.reshape(-1))


def test_add_same_shape_keeps_vecadd(mods):
    """Same-shape ADD still selects the vecadd kernel."""
    specs = mods["binary_ops"].ggml_op_add(
        ARCH,
        [_td(mods, (500, 500)), _td(mods, (500, 500))],
        _td(mods, (500, 500)),
        bytearray(),
    )
    assert specs[0].config["triton_fn"].fn.__name__ == "vecadd"


@pytest.mark.parametrize("n_elements", [500 * 500, 1000, 7])
def test_relu_matches_reference(tmp_path, mods, n_elements):
    """RELU GPU kernel matches a numpy reference."""
    specs = mods["unary_ops"].ggml_unary_op_relu(
        ARCH, [_td(mods, (n_elements,))], _td(mods, (n_elements,)), bytearray()
    )
    assert [s.backend for s in specs] == [mods["Backend"].TRITON]

    x = np.random.default_rng(0).standard_normal(n_elements, dtype=np.float32)
    y = _run(
        tmp_path, "unary_ops", "ggml_unary_op_relu", [(n_elements,)], (n_elements,), x
    )

    np.testing.assert_array_equal(y, np.maximum(x, 0))


def test_unsupported_gpu_cases_yield_no_spec(mods):
    """Unsupported GPU cases produce no kernel spec."""
    # batched MUL_MAT
    assert (
        mods["mul_mat"].ggml_op_mul_mat(
            ARCH,
            [_td(mods, (32, 16, 2)), _td(mods, (32, 8, 2))],
            _td(mods, (16, 8, 2)),
            bytearray(),
        )
        == []
    )
    # src1 not repeatable to dst
    assert (
        mods["binary_ops"].ggml_op_add(
            ARCH, [_td(mods, (6,)), _td(mods, (4,))], _td(mods, (6,)), bytearray()
        )
        == []
    )


def _has_hip_backend():
    try:
        import triton.backends.amd.compiler  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means no AMD backend
        return False
    return True


@pytest.mark.skipif(not _has_hip_backend(), reason="Triton AMD GPU backend unavailable")
@pytest.mark.parametrize(
    ("module", "fn", "inputs", "output"),
    [
        ("mul_mat", "ggml_op_mul_mat", [(784, 500), (784, 500)], (500, 500)),
        ("binary_ops", "ggml_op_add", [(500, 500), (500,)], (500, 500)),
        ("unary_ops", "ggml_unary_op_relu", [(500, 500)], (500, 500)),
    ],
)
def test_compiles_to_hsaco(tmp_path, mods, module, fn, inputs, output):
    """Each GPU kernel compiles to a non-empty hsaco with consistent launch geometry."""
    from build_triton import compile_triton_kernel

    (spec,) = getattr(mods[module], fn)(
        ARCH, [_td(mods, s) for s in inputs], _td(mods, output), bytearray()
    )
    compile_triton_kernel(
        kernel_spec=spec,
        exported_name=fn,
        output_directory=tmp_path,
        logger=logging.getLogger(__name__),
        verbose=False,
    )

    assert (tmp_path / f"{fn}.hsaco").stat().st_size > 0
    launch = json.loads((tmp_path / f"{fn}.hsaco.json").read_text())
    assert (
        launch["grid_size_x"]
        == spec.config["num_programs"] * launch["workgroup_size_x"]
    )
