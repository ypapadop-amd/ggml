# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""End-to-end compile check for the Triton matmul (no device required)."""

import logging
import sys

import pytest

from conftest import KERNELS_DIR

# build_triton.py and its deps use flat imports (from kernel import ...), so the
# kernels dir must be on sys.path. mul_mat.py uses package-relative imports and
# is loaded via the import_kernel_module fixture. compile_triton_kernel uses
# duck typing on the spec, so the two KernelSpec classes coexisting is harmless.
sys.path.insert(0, str(KERNELS_DIR))

pytest.importorskip("triton")


def _has_npu_backend():
    try:
        import triton.backends.amd_triton_npu.driver  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_npu_backend(), reason="amd_triton_npu backend unavailable")
@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_matmul_compiles_to_pdi(tmp_path, arch, import_kernel_module):
    from build_triton import compile_triton_kernel
    from tensor_desc import TensorDesc

    mul_mat = import_kernel_module("mul_mat")

    def _td(dtype):
        return TensorDesc(dtype=dtype, shape=(256, 256, 1, 1))

    spec = mul_mat._make_triton_matmul_kernel_spec(
        arch, [_td("bf16"), _td("bf16")], _td("f32")
    )
    name = f"mul_mat_{arch}"
    compile_triton_kernel(spec, name, tmp_path, logging.getLogger("test"), verbose=False)

    assert (tmp_path / f"{name}.pdi").is_file()
    assert (tmp_path / f"{name}_insts.bin").is_file()
