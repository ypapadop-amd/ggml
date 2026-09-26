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
# M must be a multiple of the arch's L3 block M (_BLOCK_MN_BY_ARCH in
# mul_mat.py): 256 on aie2, 512 on aie2p. N=256 and K=256 satisfy both archs.
@pytest.mark.parametrize(("arch", "m"), [("aie2", 256), ("aie2p", 512)])
def test_matmul_compiles_to_pdi(tmp_path, arch, m, import_kernel_module):
    from build_triton import compile_triton_kernel
    from tensor_desc import TensorDesc

    mul_mat = import_kernel_module("mul_mat")

    n = k = 256

    def _td(dtype, shape):
        return TensorDesc(dtype=dtype, shape=(*shape, 1, 1))

    # GGML shape convention (innermost first): A is [K, M], B is [K, N], C is [M, N].
    spec = mul_mat._make_triton_matmul_kernel_spec(
        arch,
        [_td("bf16", (k, m)), _td("bf16", (k, n))],
        _td("f32", (m, n)),
    )
    name = f"mul_mat_{arch}"
    compile_triton_kernel(spec, name, tmp_path, logging.getLogger("test"), verbose=False)

    assert (tmp_path / f"{name}.pdi").is_file()
    assert (tmp_path / f"{name}_insts.bin").is_file()
