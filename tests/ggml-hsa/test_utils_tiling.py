import os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "ggml-hsa", "kernels"))

import numpy as np
import pytest

from iron_kernels.utils import _ARCH_PARAMS, _arch_params, max_tile_size


def test_arch_params_has_known_archs():
    for arch in ("aie2", "aie2p"):
        p = _arch_params(arch)
        assert p["core_data_mem_bytes"] == 64 * 1024
        assert p["vector_reg_bits"] == 512


def test_arch_params_unknown_raises():
    with pytest.raises(ValueError):
        _arch_params("nope")


def test_max_tile_size_unchanged_f32_250000():
    # 250000 = 2^4 * 5^6 -> largest pow2 divisor within 512 bits is 16
    assert max_tile_size("aie2", np.dtype(np.float32), 250000) == 16


def test_max_tile_size_unchanged_pow2():
    # 512-bit vector / 4-byte f32 = 128 elements/tile; 2048 divides evenly, so 128
    assert max_tile_size("aie2", np.dtype(np.float32), 2048) == 128
