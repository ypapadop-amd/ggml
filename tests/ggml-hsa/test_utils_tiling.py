import os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src", "ggml-hsa", "kernels"))

import numpy as np
import pytest

from iron_kernels.utils import (
    _ARCH_PARAMS,
    _arch_params,
    max_tile_size,
    tiled_tile_size,
)


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


def test_tiled_tile_size_f32_mnist():
    # aie2 f32: V=16, budget=32768 bytes, 4 buffers (in+out, depth 2) of tile*4 bytes.
    # max_by_mem = (32768 // (4*4) // 16) * 16 = (2048 // 16) * 16 = 2048
    assert tiled_tile_size("aie2", np.dtype(np.float32), 250000) == 2048


def test_tiled_tile_size_multiple_of_vector_width():
    t = tiled_tile_size("aie2", np.dtype(np.float32), 250000)
    assert t % 16 == 0


def test_tiled_tile_size_capped_by_num_elements():
    # tiny tensor: capped at largest multiple of V <= N
    assert tiled_tile_size("aie2", np.dtype(np.float32), 48) == 48
    assert tiled_tile_size("aie2", np.dtype(np.float32), 50) == 48  # floor to mult of 16


def test_tiled_tile_size_floor_is_vector_width():
    # N smaller than V still returns V (kernel handles sub-V via scalar tail)
    assert tiled_tile_size("aie2", np.dtype(np.float32), 8) == 16


def test_tiled_tile_size_unknown_arch_raises():
    with pytest.raises(ValueError):
        tiled_tile_size("nope", np.dtype(np.float32), 250000)
