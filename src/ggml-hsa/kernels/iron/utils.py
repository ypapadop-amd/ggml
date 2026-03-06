# suppress stderr from aie imports until https://github.com/Xilinx/mlir-aie/issues/2833
# is resolved

"""
Utility functions for IRON kernel implementations.
"""

import contextlib
import os

import numpy as np

from aie.iron.device import NPU1, NPU2

with open(os.devnull, "w", encoding="utf-8") as _ggml_hsa_devnull:
    with contextlib.redirect_stderr(_ggml_hsa_devnull):
        import aie.utils as _ggml_hsa_aie_utils


def suppress_import_pyxrt_msg():
    """Return the pre-imported aie.utils module with pyxrt message suppressed.

    The aie.utils module is imported once at module import time with stderr
    suppressed to avoid noisy pyxrt not found messages. This function is
    retained for backward compatibility and simply returns the cached module.
    """

    return _ggml_hsa_aie_utils


def align_to_arch(
    arch: str, size: int, dtype: np.dtype, alignment_bytes: int = 4
) -> int:
    """
    Align a size to architecture requirements.

    Parameters:
        arch (str): Target architecture.
        size (int): Size to align (number of elements).
        dtype (np.dtype): Data type of elements.
        alignment_bytes (int): Alignment in bytes.

    Returns:
        int: Aligned size.
    """
    if arch in ["aie2", "aie2p"]:
        dtype_size = dtype.itemsize
        data_size = size * dtype_size
        if data_size % alignment_bytes != 0:
            aligned_size = (
                alignment_bytes
                * ((data_size + (alignment_bytes - 1)) // alignment_bytes)
                // dtype_size
            )
            return aligned_size
        return size
    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def arch_aligned_num_elements(arch: str, tensor) -> int:
    """
    Returns the number of elements in the tensor aligned to what the architecture expects for the data type of the tensor.

    Parameters:
        arch (str): Target architecture.
        tensor: Tensor.

    Returns:
        int: Number of elements aligned to architecture requirements.
    """
    return align_to_arch(arch, tensor.numel(), tensor.dtype)


def max_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """
    Returns the maximum tile size based on device, data type and number of elements.

    Parameters:
        arch (str): Target architecture.
        dtype (np.dtype): Data type of the tensor elements.
        num_elements (int): Total number of elements in the tensor.

    Returns:
        int: Maximum tile size.
    """
    vector_register_width = 0
    if arch == "aie2" or arch == "aie2p":
        vector_register_width = 512  # bits
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    tile_size = int(vector_register_width / dtype.itemsize)

    while num_elements % tile_size != 0 and tile_size > 1:
        tile_size //= 2

    assert (
        num_elements % tile_size == 0
    ), f"Number of elements ({num_elements}) must be a multiple of tile size ({tile_size})."

    return tile_size


def arch_to_device(device):
    """
    Converts an architecture string to an IRON device object.

    Parameters:
        device: Architecture string ("aie2" or "aie2p") or an existing device object.

    Returns:
        NPU1 for "aie2", NPU2 for "aie2p", or the input if already a device object.

    Raises:
        ValueError: If the architecture string is not supported.
    """
    if isinstance(device, str):
        if device == "aie2":
            return NPU1()
        elif device == "aie2p":
            return NPU2()
        else:
            raise ValueError(f"Unsupported device: {device}")
    return device
