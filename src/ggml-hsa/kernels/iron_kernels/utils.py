# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates

"""Utility functions for IRON kernel implementations."""

import numpy as np
from aie.iron.device import NPU1, NPU2


def align_to_arch(
    arch: str, size: int, dtype: np.dtype, alignment_bytes: int = 4
) -> int:
    """Align an element count so its byte size is a multiple of alignment_bytes.

    Parameters:
        arch: Target architecture.
        size: Element count to align.
        dtype: Element data type.
        alignment_bytes: Byte boundary to align to.

    Returns:
        The aligned element count.

    """
    if arch in ["aie2", "aie2p"]:
        dtype_size = dtype.itemsize
        data_size = size * dtype_size
        if data_size % alignment_bytes != 0:
            return (
                alignment_bytes
                * ((data_size + (alignment_bytes - 1)) // alignment_bytes)
                // dtype_size
            )
        return size
    msg = f"Unsupported architecture: {arch}"
    raise ValueError(msg)


def arch_aligned_num_elements(arch: str, tensor) -> int:
    """Tensor element count aligned to the architecture for its dtype.

    Parameters:
        arch: Target architecture.
        tensor: Tensor whose element count is aligned.

    Returns:
        The arch-aligned element count.

    """
    return align_to_arch(arch, tensor.numel(), tensor.dtype)


def max_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """Largest power-of-two tile within a 512-bit vector dividing num_elements.

    Parameters:
        arch: Target architecture.
        dtype: Element data type.
        num_elements: Total number of elements to tile.

    Returns:
        The chosen tile size.

    """
    vector_register_width = 0
    if arch in {"aie2", "aie2p"}:
        vector_register_width = 512  # bits
    else:
        msg = f"Unsupported architecture: {arch}"
        raise ValueError(msg)
    tile_size = int(vector_register_width / dtype.itemsize)

    while num_elements % tile_size != 0 and tile_size > 1:
        tile_size //= 2

    assert num_elements % tile_size == 0, (
        f"Number of elements ({num_elements}) must be a multiple of "
        f"tile size ({tile_size})."
    )

    return tile_size


def arch_to_device(device):
    """Map "aie2" -> NPU1, "aie2p" -> NPU2; pass through existing device objects.

    Parameters:
        device: Architecture string or an existing device object.

    Returns:
        The corresponding device object.

    """
    if isinstance(device, str):
        if device == "aie2":
            return NPU1()
        if device == "aie2p":
            return NPU2()
        msg = f"Unsupported device: {device}"
        raise ValueError(msg)
    return device
