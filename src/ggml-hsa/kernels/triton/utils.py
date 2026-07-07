# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utility functions for Triton kernel integration."""

import ml_dtypes
import numpy as np
import torch

_NUMPY_TO_TORCH_DTYPE = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    ml_dtypes.bfloat16: torch.bfloat16,
}


NPU_ARCH_MAP = {
    "aie2": "npu1",
    "aie2p": "npu2",
}


def is_npu_arch(arch: str) -> bool:
    """Return True if arch is a raw NPU architecture string (e.g. "aie2", "aie2p")."""
    return arch in NPU_ARCH_MAP


def is_gpu_arch(arch: str) -> bool:
    """Return True if arch is an AMD GPU architecture string (e.g. "gfx942")."""
    return arch.startswith("gfx")


def triton_device(arch: str) -> str:
    """Return the Triton device string for the given architecture.

    NPU architectures use "cpu"; GPU architectures use "cuda".

    Parameters:
        arch: Raw architecture string (e.g. "aie2", "aie2p", "gfx942").

    Returns:
        "cpu" for NPU targets, "cuda" for GPU targets.

    Raises:
        ValueError: If arch is not a recognised NPU or GPU architecture.
    """
    if is_npu_arch(arch):
        return "cpu"
    if is_gpu_arch(arch):
        return "cuda"
    msg = f"Unsupported architecture: {arch!r}"
    raise ValueError(msg)


def numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to the corresponding torch dtype.

    Parameters:
        dtype: A numpy dtype instance.

    Returns:
        The corresponding torch dtype.

    Raises:
        ValueError: If the numpy dtype has no torch equivalent.

    """
    dtype = np.dtype(dtype)
    torch_dtype = _NUMPY_TO_TORCH_DTYPE.get(dtype.type)
    if torch_dtype is None:
        msg = f"No torch dtype equivalent for numpy dtype: {dtype}"
        raise ValueError(msg)
    return torch_dtype
