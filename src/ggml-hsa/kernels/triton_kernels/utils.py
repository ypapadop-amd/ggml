# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utility functions for Triton kernel integration."""

import ml_dtypes
import numpy as np
import torch

from .spec_utils import NPU_ARCH_MAP, ArchKind, arch_kind  # noqa: F401

_NUMPY_TO_TORCH_DTYPE = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    ml_dtypes.bfloat16: torch.bfloat16,
}


def triton_device(arch: str) -> str:
    """Return the Triton device string: "cpu" for NPU targets, "cuda" for GPU targets.

    Args:
        arch: Architecture string to map to a Triton device.

    Returns:
        "cpu" for NPU architectures, "cuda" for GPU architectures.

    Raises:
        ValueError: If arch is not a recognised NPU or GPU architecture.
    """
    kind = arch_kind(arch)
    if kind is ArchKind.NPU:
        return "cpu"
    if kind is ArchKind.GPU:
        return "cuda"
    msg = f"Unsupported architecture: {arch!r}"
    raise ValueError(msg)


def numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to the corresponding torch dtype.

    Args:
        dtype: The numpy dtype to convert.

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
