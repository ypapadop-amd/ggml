# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tensor descriptor for GGML HSA kernel operations.

TensorDesc captures the dtype, shape, stride, and contiguity a kernel needs to
compile. Dimensions follow GGML convention: innermost to outermost (reverse of
PyTorch).
"""

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

# Maps GGML dtype strings to numpy dtypes.
_GGML_NP_DTYPE_MAP = {
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "bf16": bfloat16,
    "f16": np.float16,
    "f32": np.float32,
}


def str_to_dtype(dtype_str: str):
    """Convert a GGML dtype string to its corresponding numpy dtype.

    Args:
        dtype_str: GGML dtype string to convert (e.g. "f32", "bf16").

    Returns:
        The corresponding numpy dtype.

    Raises:
        ValueError: If dtype_str is not recognized.

    """
    try:
        return _GGML_NP_DTYPE_MAP[dtype_str]
    except KeyError as e:
        msg = f"Unrecognized dtype: {dtype_str}. Supported dtypes are: {list(_GGML_NP_DTYPE_MAP.keys())}"
        raise ValueError(msg) from e


@dataclass(frozen=True)
class TensorDesc:
    """ggml_tensor description.

    Attributes:
        dtype: Data type of the tensor.
        shape: Shape of the tensor as a tuple of integers. Dimensions are from
            innermost to outermost (reverse of PyTorch).
        stride: Stride of the tensor as a tuple of integers, or None if not
            specified. Dimensions are from innermost to outermost (reverse of PyTorch).
        contiguous: Indicates if the tensor is contiguous in memory.

    """

    dtype: np.dtype | str
    shape: tuple[int, int, int, int]
    stride: tuple[int, int, int, int] | None = None
    contiguous: bool = True

    def __post_init__(self) -> None:
        """Validate and compute derived properties of the tensor descriptor."""
        # convert dtype to np.dtype if it's a string
        if isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", np.dtype(str_to_dtype(self.dtype)))

        # compute stride if not provided as if the tensor is contiguous
        if self.stride is None:
            stride = [0, 0, 0, 0]
            stride[0] = self.dtype.itemsize
            stride[1] = stride[0] * self.shape[0]
            for i in range(2, len(self.shape)):
                stride[i] = stride[i - 1] * self.shape[i - 1]
            object.__setattr__(self, "stride", tuple(stride))

    @property
    def size(self):
        """Number of elements in the tensor."""
        return int(np.prod(self.shape))

    def numel(self):
        """Number of elements in the tensor."""
        return self.size


def ggml_tensor_to_tensordesc(
    dtype: str,
    ne: tuple[int, int, int, int],
    nb: tuple[int, int, int, int],
    contiguous: bool,
) -> TensorDesc:
    """Create a TensorDesc from ggml_tensor parameters.

    Args:
        dtype: Tensor data type.
        ne: Number of elements per dimension (innermost to outermost).
        nb: Stride in bytes per dimension (innermost to outermost).
        contiguous: Whether the tensor is contiguous in memory.

    Returns:
        The constructed TensorDesc.

    """
    return TensorDesc(dtype=dtype, shape=ne, stride=nb, contiguous=contiguous)
