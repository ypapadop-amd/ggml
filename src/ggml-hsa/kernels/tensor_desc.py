# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Tensor descriptor for GGML HSA kernel operations.

This module provides the TensorDesc dataclass used to describe tensors passed
to kernels. It captures the essential properties needed for kernel
compilation: data type, shape, stride, and contiguity information.

The tensor dimensions follow GGML conventions where dimensions are ordered
from innermost to outermost (reverse of PyTorch).
"""

from dataclasses import dataclass

import numpy as np
from aie.iron import str_to_dtype

# Mapping for dtypes not natively supported by IRON but still valid GGML types.
# These tensors can still be described, but kernels need to have special handling for
# them.
_FALLBACK_DTYPE_MAP = {
    "i64": np.int64,
    "u64": np.uint64,
    "f64": np.float64,
}


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
            # First try AIE-supported dtypes, then fall back to numpy for others
            try:
                object.__setattr__(self, "dtype", np.dtype(str_to_dtype(self.dtype)))
            except ValueError:
                # dtype not supported by AIE - use numpy dtype for fallback
                if self.dtype in _FALLBACK_DTYPE_MAP:
                    object.__setattr__(
                        self, "dtype", np.dtype(_FALLBACK_DTYPE_MAP[self.dtype])
                    )
                else:
                    raise

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
        """Return the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.

        """
        return int(np.prod(self.shape))

    def numel(self):
        """Return the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.

        """
        return self.size


def ggml_tensor_to_tensordesc(
    dtype: str,
    ne: tuple[int, int, int, int],
    nb: tuple[int, int, int, int],
    contiguous: bool,
) -> TensorDesc:
    """Create a TensorDesc from the ggml_tensor parameters.

    Parameters:
        dtype: Tensor data type.
        ne: Number of elements in each dimension. Dimensions
            are from innermost to outermost (reverse of PyTorch).
        nb: Tensor stride in bytes for each dimension.
            Dimensions are from innermost to outermost (reverse of PyTorch).
        contiguous: Indicates if the tensor is contiguous in memory.

    Returns:
        TensorDesc: A new TensorDesc instance.

    """
    return TensorDesc(dtype=dtype, shape=ne, stride=nb, contiguous=contiguous)
