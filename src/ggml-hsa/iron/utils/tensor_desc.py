# tensor_desc.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np
from ml_dtypes import bfloat16


supported_dtypes = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "f32": np.float32,
}


def dtype_to_str(dtype):
    """Returns the datatype as a string."""
    if isinstance(dtype, str):
        return dtype
    for key, value in supported_dtypes.items():
        if value == dtype:
            return key
    return None


class TensorDesc:
    """
    Tensor description.

    This class provides the tensor information, such as shape and datatype.

    The shape is a tuple of integers, where the innermost dimension is first.
    """

    def __init__(self, shape: tuple, dtype):
        if len(shape) != 4:
            raise ValueError(
                f"Shape must be a tuple of 4 integers, got {shape} with length {len(shape)}"
            )
        self.shape = shape
        self.size = int(np.prod(shape))
        self.dtype = np.dtype(dtype)

    def __str__(self):
        return f"{str(self.shape)}/{str(self.dtype)}"

    def numel(self):
        """
        Returns the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return self.size


def tensordesc(shape, dtype) -> TensorDesc:
    """
    Creates a TensorDesc from the specified shape and dtype.

    Parameters:
        shape(tuple): Tensor shape. This follows the GGML convention, where dimensions are from innermost to outermost (reverse of PyTorch).
        dtype: Tensor data type.

    Returns:
        TensorDesc: A new TensorDesc instance.
    """
    if isinstance(dtype, str):
        dtype = supported_dtypes[dtype]
    return TensorDesc(shape=shape, dtype=dtype)
