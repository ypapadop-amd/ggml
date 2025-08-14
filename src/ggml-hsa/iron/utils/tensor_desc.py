# tensor_desc.py -*- Python -*-
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import numpy as np

from aie.iron import str_to_dtype


class TensorDesc:
    """ggml_tensor description.

    Attributes:
        dtype: Data type of the tensor.
        shape (tuple): Shape of the tensor as a tuple of integers. Dimensions are from innermost to outermost (reverse of PyTorch).
        stride (tuple): Stride of the tensor as a tuple of integers, or None if not specified. Dimensions are from innermost to outermost (reverse of PyTorch).
        size (int): Total number of elements in the tensor, calculated as the product of the shape dimensions.
        contiguous (bool): Indicates if the tensor is contiguous in memory.
    """

    def __init__(
        self,
        dtype,
        shape: tuple[int, int, int, int],
        stride,
        contiguous: bool,
    ):
        if len(shape) != 4:
            raise ValueError(f"Shape must be a tuple of 4 integers, got {shape}")
        if stride is not None and len(stride) != 4:
            raise ValueError(f"Stride must be a tuple of 4 integers, got {stride}")
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.size = int(np.prod(shape))
        self.stride = stride
        self.contiguous = contiguous

    def __repr__(self):
        """Returns a string representation of the TensorDesc."""
        if self.stride is not None:
            return (
                f"{self.__class__.__name__}"
                f"(dtype={str(self.dtype)} "
                f"shape={str(self.shape)} "
                f"stride={str(self.stride)} "
                f"contiguous={str(self.contiguous)})"
            )
        else:
            return (
                f"{self.__class__.__name__}"
                f"(dtype={str(self.dtype)} "
                f"shape={str(self.shape)} "
                f"contiguous={str(self.contiguous)})"
            )

    def numel(self):
        """Returns the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return self.size


def tensordesc(dtype, shape, stride=None, contiguous=True) -> TensorDesc:
    """Creates a TensorDesc from the specified shape and dtype.

    Parameters:
        dtype: Tensor data type.
        shape (tuple): Tensor number of elements in each dimension. Dimensions are from innermost to outermost (reverse of PyTorch).
        stride (tuple): Tensor stride in bytes for each dimension. Dimensions are from innermost to outermost (reverse of PyTorch).
        contiguous (bool): Indicates if the tensor is contiguous in memory.

    Returns:
        TensorDesc: A new TensorDesc instance.
    """
    if isinstance(dtype, str):
        dtype = str_to_dtype(dtype)
    return TensorDesc(dtype=dtype, shape=shape, stride=stride, contiguous=contiguous)
