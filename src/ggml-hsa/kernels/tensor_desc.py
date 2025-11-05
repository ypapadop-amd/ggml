# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from dataclasses import dataclass
import numpy as np

from aie.iron import str_to_dtype


@dataclass(frozen=True)
class TensorDesc:
    """
    ggml_tensor description.

    Attributes:
        dtype: Data type of the tensor.
        shape (tuple): Shape of the tensor as a tuple of integers. Dimensions are from innermost to outermost (reverse of PyTorch).
        stride (tuple): Stride of the tensor as a tuple of integers, or None if not specified. Dimensions are from innermost to outermost (reverse of PyTorch).
        contiguous (bool): Indicates if the tensor is contiguous in memory.
    """

    dtype: np.dtype | str
    shape: tuple[int, int, int, int]
    stride: tuple[int, int, int, int] | None = None
    contiguous: bool = True

    def __post_init__(self):
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
        """
        Returns the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return int(np.prod(self.shape))

    def numel(self):
        """
        Returns the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return self.size


def ggml_tensor_to_tensordesc(
    type: str,
    ne: tuple[int, int, int, int],
    nb: tuple[int, int, int, int],
    contiguous: bool,
) -> TensorDesc:
    """
    Creates a TensorDesc from the ggml_tensor parameters.

    Parameters:
        type: Tensor data type.
        ne (tuple[int, int, int, int]): Number of elements in each dimension. Dimensions are from innermost to outermost (reverse of PyTorch).
        nb (tuple[int, int, int, int]): Tensor stride in bytes for each dimension. Dimensions are from innermost to outermost (reverse of PyTorch).
        contiguous (bool): Indicates if the tensor is contiguous in memory.

    Returns:
        TensorDesc: A new TensorDesc instance.
    """
    return TensorDesc(dtype=type, shape=ne, stride=nb, contiguous=contiguous)
