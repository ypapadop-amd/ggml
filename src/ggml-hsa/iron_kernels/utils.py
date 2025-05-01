# add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np

from aie.iron.device import NPU1Col1
from ml_dtypes import bfloat16

supported_devices = {
    "aie2": NPU1Col1(),
}

supported_dtypes = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "f32": np.float32,
}


def to_device(device: str):
    """
    Returns the supported device from the string.
    """
    return supported_devices[device]


def to_dtype(dtype: str):
    """
    Returns the supported datatype from the string.
    """
    return supported_dtypes[dtype]


class TensorDesc:
    """
    Tensor description.

    This class provides the tensor information, such as shape and datatype.
    """

    def __init__(self, shape: tuple, dtype):
        self.shape = shape
        self.dtype = dtype

    def __str__(self):
        return f"{str(self.shape)}/{str(self.dtype)}"

    def numel(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return int(np.prod(self.shape))


def to_tuple_of_ints(string: str):
    """
    Converts a string of the form (x,...) to a tuple of ints.
    """
    string = string.replace("(", "").replace(")", "").strip(",")
    ints = map(int, string.split(","))
    return tuple(ints)


def to_tensor_desc(string: str) -> TensorDesc:
    """
    Converts a string of the form (x,...)/dtype to a TensorDesc object.
    """
    shape, dtype = string.split("/")
    shape = to_tuple_of_ints(shape)
    dtype = to_dtype(dtype)
    return TensorDesc(shape=shape, dtype=dtype)
