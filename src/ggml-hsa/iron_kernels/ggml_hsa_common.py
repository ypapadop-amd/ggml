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

supported_devices = ["aie2"]

supported_dtypes = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "f32": np.float32,
}


def create_device(device_name):
    """Return the device from the given device name."""
    if device_name == "aie2":
        dev = NPU1Col1()
    else:
        raise ValueError(f"Device name {device_name} is unknown")
    return dev


def create_dtype(dtype_name):
    """Return the numpy datatype from the datatype name."""
    return np.dtype[supported_dtypes[dtype_name]]


def create_dims(dims_str):
    """Return a tuple of dimensions from the string."""
    dims_str = dims_str.replace("(", "").replace(")", "")
    dims_ints = map(int, dims_str.split(","))
    return tuple(dims_ints)
