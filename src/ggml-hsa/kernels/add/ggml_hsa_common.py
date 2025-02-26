# add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import numpy as np

from aie.iron.device import NPU1Col1, NPU2
from ml_dtypes import bfloat16

supported_devices = ["npu", "npu2"]

supported_dtypes = {
    "bfloat16_t": bfloat16,
    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "float": np.float32,
}


def create_device(device_name):
    """Return the device from the given device name."""
    if device_name == "npu":
        dev = NPU1Col1()
    elif device_name == "npu2":
        dev = NPU2()
    else:
        raise ValueError("Device name {} is unknown".format(device_name))
    return dev


def create_dtype(dtype_name):
    """Return the numpy datatype from the datatype name."""
    return np.dtype[supported_dtypes[dtype_name]]


def create_dims(dims_str):
    """Return a tuple of dimensions from the string."""
    dims_str = dims_str.replace("(", "").replace(")", "")
    dims_ints = map(int, dims_str.split(","))
    return tuple(dims_ints)
