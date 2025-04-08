# add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2025 Advanced Micro Devices, Inc. or its affiliates

import argparse
import sys
import numpy as np

from ggml_hsa_common import (
    create_device,
    create_dtype,
    create_dims,
    supported_devices,
    supported_dtypes,
)
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


def vector_add(dev, dtype, dims):
    if len(dims) > 1:
        raise ValueError("Function accepts only 1D dims")

    n = 16
    N_div_n = dims[0] // n

    # Define tensor types
    tensor_ty = np.ndarray[dims, dtype]
    tile_ty = np.ndarray[(n,), dtype]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] + elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


def main():
    parser = argparse.ArgumentParser(
        prog="add.py",
        description="AIE Vector Add MLIR Design (Whole Array)",
    )
    parser.add_argument(
        "--dev",
        type=str,
        required=True,
        choices=supported_devices,
        help="Target device",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=list(supported_dtypes.keys()),
        help="Input and output vector sizes",
    )
    parser.add_argument(
        "--dims", type=str, required=True, help="Input and output vector sizes"
    )
    args = parser.parse_args()

    dev = create_device(args.dev)
    dtype = create_dtype(args.dtype)
    dims = create_dims(args.dims)
    module = vector_add(dev, dtype, dims)
    print(module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
