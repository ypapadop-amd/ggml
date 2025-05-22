# compiler.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import importlib.util
import os
import sys
from typing import Callable

import numpy as np

from aie.iron import set_current_device, get_current_device
from aie.iron.device import NPU1Col4, NPU2
from aie.iron.compile import compile_cxx_core_function, compile_mlir_module_to_pdi
from ml_dtypes import bfloat16

supported_devices = {
    "aie2": NPU1Col4(),
    "aie2p": NPU2(),
}

supported_dtypes = {
    "bf16": bfloat16,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "f32": np.float32,
}


def to_device(device):
    """Returns the device from the string."""
    if isinstance(device, str):
        return supported_devices[device]
    return device


def str_to_dtype(dtype):
    """Returns the datatype from the string."""
    if isinstance(dtype, str):
        return supported_dtypes[dtype]
    return dtype


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
    """

    def __init__(self, shape: tuple, dtype):
        self.shape = shape
        self.size = int(np.prod(shape))
        self.dtype = str_to_dtype(dtype)

    def __str__(self):
        return f"{str(self.shape)}/{str(self.dtype)}"

    def numel(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return self.size


def tensordesc(shape, dtype) -> TensorDesc:
    """
    Creates a TensorDesc from the specified shape and dtype.

    Parameters:
        shape(tuple): Tensor shape.
        dtype (np.dtype, optional): Desired data type.

    Returns:
        TensorDesc: A new TensorDesc instance.
    """
    return TensorDesc(shape=shape, dtype=dtype)


class CoreFunctionInfo:
    """
    Core function information.

    This class provides information necessary to compile a core function via Peano and use it in a kernel.
    """

    def __init__(self, source_file: str, exported_function, compile_args):
        self.source_file = source_file
        if compile_args is None:
            self.compile_args = []
        else:
            self.compile_args = compile_args
        self.exported_function = exported_function
        self.object_file = None

    def __str__(self):
        return f'Source file: "{self.source_file}", Compile args: {self.compile_args}, Exported function: "{self.exported_function}", Object file: "{self.object_file}"'


def core_function(function_info=None) -> Callable:
    """Associates a core function with a kernel."""

    def wrapper(func):
        if function_info:
            func.core_function_info = function_info
        return func

    return wrapper


def import_from_path(module_name, path):
    """Imports the module with name module_name from path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_kernel(
    kernel_name: str,
    kernel_source: str,
    device: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    exported_name: str,
    output_directory: str,
    verbose: bool = False,
):
    """Compiles the kernel code to PDI and instruction files."""
    os.makedirs(output_directory, exist_ok=True)

    # import IRON kernel
    module = import_from_path(kernel_name, kernel_source)
    kernel = getattr(module, kernel_name)

    # compile core function if it exists
    core_function_info = None
    try:
        core_function_info_func = getattr(kernel, "core_function_info")
        core_function_info = core_function_info_func(
            device=device, input_tensors=input_tensors, output_tensor=output_tensor
        )
        output_path = os.path.join(output_directory, exported_name + ".o")
        compile_cxx_core_function(
            source_path=core_function_info.source_file,
            target_arch=device,
            output_path=output_path,
            compile_args=core_function_info.compile_args,
            cwd=output_directory,
            verbose=verbose,
        )
        core_function_info.object_file = output_path
    except AttributeError:
        # ignore missing attribute
        pass

    # generate MLIR and write to file for debugging
    dev = to_device(device)
    set_current_device(dev)
    if core_function_info:
        mlir_module = kernel(
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            core_function_info=core_function_info,
        )
    else:
        mlir_module = kernel(input_tensors=input_tensors, output_tensor=output_tensor)
    mlir_path = os.path.join(output_directory, f"{exported_name}.mlir")
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # generate PDI and insts files
    pdi_path = os.path.join(output_directory, f"{exported_name}.pdi")
    insts_path = os.path.join(output_directory, f"{exported_name}_insts.bin")
    compile_mlir_module_to_pdi(
        mlir_module=mlir_module,
        options=["--alloc-scheme=basic-sequential"],
        insts_path=insts_path,
        pdi_path=pdi_path,
        verbose=verbose,
    )


def to_tuple_of_ints(string: str):
    """Converts a string of the form (x,...) to a tuple of ints."""
    string = string.replace("(", "").replace(")", "").strip(",")
    ints = map(int, string.split(","))
    return tuple(ints)


def to_tensordesc(string: str) -> TensorDesc:
    """
    Creates a TensorDesc from the string.

    Parameters:
        string (str): string of the form (shape)/dtype.

    Returns:
    Returns:
        TensorDesc: A new TensorDesc instance.
    """
    shape, dtype = string.split("/")
    shape = to_tuple_of_ints(shape)
    dtype = str_to_dtype(dtype)
    return TensorDesc(shape=shape, dtype=dtype)


def file_path(string: str):
    """Checks if a string is an existing file."""
    if not os.path.isfile(string):
        raise FileNotFoundError(string)
    return string


def main():
    """Main function for use during AOT compilation."""

    from argparse import ArgumentParser  # pylint: disable=import-outside-toplevel

    parser = ArgumentParser(
        prog="compiler.py",
        description="Compiles IRON kernels",
    )
    parser.add_argument(
        "--kernel_name",
        type=str,
        required=True,
        help="Kernel name",
    )
    parser.add_argument(
        "--kernel_source",
        type=file_path,
        required=True,
        help="Kernel source file",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Target device",
    )
    parser.add_argument(
        "--input_tensors",
        type=to_tensordesc,
        nargs="+",
        required=True,
        help="Input kernel tensor shapes and datatypes",
    )
    parser.add_argument(
        "--output_tensor",
        type=to_tensordesc,
        required=True,
        help="Output kernel tensor shape and datatype",
    )
    parser.add_argument(
        "--exported_name",
        type=str,
        required=True,
        help="Kernel exported name",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    compile_kernel(
        kernel_name=args.kernel_name,
        kernel_source=args.kernel_source,
        device=args.device,
        input_tensors=args.input_tensors,
        output_tensor=args.output_tensor,
        exported_name=args.exported_name,
        output_directory=args.output_directory,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
