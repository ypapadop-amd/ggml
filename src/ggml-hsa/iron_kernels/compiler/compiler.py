# compiler.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import subprocess
import importlib.util
import os
import sys
import numpy as np
from aie.iron.device import NPU1Col1
from aie.iron.compile import compile_mlir_module_to_pdi
from ml_dtypes import bfloat16


class TensorDesc:
    """
    Tensor description.

    This class provides the tensor information, such as shape and datatype.
    """

    def __init__(self, shape: tuple, dtype):
        self.shape = shape
        self.dtype = to_dtype(dtype)

    def __str__(self):
        return f"{str(self.shape)}/{str(self.dtype)}"

    def numel(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return int(np.prod(self.shape))


peano_install_dir = os.getenv("PEANO_INSTALL_DIR")
if not peano_install_dir:
    raise RuntimeError("PEANO_INSTALL_DIR is not defined")

peano_cxx = os.path.join(peano_install_dir, "bin/clang++")
if not os.path.isfile(peano_cxx):
    raise RuntimeError(f"Peano compile not found in {peano_install_dir}")

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


def to_device(device):
    """
    Returns the supported device from the string.
    """
    if isinstance(device, str):
        return supported_devices[device]
    return device


def to_dtype(dtype):
    """
    Returns the supported datatype from the string.
    """
    if isinstance(dtype, str):
        return supported_dtypes[dtype]
    return dtype


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


def has_single_core_solution(pkg):
    """
    Returns if a single-core solution exists for the package.
    """
    return "single_core_solution" in dir(pkg)


def compile_single_core(
    source: str,
    device: str,
    compile_args: str,
    object_filename: str,
    output_directory: str,
):
    """
    Compile a C++ file using Peano.
    """
    output_path = os.path.join(output_directory, object_filename)
    cmd = [
        sys.executable,
        peano_cxx,
        source,
        "-std=c++20",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-O2",
        "-DNDEBUG",
        f"--target={device}-none-unknown-elf",
    ] + list(compile_args.split())
    with open(output_path, "w", encoding="utf-8") as output_file:
        try:
            subprocess.run(
                cmd,
                cwd=output_directory,
                check=True,
                stdout=output_file,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError("Peano Compilation failed") from ex


def import_from_path(module_name, path):
    """
    Imports module_name from path.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_kernel(
    kernel_name: str,
    kernel_source: str,
    device: str,
    tensors: list[TensorDesc],
    exported_name: str,
    output_directory: str,
):
    """
    Compiles the kernel code to PDI and instruction files.

    This function should be called when the compilation is initiated from another function (e.g., during JIT).
    """
    from aie.iron import set_current_device

    os.makedirs(output_directory, exist_ok=True)

    device = to_device(device)

    # generate MLIR and write to file for debugging
    module = import_from_path(kernel_name, kernel_source)
    kernel_mlir = getattr(module, kernel_name)
    set_current_device(device)
    mlir_module = kernel_mlir(*tensors)
    mlir_path = os.path.join(output_directory, f"{exported_name}.mlir")
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module.body))

    # generate PDI and insts files
    pdi_path = os.path.join(output_directory, f"{exported_name}.pdi")
    insts_path = os.path.join(output_directory, f"{exported_name}_insts.bin")
    compile_mlir_module_to_pdi(
        mlir_module=mlir_module,
        insts_path=insts_path,
        pdi_path=pdi_path,
        options=[
            "--alloc-scheme=basic-sequential",
            "--no-compile-host",
            "--no-xchesscc",
            "--no-xbridge",
            f"--peano={peano_install_dir}",
        ],
    )


def file_path(string: str):
    """
    Checks if a string is an existing file.
    """
    if not os.path.isfile(string):
        raise FileNotFoundError(string)
    return string


def main():
    """
    Main function for use during AOT compilation.
    """

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
        "--tensors",
        type=to_tensor_desc,
        nargs="+",
        required=True,
        help="Kernel tensor shapes and datatypes",
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
    args = parser.parse_args()

    compile_kernel(
        kernel_name=args.kernel_name,
        kernel_source=args.kernel_source,
        device=args.device,
        tensors=args.tensors,
        exported_name=args.exported_name,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
