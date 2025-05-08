# compiler.py -*- Python -*-
#  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import subprocess
import importlib.util
import os
import sys
import numpy as np
from aie.iron.device import NPU1Col4, NPU2
from aie.iron.compile import compile_mlir_module_to_pdi
from ml_dtypes import bfloat16

peano_install_dir = os.getenv("PEANO_INSTALL_DIR")
if not os.path.isdir(peano_install_dir):
    raise RuntimeError("PEANO_INSTALL_DIR is not defined or does not exist")

peano_cxx = os.path.join(peano_install_dir, "bin/clang++")
if not os.path.isfile(peano_cxx):
    raise RuntimeError(f"Peano compiler not found in {peano_install_dir}")

mlir_aie_include_dir = os.path.join(os.getenv("MLIR_AIE_INSTALL_DIR"), "include")
if not os.path.isdir(mlir_aie_include_dir):
    raise RuntimeError(f"MLIR-AIE headers not found in {mlir_aie_include_dir}")

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


class TensorDesc:
    """
    Tensor description.

    This class provides the tensor information, such as shape and datatype.
    """

    def __init__(self, shape: tuple, dtype):
        self.shape = shape
        self.dtype = str_to_dtype(dtype)

    def __str__(self):
        return f"{str(self.shape)}/{str(self.dtype)}"

    def numel(self):
        """
        Calculates the number of elements in the tensor.

        Returns:
            int: The total number of elements in the tensor.
        """
        return int(np.prod(self.shape))


class CoreFunctionCompileSpec:
    """
    Core function compilation specification.

    This class provides information necessary to compile a single-core kernel via Peano.
    """

    def __init__(self, source_path: str, compile_args: list[str], output_filename: str):
        self.source_path = source_path
        self.compile_args = compile_args
        self.output_filename = output_filename

    def __str__(self):
        return f'Source:"{self.source_path}", Output:"{self.output_filename}", Compile args:"{self.compile_args}"'


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


def to_tuple_of_ints(string: str):
    """Converts a string of the form (x,...) to a tuple of ints."""
    string = string.replace("(", "").replace(")", "").strip(",")
    ints = map(int, string.split(","))
    return tuple(ints)


def to_tensor_desc(string: str) -> TensorDesc:
    """Converts a string of the form (x,...)/dtype to a TensorDesc object."""
    shape, dtype = string.split("/")
    shape = to_tuple_of_ints(shape)
    dtype = str_to_dtype(dtype)
    return TensorDesc(shape=shape, dtype=dtype)


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
):
    """
    Compiles the kernel code to PDI and instruction files.

    This function should be called when the compilation is initiated from another function (e.g., during JIT).
    """
    from aie.iron import set_current_device

    os.makedirs(output_directory, exist_ok=True)

    dev = to_device(device)

    # generate MLIR and write to file for debugging
    module = import_from_path(kernel_name, kernel_source)
    kernel_mlir = getattr(module, kernel_name)
    set_current_device(dev)
    mlir_module = kernel_mlir(input_tensors=input_tensors, output_tensor=output_tensor)
    mlir_path = os.path.join(output_directory, f"{exported_name}.mlir")
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # if there is a single-core spec, compile it with Peano
    if hasattr(module, "core_function_compile_spec"):
        core_function_compile_spec = getattr(module, "core_function_compile_spec")
        spec = core_function_compile_spec(
            device=dev, input_tensors=input_tensors, output_tensor=output_tensor
        )
        output_path = os.path.join(output_directory, spec.output_filename)
        cmd = [
            peano_cxx,
            spec.source_path,
            "-c",
            "-o",
            f"{output_path}",
            f"-I{mlir_aie_include_dir}",
            "-std=c++20",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-O2",
            "-DNDEBUG",
            f"--target={device}-none-unknown-elf",
        ] + spec.compile_args
        try:
            subprocess.run(
                cmd,
                cwd=output_directory,
                check=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError("Peano compilation failed") from ex

    # generate PDI and insts files
    pdi_path = os.path.join(output_directory, f"{exported_name}.pdi")
    insts_path = os.path.join(output_directory, f"{exported_name}_insts.bin")
    try:
        previous_cwd = os.getcwd()
        os.chdir(output_directory)
        compile_mlir_module_to_pdi(
            mlir_module=mlir_module,
            options=[
                "--alloc-scheme=basic-sequential",
                "--no-compile-host",
                "--no-xchesscc",
                "--no-xbridge",
                f"--peano={peano_install_dir}",
            ],
            insts_path=insts_path,
            pdi_path=pdi_path,
        )
    except:  # pylint: disable=try-except-raise
        raise
    finally:
        os.chdir(previous_cwd)


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
        type=to_tensor_desc,
        nargs="+",
        required=True,
        help="Input kernel tensor shapes and datatypes",
    )
    parser.add_argument(
        "--output_tensor",
        type=to_tensor_desc,
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
    args = parser.parse_args()

    compile_kernel(
        kernel_name=args.kernel_name,
        kernel_source=args.kernel_source,
        device=args.device,
        input_tensors=args.input_tensors,
        output_tensor=args.output_tensor,
        exported_name=args.exported_name,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
