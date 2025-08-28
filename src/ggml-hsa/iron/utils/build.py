# compiler.py -*- Python -*-
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import importlib.util
import os
import sys
import logging

import aie.iron
import aie.iron.compile
import aie.iron.device

from tensor_desc import tensordesc, TensorDesc


def arch_to_device(device):
    """Returns the device from the string."""
    if isinstance(device, str):
        if device == "aie2":
            return aie.iron.device.NPU1()
        elif device == "aie2p":
            return aie.iron.device.NPU2()
        else:
            raise ValueError(f"Unsupported device: {device}")
    return device


def import_from_path(module_name: str, path: os.PathLike):
    """
    Imports the module with name module_name from path.

    Parameters:
        module_name (str): Name of the module.
        path (os.PathLike): Path to the module file.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_kernel(
    kernel_name: str,
    kernel_source: os.PathLike,
    device: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    exported_name: str,
    output_directory: os.PathLike,
    verbose: bool = False,
):
    """
    Compiles the kernel code to PDI and instruction files.

    Parameters:
        kernel_name (str): Name of the IRON kernel.
        kernel_source (str): Path to the IRON kernel source file.
        device (str): Target device for compilation.
        input_tensors (list[TensorDesc]): List of input tensor descriptions.
        output_tensor (TensorDesc): Output tensor description.
        exported_name (str): Name to export the compiled kernel as.
        output_directory (str): Directory to save the compiled files.
        verbose (bool): If True, enables verbose logging.
    """

    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info(
        "Compiling kernel: %s\n\tKernel source: %s\n\tDevice: %s\n\tInput tensors: %s\n\tOutput tensor: %s\n\tExported name: %s\n\tOutput directory: %s",
        kernel_name,
        kernel_source,
        device,
        input_tensors,
        output_tensor,
        exported_name,
        output_directory,
    )

    # create output and work directories
    os.makedirs(output_directory, exist_ok=True)
    work_dir = os.path.join(output_directory, f"{exported_name}-artifacts")
    os.makedirs(work_dir, exist_ok=True)

    # find IRON kernel
    module = import_from_path(kernel_name, kernel_source)
    kernel = getattr(module, kernel_name)

    # compile core function if it exists
    core_function_info = None
    try:
        core_function_info_func = getattr(kernel, "core_function_info")
        core_function_info = core_function_info_func(
            device=device, input_tensors=input_tensors, output_tensor=output_tensor
        )
        output_path = os.path.join(work_dir, kernel_name + ".o")
        aie.iron.compile.compile_cxx_core_function(
            source_path=core_function_info.source_file,
            target_arch=device,
            output_path=output_path,
            compile_args=core_function_info.compile_args,
            cwd=work_dir,
            verbose=verbose,
        )
        core_function_info.object_file = output_path
        logger.info(
            "Core function found for kernel %s: %s", kernel_name, core_function_info
        )
    except AttributeError:
        # ignore missing attribute
        logger.info("No core function found for kernel %s", kernel_name)

    # remove any existing external functions
    aie.iron.ExternalFunction._instances.clear()

    # generate MLIR module
    if core_function_info:  # probably not needed
        mlir_module = kernel(
            device=device,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            core_function_info=core_function_info,
        )
    else:
        mlir_module = kernel(
            device=device,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        )

    # compile any external functions
    for func in aie.iron.kernel.ExternalFunction._instances:
        output_path = os.path.join(work_dir, func._object_file_name)
        aie.iron.compile.compile_cxx_core_function(
            source_path=func._source_file,
            target_arch=device,
            output_path=output_path,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=work_dir,
            verbose=verbose,
        )

    # remove generated external functions
    aie.iron.ExternalFunction._instances.clear()

    # write MLIR module to file
    mlir_path = os.path.join(work_dir, f"{exported_name}.mlir")
    logger.info("Writing MLIR module for kernel %s in %s", kernel_name, mlir_path)
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # generate PDI and instructions files
    pdi_path = os.path.join(output_directory, f"{exported_name}.pdi")
    insts_path = os.path.join(output_directory, f"{exported_name}_insts.bin")
    current_directory = os.getcwd()
    try:
        logger.info("Changing working directory to %s", work_dir)
        # change directory to avoid core files in current path
        os.chdir(work_dir)
        aie.iron.compile.compile_mlir_module(
            mlir_module=mlir_module,
            options=["--alloc-scheme=basic-sequential"],
            insts_path=insts_path,
            pdi_path=pdi_path,
            verbose=verbose,
            work_dir=work_dir,
        )
    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Failed to change working directory to %s", work_dir)
        raise
    finally:
        os.chdir(current_directory)
    logger.info(
        "Finished compilation for kernel %s in %s", kernel_name, output_directory
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
        TensorDesc: A new TensorDesc instance.
    """
    shape, dtype = string.split("/")
    shape = to_tuple_of_ints(shape)
    return tensordesc(dtype=dtype, shape=shape)


def file_path(string: str):
    """Checks if a string is an existing file."""
    if not os.path.isfile(string):
        raise FileNotFoundError(string)
    return string


def main():
    """Main function for use during AOT compilation."""

    from argparse import ArgumentParser  # pylint: disable=import-outside-toplevel

    parser = ArgumentParser(
        prog="build.py",
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
