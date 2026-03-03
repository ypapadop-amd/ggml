# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
GGML HSA backend kernel build system.

This module provides the infrastructure for compiling kernels to executable code
for AMD XDNA / XDNA2 devices. It handles mapping GGML operations to their corresponding
kernel implementations, dynamic module loading, and orchestrating the compilation
pipeline.

Usage:
    As a module:
        from kernels import ggml_compile_op, TensorDesc
        ggml_compile_op(ggml_op="ADD", arch="aie2", ...)

    As a script:
        python build.py --ggml_op ADD --arch aie2 --input_tensors "(1024,1,1,1)/f32" ...
"""

import dataclasses
import importlib.util
import logging
from pathlib import Path
import sys
import types

from kernel import Kernel
from tensor_desc import TensorDesc
from build_iron import compile_iron_kernel

# mapping of GGML operations (unary, binary, and others) to kernel source files
op_to_kernel_map = {
    # unary operation to kernel source mapping
    "ABS": Kernel("ggml_unary_op_abs", "unary_ops.py"),
    "SGN": Kernel("ggml_unary_op_sgn", "unary_ops.py"),
    "NEG": Kernel("ggml_unary_op_neg", "unary_ops.py"),
    "STEP": Kernel("ggml_unary_op_step", "unary_ops.py"),
    "TANH": Kernel("ggml_unary_op_tanh", "unary_ops.py"),
    "ELU": Kernel("ggml_unary_op_elu", "unary_ops.py"),
    "RELU": Kernel("ggml_unary_op_relu", "unary_ops.py"),
    "SIGMOID": Kernel("ggml_unary_op_sigmoid", "unary_ops.py"),
    "GELU": Kernel("ggml_unary_op_gelu", "unary_ops.py"),
    "GELU_QUICK": Kernel("ggml_unary_op_gelu_quick", "unary_ops.py"),
    "SILU": Kernel("ggml_unary_op_silu", "unary_ops.py"),
    "HARDSWISH": Kernel("ggml_unary_op_hardswish", "unary_ops.py"),
    "HARDSIGMOID": Kernel("ggml_unary_op_hardsigmoid", "unary_ops.py"),
    "EXP": Kernel("ggml_unary_op_exp", "unary_ops.py"),
    "GELU_ERF": Kernel("ggml_unary_op_gelu_erf", "unary_ops.py"),
    "XIELU": Kernel("ggml_unary_op_xielu", "unary_ops.py"),
    "FLOOR": Kernel("ggml_unary_op_floor", "unary_ops.py"),
    "CEIL": Kernel("ggml_unary_op_ceil", "unary_ops.py"),
    "ROUND": Kernel("ggml_unary_op_round", "unary_ops.py"),
    "TRUNC": Kernel("ggml_unary_op_trunc", "unary_ops.py"),
    # operation to kernel source mapping
    "ADD": Kernel("ggml_op_add", "binary_ops.py"),
    "SUB": Kernel("ggml_op_sub", "binary_ops.py"),
    "MUL": Kernel("ggml_op_mul", "binary_ops.py"),
    "DIV": Kernel("ggml_op_div", "binary_ops.py"),
    "SQR": Kernel("ggml_op_sqr", "unary_ops.py"),
    "SQRT": Kernel("ggml_op_sqrt", "unary_ops.py"),
    "LOG": Kernel("ggml_op_log", "unary_ops.py"),
    "SIN": Kernel("ggml_op_sin", "unary_ops.py"),
    "COS": Kernel("ggml_op_cos", "unary_ops.py"),
    "MUL_MAT": Kernel("ggml_op_mul_mat", "mul_mat.py"),
    "SCALE": Kernel("ggml_op_scale", "scale.py"),
    "SOFT_MAX": Kernel("ggml_op_soft_max", "soft_max.py"),
    "CLAMP": Kernel("ggml_op_clamp", "clamp.py"),
}


def import_from_path(module_name: str, path: str | Path):
    """
    Imports the module with name module_name from path.

    Parameters:
        module_name (str): Name of the module.
        path (Path): Path to the module file.
    """
    path = Path(path).resolve()
    parent_dir = path.parent
    grandparent_dir = parent_dir.parent

    # Add grandparent directory to sys.path so package imports work
    grandparent_str = str(grandparent_dir)
    if grandparent_str not in sys.path:
        sys.path.insert(0, grandparent_str)

    # Create a package name from the directory for relative imports
    package_name = parent_dir.name

    # Ensure the parent package exists in sys.modules
    parent_dir_str = str(parent_dir)
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [parent_dir_str]
        pkg.__package__ = package_name
        sys.modules[package_name] = pkg

    # Create spec with submodule_search_locations for package support
    full_module_name = f"{package_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        full_module_name,
        path,
        submodule_search_locations=[parent_dir_str],
    )
    if spec is None:
        raise ImportError(f"Cannot find module spec for {module_name} at path {path}")
    if spec.loader is None:
        raise ImportError(f"Cannot find loader for module {module_name} at path {path}")
    module = importlib.util.module_from_spec(spec)
    # Set __package__ to enable relative imports
    module.__package__ = package_name
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)
    return module


def ggml_compile_op(
    ggml_op: str,
    arch: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    op_params: bytearray,
    exported_name: str,
    output_directory: str | Path,
    verbose: bool = False,
):
    """
    Compiles the kernel code corresponding to the GGML operation to PDI and
    instruction files.

    Parameters:
        ggml_op (str): GGML operation.
        arch (str): Target architecture.
        input_tensors (list[TensorDesc]): List of input tensor descriptions.
        output_tensor (TensorDesc): Output tensor description.
        exported_name (str): Name to export the compiled kernel as.
        output_directory (str): Directory to save the compiled files.
        verbose (bool): If True, enables verbose logging.
    """

    # setup logging
    logger = logging.getLogger(__name__)
    # remove all existing handlers
    for handler in logger.handlers.copy():
        try:
            logger.removeHandler(handler)
        except ValueError:
            # ignore double removals
            pass
    if verbose:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # get kernel implementation based on operation
    kernel = op_to_kernel_map.get(ggml_op, None)
    if not kernel:
        raise ValueError(f"Unsupported GGML operation: {ggml_op}")
    kernel_source_file = Path(__file__).resolve().parent / kernel.source_file
    module = import_from_path(kernel.name, kernel_source_file)
    kernel_fn = getattr(module, kernel.name)
    kernel = dataclasses.replace(
        kernel,
        source_file=kernel_source_file,
        function=kernel_fn,
    )

    # create output and work directories
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / f"{exported_name}-artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "Compiling op: %s for arch %s\n"
            "  Kernel name:          %s\n"
            "  Kernel source:        %s\n"
            "  Input tensors:        %s\n"
            "  Output tensor:        %s\n"
            "  Operation parameters: %s\n"
            "  Exported name:        %s\n"
            "  Output directory:     %s\n"
            "  Working directory:    %s"
        ),
        ggml_op,
        arch,
        kernel.name,
        kernel.source_file,
        input_tensors,
        output_tensor,
        op_params,
        exported_name,
        output_directory,
        work_dir,
    )

    # compile kernel
    compile_iron_kernel(
        kernel=kernel,
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        work_dir=work_dir,
        exported_name=exported_name,
        output_directory=output_dir,
        logger=logger,
        verbose=verbose,
    )

    logger.info(
        "Finished compilation for kernel %s in %s", kernel.name, output_directory
    )


def to_tuple_of_ints(string: str) -> tuple[int, int, int, int]:
    """Converts a string of the form (x,...) to a tuple of ints."""
    string = string.replace("(", "").replace(")", "").strip(",")
    ints = map(int, string.split(","))
    t = tuple(ints)
    if len(t) != 4:
        raise ValueError(f"Shape must have 4 dimensions, got {len(t)}.")
    return t


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
    return TensorDesc(dtype=dtype, shape=shape, stride=None)


def file_path(string: str):
    """Checks if a string is an existing file."""
    if not Path(string).is_file():
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
        "--ggml_op",
        type=str,
        required=True,
        help="GGML operation name, e.g., MUL_MAT, ADD, RELU, etc.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Target architecture",
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

    ggml_compile_op(
        ggml_op=args.ggml_op,
        arch=args.arch,
        input_tensors=args.input_tensors,
        output_tensor=args.output_tensor,
        op_params=bytearray(),
        exported_name=args.exported_name,
        output_directory=args.output_directory,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
