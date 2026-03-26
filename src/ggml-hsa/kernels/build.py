# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""GGML HSA backend kernel build system.

This module provides the infrastructure for compiling kernels to executable code
for AMD XDNA / XDNA2 devices. It handles mapping GGML operations to their corresponding
kernel implementations, dynamic module loading, and orchestrating the compilation
pipeline.

The build system supports multiple compilation backends with per-operation dispatch
based on compilation parameters.

Usage:
    As a module:
        from kernels import ggml_compile_op, TensorDesc
        ggml_compile_op(ggml_op="ADD", arch="aie2", ...)

    As a script:
        python build.py --ggml_op ADD --arch aie2 --input_tensors "(1024,1,1,1)/f32" ...
"""

import contextlib
import importlib.util
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path

from build_iron import compile_iron_kernel
from kernel import Backend, Kernel, KernelSpec
from tensor_desc import TensorDesc

# Compiler registry mapping Backend enum to compile functions
_compilers: dict[Backend, Callable] = {
    Backend.IRON: compile_iron_kernel,
}

# Mapping of GGML operations to kernel source files.
# Each entry maps an operation name to a Kernel that identifies the dispatch module.
_op_to_kernel_map: dict[str, Kernel] = {
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
    "ARGMAX": Kernel("ggml_op_argmax", "argmax.py"),
    "COUNT_EQUAL": Kernel("ggml_op_count_equal", "count_equal.py"),
    "CROSS_ENTROPY_LOSS": Kernel("ggml_op_cross_entropy_loss", "cross_entropy_loss.py"),
}


def get_compiler(backend: Backend) -> Callable:
    """Get the compiler function for the given backend.

    Parameters:
        backend: The compilation backend to use.

    Returns:
        The compiler function for the specified backend.

    Raises:
        NotImplementedError: If the backend is not implemented.

    Note:
        Uses backend.name for lookup to handle the case where Backend enums
        from dynamically imported modules have different identity than those
        in this module.

    """
    # Lookup by name to handle different enum class identities from dynamic imports
    for registered_backend, compiler in _compilers.items():
        if registered_backend.name == backend.name:
            return compiler
    msg = f"Backend {backend.name} not implemented."
    raise NotImplementedError(msg)


def get_kernel(op_name: str) -> Kernel:
    """Get the kernel for the given operation.

    Parameters:
        op_name: Operation name.

    Returns:
        The Kernel object associated with the operation.

    Raises:
        NotImplementedError: If the Kernel is not found.

    """
    kernel = _op_to_kernel_map.get(op_name)
    if kernel is None:
        msg = f"Operation {op_name} not implemented."
        raise NotImplementedError(msg)
    return kernel


def import_from_path(module_name: str, path: str | Path):
    """Import a module by name from the specified file path.

    This function handles the complexity of importing Python modules dynamically,
    including setting up the package structure for relative imports.

    Parameters:
        module_name: Name of the module to import.
        path: Path to the Python file containing the module.

    Returns:
        The imported module object.

    Raises:
        ImportError: If the module cannot be found or loaded.

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
        msg = f"Cannot find module spec for {module_name} at path {path}"
        raise ImportError(msg)
    if spec.loader is None:
        msg = f"Cannot find loader for module {module_name} at path {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Set __package__ to enable relative imports
    module.__package__ = package_name
    sys.modules[full_module_name] = module
    spec.loader.exec_module(module)
    return module


def ggml_compile_op(
    op_name: str,
    arch: str,
    input_tensors: list[TensorDesc | None],
    output_tensor: TensorDesc,
    op_params: bytearray,
    exported_name: str,
    output_directory: str | Path,
    verbose: bool = False,
) -> None:
    """Compile a GGML operation kernel to PDI and instruction files.

    This is the main entry point for kernel compilation. It:
    1. Looks up the kernel dispatch module for the operation
    2. Calls the dispatch function to get a KernelSpec (backend + function)
    3. Invokes the appropriate backend compiler

    Parameters:
        op_name: Operation name (e.g., "ADD", "MUL_MAT").
        arch: Target architecture (e.g., "aie2", "aie2p").
        input_tensors: List of input tensor descriptions.
        output_tensor: Output tensor description.
        op_params: Operation-specific parameters as a bytearray.
        exported_name: Name to export the compiled kernel as.
        output_directory: Directory to save the compiled PDI and instruction files.
        verbose: If True, enables verbose logging output.

    Raises:
        ValueError: If the operation is not supported.
        NotImplementedError: If the selected backend is not implemented.

    """
    # Setup logging
    logger = logging.getLogger(__name__)
    # remove all existing handlers
    for handler in logger.handlers.copy():
        # ignore double removals
        with contextlib.suppress(ValueError):
            logger.removeHandler(handler)
    if verbose:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Get kernel mapping
    kernel = get_kernel(op_name)

    # Load dispatch module and get dispatch function
    kernel_source_file = Path(__file__).resolve().parent / kernel.source_file
    module = import_from_path(kernel.name, kernel_source_file)
    dispatch_fn = getattr(module, kernel.name)

    # Dispatch to get KernelSpec (determines backend and function)
    kernel_spec: KernelSpec = dispatch_fn(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )

    # Create output and work directories
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / f"{exported_name}-artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "Compiling op: %s\n"
            "  Architecture:         %s\n"
            "  Backend:              %s\n"
            "  Op name:              %s\n"
            "  Kernel source:        %s\n"
            "  Input tensors:        %s\n"
            "  Output tensor:        %s\n"
            "  Operation parameters: %s\n"
            "  Exported name:        %s\n"
            "  Output directory:     %s\n"
            "  Working directory:    %s"
        ),
        op_name,
        arch,
        kernel_spec.backend.name,
        kernel_spec.op_name,
        str(kernel_source_file),
        kernel_spec.input_tensors,
        kernel_spec.output_tensor,
        kernel_spec.op_params,
        exported_name,
        str(output_dir),
        str(work_dir),
    )

    # Get compiler for the selected backend and compile
    compile_fn = get_compiler(kernel_spec.backend)
    compile_fn(
        kernel_spec=kernel_spec,
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
    """Convert a string of the form "(x,y,z,w)" to a tuple of integers.

    Parameters:
        string: String representation of a 4-element tuple.

    Returns:
        A tuple of 4 integers.

    Raises:
        ValueError: If the string does not represent exactly 4 integers.

    """
    string = string.replace("(", "").replace(")", "").strip(",")
    ints = map(int, string.split(","))
    t = tuple(ints)
    if len(t) != 4:
        msg = f"Shape must have 4 dimensions, got {len(t)}."
        raise ValueError(msg)
    return t


def to_tensordesc(string: str) -> TensorDesc:
    """Create a TensorDesc from a string representation.

    Parameters:
        string: String of the form "(shape)/dtype", e.g., "(1024,1,1,1)/f32".

    Returns:
        A TensorDesc instance with the specified shape and dtype.

    """
    shape, dtype = string.split("/")
    shape = to_tuple_of_ints(shape)
    return TensorDesc(dtype=dtype, shape=shape, stride=None)


def file_path(string: str):
    """Validate that a string represents an existing file path.

    Parameters:
        string: The file path to validate.

    Returns:
        The validated file path string.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    if not Path(string).is_file():
        raise FileNotFoundError(string)
    return string


def main() -> None:
    """Entry point for command-line AOT compilation."""
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="build.py",
        description="Compiles GGML HSA kernels for AMD XDNA / XDNA2 devices",
    )
    parser.add_argument(
        "--op_name",
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
        op_name=args.op_name,
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
