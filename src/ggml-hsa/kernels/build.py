# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""GGML HSA kernel build system for AMD XDNA / XDNA2 devices.

Maps GGML operations to kernel implementations, dynamically loads dispatch
modules, and orchestrates compilation across multiple backends with
per-operation dispatch.

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
from dataclasses import dataclass
from pathlib import Path

from kernel import Backend, Kernel
from tensor_desc import TensorDesc


@dataclass(frozen=True)
class CompilerConfig:
    """Compiler configuration.

    Attributes:
        output_directory: Destination for compilation artifacts.
        compilers: Ordered backend names (case-insensitive, e.g. ("iron",
            "triton")) to try in order. A candidate KernelSpec whose backend is
            not listed is dropped. Empty (the default) keeps the dispatch
            function's order unchanged.
        verbose: If True, enables verbose logging output.

    """

    output_directory: str | Path
    compilers: tuple[str, ...] = ()
    verbose: bool = False


def _make_kernel_specs(
    dispatch_result, compilers: tuple[str, ...], logger: logging.Logger
) -> list:
    """Normalize a dispatch result into an ordered list of KernelSpecs.

    The build system compiles the returned specs in order and uses the first that
    succeeds, falling back to the next. Specs are grouped by the position of their
    backend name in ``compilers`` (case-insensitive); a spec whose backend is not
    listed is dropped. An empty ``compilers`` keeps the dispatch function's order
    unchanged (no reordering, no dropping).

    Args:
        dispatch_result: A single KernelSpec or a list of KernelSpecs, as returned
            by a dispatch function.
        compilers: Ordered backend names to try (e.g. ("iron", "triton")).
        logger: Logger used to warn when the compiler order drops every candidate.

    Raises:
        ValueError: If ``compilers`` contains a name that is not a known backend.

    Returns:
        The listed specs in ``compilers`` order; original order is preserved
        within a backend group. The specs unchanged if ``compilers`` is empty.
    """
    specs = dispatch_result if isinstance(dispatch_result, list) else [dispatch_result]
    if not compilers:
        return specs
    known = {b.name.lower() for b in Backend}
    if unknown := [name for name in compilers if name.lower() not in known]:
        msg = f"Unknown backend(s) {unknown} in compiler order; known backends are {sorted(known)}."
        raise ValueError(msg)
    order = {name.lower(): i for i, name in enumerate(compilers)}
    listed = sorted(
        (s for s in specs if s.backend.name.lower() in order),
        key=lambda s: order[s.backend.name.lower()],
    )
    if not listed:
        available = sorted({s.backend.name.lower() for s in specs})
        logger.warning(
            "Compiler order %s dropped all candidates; this op only provides backend(s) %s. "
            "Add one of them to GGML_HSA_JIT_COMPILER_ORDER or unset it to use the default order.",
            list(compilers),
            available,
        )
    return listed


def _get_compiler(backend: Backend) -> Callable:
    """Return the compiler function for the given backend.

    Args:
        backend: The backend whose compiler function to return.

    Raises:
        NotImplementedError: If the backend is not implemented.

    Note:
        Backend compilers are imported lazily so that an IRON-only environment
        (without the Triton/torch dependencies) can still compile IRON kernels.
        Lookup is by ``backend.name`` because Backend enums from dynamically
        imported modules have different identity than those defined here.
    """
    if backend.name == Backend.IRON.name:
        from build_iron import compile_iron_kernel

        return compile_iron_kernel
    if backend.name == Backend.TRITON.name:
        from build_triton import compile_triton_kernel

        return compile_triton_kernel
    msg = f"Backend {backend.name} not implemented."
    raise NotImplementedError(msg)


# Maps each GGML operation name to a Kernel identifying its dispatch module.
_OP_KERNEL_MAP: dict[str, Kernel] = {
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
    "POOL_2D": Kernel("ggml_op_pool_2d", "pool_2d.py"),
    "IM2COL": Kernel("ggml_op_im2col", "im2col.py"),
    "SCALE": Kernel("ggml_op_scale", "scale.py"),
    "NORM": Kernel("ggml_op_norm", "norm.py"),
    "SOFT_MAX": Kernel("ggml_op_soft_max", "soft_max.py"),
    "CLAMP": Kernel("ggml_op_clamp", "clamp.py"),
    "ARGMAX": Kernel("ggml_op_argmax", "argmax.py"),
    "COUNT_EQUAL": Kernel("ggml_op_count_equal", "count_equal.py"),
    "CROSS_ENTROPY_LOSS": Kernel("ggml_op_cross_entropy_loss", "cross_entropy_loss.py"),
    "CONV_2D": Kernel("ggml_op_conv_2d", "conv_2d.py"),
}


def _get_kernel(op_name: str) -> Kernel:
    """Return the Kernel for the given operation.

    Args:
        op_name: Operation name to look up.

    Raises:
        NotImplementedError: If the Kernel is not found.
    """
    try:
        return _OP_KERNEL_MAP[op_name]
    except KeyError:
        msg = f"Operation {op_name} not implemented."
        raise NotImplementedError(msg) from None


def _import_from_path(module_name: str, path: str | Path):
    """Dynamically import a module from a file path, wiring up the package structure for relative imports.

    Args:
        module_name: Name to assign the imported module.
        path: File path of the module to import.

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


def _setup_logger(name: str, verbose: bool) -> logging.Logger:
    """Configure and return a logger for kernel compilation.

    Args:
        name: Logger name, typically __name__ of the calling module.
        verbose: If True, enables DEBUG-level output to stderr.
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers.copy():
        with contextlib.suppress(ValueError):
            logger.removeHandler(handler)
    if verbose:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(ch)
    return logger


def ggml_compile_op(
    op_name: str,
    arch: str,
    input_tensors: list[TensorDesc | None],
    output_tensor: TensorDesc,
    op_params: bytearray,
    exported_name: str,
    config: CompilerConfig,
) -> None:
    """Compile a GGML operation kernel to PDI and instruction files.

    Main entry point for kernel compilation: looks up the dispatch module,
    calls it to obtain a KernelSpec (backend + function), then invokes the
    matching backend compiler.

    Args:
        op_name: Operation name (e.g., "ADD", "MUL_MAT").
        arch: Target architecture (e.g., "aie2", "aie2p").
        input_tensors: Input tensor descriptions.
        output_tensor: Output tensor description.
        op_params: Operation-specific parameters.
        exported_name: Name to export the compiled kernel as.
        config: Compiler configuration.

    Raises:
        NotImplementedError: If the operation or its selected backend is not
            implemented.
        ValueError: If the dispatch function rejects the given inputs (e.g.
            wrong tensor count).
        RuntimeError: If compilation fails with every available backend.
    """
    verbose = config.verbose
    logger = _setup_logger(__name__, verbose)

    # Get kernel mapping for the operation
    kernel = _get_kernel(op_name)

    # Load dispatch module and get dispatch function
    kernel_source_file = Path(__file__).resolve().parent / kernel.source_file
    module = _import_from_path(kernel.name, kernel_source_file)
    dispatch_fn = getattr(module, kernel.name)

    # Create output and work directories
    output_dir = Path(config.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch to get KernelSpec or list[KernelSpec], then normalize and order
    dispatch_result = dispatch_fn(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )
    kernel_specs = _make_kernel_specs(dispatch_result, config.compilers, logger)

    for kernel_spec in kernel_specs:
        logger.info(
            (
                "Compiling op: %s\n"
                "  Op name:              %s\n"
                "  Architecture:         %s\n"
                "  Backend:              %s\n"
                "  Kernel source:        %s\n"
                "  Input tensors:        %s\n"
                "  Output tensor:        %s\n"
                "  Operation parameters: %s\n"
                "  Exported name:        %s\n"
                "  Output directory:     %s"
            ),
            op_name,
            kernel_spec.op_name,
            arch,
            kernel_spec.backend.name,
            str(kernel_source_file),
            kernel_spec.input_tensors,
            kernel_spec.output_tensor,
            kernel_spec.op_params,
            exported_name,
            str(output_dir),
        )

        # Get compiler for the selected backend and compile
        compile_fn = _get_compiler(kernel_spec.backend)

        try:
            compile_fn(
                kernel_spec=kernel_spec,
                exported_name=exported_name,
                output_directory=output_dir,
                logger=logger,
                verbose=verbose,
            )
        except Exception:
            logger.exception(
                "Compilation failed for operation %s, kernel %s with backend %s",
                op_name,
                kernel.name,
                kernel_spec.backend.name,
            )
        else:
            return

    msg = f"Could not compile kernel {kernel.name} for operation {op_name} with any backend."
    logger.error(msg)
    raise RuntimeError(msg)


def _to_tuple_of_ints(string: str) -> tuple[int, int, int, int]:
    """Convert a string of the form "(x,y,z,w)" to a 4-tuple of integers.

    Args:
        string: String of the form "(x,y,z,w)" to convert.

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


def _to_tensordesc(string: str) -> TensorDesc:
    """Create a TensorDesc from a string of the form "(shape)/dtype", e.g. "(1024,1,1,1)/f32".

    Args:
        string: String of the form "(shape)/dtype" to convert.
    """
    shape_str, dtype = string.split("/")
    shape = _to_tuple_of_ints(shape_str)
    return TensorDesc(dtype=dtype, shape=shape, stride=None)


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
        type=_to_tensordesc,
        nargs="+",
        required=True,
        help="Input kernel tensor shapes and datatypes",
    )
    parser.add_argument(
        "--output_tensor",
        type=_to_tensordesc,
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
        config=CompilerConfig(
            output_directory=args.output_directory,
            verbose=args.verbose,
        ),
    )


if __name__ == "__main__":
    main()
