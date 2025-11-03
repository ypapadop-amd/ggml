# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

from dataclasses import dataclass
import dataclasses
import importlib.util
import os
import sys
import logging
import numpy as np

import aie.iron
import aie.iron.compile
import aie.iron.device

from tensor_desc import tensordesc, TensorDesc


def arch_aligned_num_elements(arch: str, tensor) -> int:
    """
    Returns the number of elements in the tensor aligned to what the architecture expects for the data type of the tensor.

    Parameters:
        arch (str): Target architecture.
        tensor: Tensor.

    Returns:
        int: Number of elements aligned to architecture requirements.
    """

    num_elements = np.size(tensor)
    if arch in ["aie2", "aie2p"]:
        # align to 4 bytes for data types with size < 4
        ALIGNMENT_BYTES = 4
        dtype_size = tensor.dtype.itemsize
        data_size = num_elements * dtype_size
        if data_size % ALIGNMENT_BYTES != 0:
            num_elements = (
                ALIGNMENT_BYTES
                * ((data_size + (ALIGNMENT_BYTES - 1)) // ALIGNMENT_BYTES)
                // dtype_size
            )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return num_elements


def max_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """
    Returns the maximum tile size based on device, data type and number of elements.

    Parameters:
        arch (str): Target architecture.
        dtype (np.dtype): Data type of the tensor elements.
        num_elements (int): Total number of elements in the tensor.

    Returns:
        int: Maximum tile size.
    """
    vector_register_width = 0
    if arch == "aie2" or arch == "aie2p":
        vector_register_width = 512  # bits
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    tile_size = int(vector_register_width / dtype.itemsize)

    while num_elements % tile_size != 0 and tile_size > 1:
        tile_size //= 2

    assert (
        num_elements % tile_size == 0
    ), f"Number of elements ({num_elements}) must be a multiple of tile size ({tile_size})."

    return tile_size


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


def import_from_path(module_name: str, path: str | os.PathLike):
    """
    Imports the module with name module_name from path.

    Parameters:
        module_name (str): Name of the module.
        path (os.PathLike): Path to the module file.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Cannot find module spec for {module_name} at path {path}")
    if spec.loader is None:
        raise ImportError(f"Cannot find loader for module {module_name} at path {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Kernel:
    """Dataclass representing a kernel."""

    name: str
    source_file: str


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
    "MUL_MAT": Kernel("ggml_op_mul_mat", "mat_mul.py"),
}


def compile_kernel(
    ggml_op: str,
    arch: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    exported_name: str,
    output_directory: os.PathLike,
    verbose: bool = False,
):
    """
    Compiles the kernel code to PDI and instruction files.

    Parameters:
        ggml_op (str): GGML operation.
        arch (str): Target architecture.
        input_tensors (list[TensorDesc]): List of input tensor descriptions.
        output_tensor (TensorDesc): Output tensor description.
        exported_name (str): Name to export the compiled kernel as.
        output_directory (str): Directory to save the compiled files.
        verbose (bool): If True, enables verbose logging.
    """

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

    # get kernel source file based on operation
    kernel = op_to_kernel_map.get(ggml_op, None)
    if kernel is None:
        raise ValueError(f"Unsupported GGML operation: {ggml_op}")
    kernel = dataclasses.replace(
        kernel,
        source_file=os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), kernel.source_file)
        ),
    )

    logger.info(
        (
            "Compiling op: %s for arch %s\n"
            "  Kernel name:      %s\n"
            "  Kernel source:    %s\n"
            "  Input tensors:    %s\n"
            "  Output tensor:    %s\n"
            "  Exported name:    %s\n"
            "  Output directory: %s"
        ),
        ggml_op,
        arch,
        kernel.name,
        kernel.source_file,
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
    module = import_from_path(kernel.name, kernel.source_file)
    kernel_fn = getattr(module, kernel.name)

    # remove any existing external functions
    aie.iron.ExternalFunction._instances.clear()

    # generate MLIR module
    mlir_module = kernel_fn(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
    )

    # compile any external functions
    for func in aie.iron.kernel.ExternalFunction._instances:
        aie.iron.compile.compile_cxx_core_function(
            source_path=func._source_file,
            target_arch=arch,
            output_path=func._object_file_name,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=work_dir,
            verbose=verbose,
        )

    # remove generated external functions
    aie.iron.ExternalFunction._instances.clear()

    # write MLIR module to file
    mlir_path = os.path.join(work_dir, f"{exported_name}.mlir")
    logger.info("Writing MLIR module for kernel %s in %s", kernel.name, mlir_path)
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # generate PDI and instructions files
    pdi_path = os.path.join(output_directory, f"{exported_name}.pdi")
    insts_path = os.path.join(output_directory, f"{exported_name}_insts.bin")
    aie.iron.compile.compile_mlir_module(
        mlir_module=mlir_module,
        options=["--alloc-scheme=basic-sequential"],
        insts_path=insts_path,
        pdi_path=pdi_path,
        verbose=verbose,
        work_dir=work_dir,
    )
    logger.info(
        "Finished compilation for kernel %s in %s", kernel.name, output_directory
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

    compile_kernel(
        ggml_op=args.ggml_op,
        arch=args.arch,
        input_tensors=args.input_tensors,
        output_tensor=args.output_tensor,
        exported_name=args.exported_name,
        output_directory=args.output_directory,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
