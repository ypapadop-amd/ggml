# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates

import logging
from collections.abc import Iterable
from pathlib import Path

from kernel import Kernel
from tensor_desc import TensorDesc
from iron.utils import suppress_import_pyxrt_msg

suppress_import_pyxrt_msg()

from aie.iron import ExternalFunction
from aie.utils.compile import compile_cxx_core_function
from aie.utils.compile import compile_mlir_module


def _compile_aie_core_kernels(
    arch: str,
    functions: Iterable,
    work_dir: Path,
    verbose: bool,
) -> None:
    """
    Compiles AIE core functions to object files.

    Parameters:
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        functions: aie.iron.ExternalFunction objects
        work_dir (Path): Working directory for intermediate files.
        verbose (bool): If True, enables verbose compilation output.
    """
    for func in functions:
        compile_cxx_core_function(
            source_path=func._source_file,
            target_arch=arch,
            output_path=func.bin_name,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=str(work_dir),
            verbose=verbose,
        )


def compile_iron_kernel(
    kernel: Kernel,
    arch: str,
    input_tensors: list[TensorDesc],
    output_tensor: TensorDesc,
    op_params: bytearray,
    work_dir: Path,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """
    Compiles an IRON kernel to PDI and instruction files.

    This function executes the kernel's Python function to generate an MLIR module,
    compiles any external C++ core functions, and then compiles the MLIR module
    to produce the final PDI and instruction binary files.

    Parameters:
        kernel (Kernel): The kernel to compile.
        arch (str): Target architecture (e.g., "aie2", "aie2p").
        input_tensors (list[TensorDesc]): List of input tensor descriptions.
        output_tensor (TensorDesc): Output tensor description.
        op_params (bytearray): Operation-specific parameters.
        work_dir (Path): Working directory for intermediate files.
        exported_name (str): Name for the exported kernel files.
        output_directory (Path): Directory for output PDI and instruction files.
        logger (logging.Logger): Logger for status messages.
        verbose (bool): If True, enables verbose compilation output.
    """
    # remove any existing external functions
    ExternalFunction._instances.clear()

    # generate MLIR module (populates ExternalFunction._instances)
    mlir_module = kernel.function(
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
    )

    # compile any external functions
    _compile_aie_core_kernels(
        arch=arch,
        functions=ExternalFunction._instances,
        work_dir=work_dir,
        verbose=verbose,
    )

    # remove generated external functions
    ExternalFunction._instances.clear()

    # write MLIR module to file
    mlir_path = work_dir / f"{exported_name}.mlir"
    logger.info("Writing MLIR module for kernel %s in %s", kernel.name, mlir_path)
    with open(mlir_path, "wt", encoding="utf-8") as file:
        file.write(str(mlir_module))

    # generate PDI and instructions files
    pdi_path = output_directory / f"{exported_name}.pdi"
    insts_path = output_directory / f"{exported_name}_insts.bin"
    compile_mlir_module(
        mlir_module=mlir_module,
        options=["--alloc-scheme=basic-sequential"],
        insts_path=str(insts_path),
        pdi_path=str(pdi_path),
        verbose=verbose,
        work_dir=str(work_dir),
    )
