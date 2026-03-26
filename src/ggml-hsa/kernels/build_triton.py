# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Triton-XDNA backend compiler for GGML HSA kernels.
"""

import logging
from pathlib import Path

from kernel import KernelSpec

# Map numpy dtypes to Triton type strings
_numpy_to_triton_dtype_map = {
    "int32": "*i32",
    "float32": "*fp32",
    "float16": "*fp16",
    "bfloat16": "*bf16",
}


def _map_dtype_to_triton(dtype) -> str:
    """
    Map a numpy dtype to a Triton type string.

    Parameters:
        dtype: A numpy dtype object.

    Returns:
        A string representing the corresponding Triton type.
    """
    dtype_str = str(dtype)
    if dtype_str in _numpy_to_triton_dtype_map:
        return _numpy_to_triton_dtype_map[dtype_str]
    else:
        raise ValueError(f"Unsupported dtype for Triton kernel: {dtype_str}")


def compile_triton_kernel(
    kernel_spec: KernelSpec,
    work_dir: Path,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """
    Compile a Triton kernel.

    This function executes the Triton-XDNA compilation pipeline:
    1. Translates the kernel specification into a Triton kernel
    2. Compiles via Triton-XDNA stack to produce PDI and instructions

    Parameters:
        kernel_spec: The KernelSpec containing the Triton kernel function.
        work_dir: Working directory for intermediate files.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for output PDI and instruction files.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.
    """

    # Import Triton compilation modules
    try:
        import triton
        import triton.backends
        from triton.compiler import compile as triton_compile
    except ImportError as e:
        raise ImportError(
            f"Failed to import Triton compilation modules: {e}\n"
            "Ensure triton-xdna is installed correctly."
        ) from e

    # Get kernel from KernelSpec
    kernel_fn, kernel_config = kernel_spec.function(
        arch=kernel_spec.arch,
        input_tensors=kernel_spec.input_tensors,
        output_tensor=kernel_spec.output_tensor,
        op_params=kernel_spec.op_params,
    )

    # Step 3: Extract compilation parameters
    n_elements = kernel_config["n_elements"]
    block_size = kernel_config["block_size"]
    grid = (n_elements // block_size,)

    logger.info(f"Compiling Triton kernel: {kernel_spec.op_name}")
    logger.info(f"  Architecture: {kernel_spec.arch}")
    logger.info(f"  Exported name: {exported_name}")
    logger.info(f"  Total elements: {n_elements}")
    logger.info(f"  Block size: {block_size}")
    logger.info(f"  Grid size: {grid}")

    # Step 4: Configure kernel signature and constexprs
    # The signature maps arg names to their types
    # For vecadd(A, B, C, n_elements, block_size):
    #   A, B, C are pointers
    #   n_elements, block_size are constexpr (compile-time constants)

    input_dtype = kernel_spec.input_tensors[0].dtype
    output_dtype = kernel_spec.output_tensor.dtype

    # Map numpy dtypes to Triton type strings
    ptr_type = _map_dtype_to_triton(input_dtype)

    # Build signature dictionary mapping arg names to types
    # Get arg names from the kernel function
    arg_names = kernel_fn.arg_names
    signature = {}
    for i, name in enumerate(arg_names):
        if name in kernel_config:
            # This is a constexpr parameter
            signature[name] = "constexpr"
        elif i < 3:  # A, B, C pointers
            signature[name] = ptr_type

    logger.info(f"  Kernel arg names: {arg_names}")
    logger.info(f"  Kernel signature: {signature}")
    logger.info(f"  Constexprs: {kernel_config}")

    # Step 5: Invoke Triton AOT compilation
    try:
        # Create ASTSource directly without create_binder()
        # create_binder() requires an active GPU driver which we don't have for AOT
        # Instead, use triton.compiler.ASTSource directly
        src = triton.compiler.ASTSource(
            fn=kernel_fn,
            signature=signature,
            constexprs=kernel_config,
            attrs={},  # No special attributes for now
        )

        # Create target for XDNA backend
        # The backend is registered as "amd_triton_npu" but expects target.backend == "npu"
        registry_name = "amd_triton_npu"
        backend_name = "npu"  # What NPUBackend.supports_target() expects

        if registry_name not in triton.backends.backends:
            raise RuntimeError(
                f"Backend '{registry_name}' not found in registered backends.\n"
                f"Available backends: {list(triton.backends.backends.keys())}"
            )

        # Create target and backend
        target = triton.backends.compiler.GPUTarget(backend_name, kernel_spec.arch, "1")
        backend_info = triton.backends.backends[registry_name]
        backend = backend_info.compiler(target)
        num_warps = 1
        num_stages = 1
        kwargs = {"num_warps": num_warps, "num_stages": num_stages}
        options = backend.parse_options(kwargs)

        logger.info(f"  Target: {backend_name}:{kernel_spec.arch}:1")
        logger.info(f"  Num warps: {num_warps}, Num stages: {num_stages}")

        # Compile the kernel
        compiled = triton_compile(src, target=target, options=options.__dict__)

        logger.info("Triton compilation successful")

    except Exception as e:
        logger.error(f"Triton compilation failed: {e}")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise RuntimeError(
            f"Failed to compile Triton kernel '{exported_name}': {e}"
        ) from e

    # Step 6: Extract binary artifacts from compiled kernel
    # The compiled object has: .asm dict, .metadata, etc.
    logger.info(f"Compiled kernel type: {type(compiled)}")
    logger.info(
        f"Compiled kernel attributes: {[a for a in dir(compiled) if not a.startswith('_')]}"
    )

    # Extract the binary from the asm dictionary
    # backend.binary_ext gives us the key to use (e.g., 'pdi', 'xclbin', etc.)
    binary_ext = backend.binary_ext
    logger.info(f"Backend binary extension: {binary_ext}")

    if hasattr(compiled, "asm") and binary_ext in compiled.asm:
        binary_data = compiled.asm[binary_ext]
        logger.info(f"Extracted binary data: {len(binary_data)} bytes")
    else:
        available_keys = list(compiled.asm.keys()) if hasattr(compiled, "asm") else []
        raise NotImplementedError(
            f"Cannot extract binary from compiled Triton kernel.\n"
            f"Expected key '{binary_ext}' in compiled.asm\n"
            f"Available keys: {available_keys}\n"
            f"Compiled object type: {type(compiled)}"
        )

    # Write PDI file
    pdi_path = output_directory / f"{exported_name}.pdi"
    logger.info(f"Writing PDI to {pdi_path}")
    with open(pdi_path, "wb") as f:
        f.write(binary_data)

    # For XDNA/AIE, the instruction buffer is typically part of the PDI
    # or in a separate metadata field. For now, create a placeholder
    # instruction file that matches the IRON convention.
    insts_path = output_directory / f"{exported_name}_insts.bin"
    logger.info(f"Writing instructions to {insts_path}")

    # Check if there's separate instruction data
    if hasattr(compiled.metadata, "instr") or hasattr(
        compiled.metadata, "instructions"
    ):
        insts_data = getattr(compiled.metadata, "instr", None) or getattr(
            compiled.metadata, "instructions", None
        )
        if insts_data:
            with open(insts_path, "wb") as f:
                f.write(insts_data)
            logger.info(f"  Instructions size: {len(insts_data)} bytes")
        else:
            # Create empty instruction file for compatibility
            with open(insts_path, "wb") as f:
                f.write(b"")
            logger.info("  Instructions: empty (embedded in PDI)")
    else:
        # Create empty instruction file for compatibility
        with open(insts_path, "wb") as f:
            f.write(b"")
        logger.info("  Instructions: empty (embedded in PDI)")

    logger.info(f"Compilation complete: {exported_name}")
    logger.info(f"  PDI size: {len(binary_data)} bytes")
