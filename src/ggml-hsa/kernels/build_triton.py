# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Triton-XDNA backend compiler for GGML HSA kernels."""

import logging
import os
import shutil
import subprocess
from contextlib import ContextDecorator
from pathlib import Path

from kernel import KernelSpec


class TempEnvSet(ContextDecorator):
    """Context manager to temporarily set an environment variable.

    This ensures that Triton uses a specific cache directory for compiled artifacts,
    which helps with organization and cleanup.

    Parameters:
        env_var: Name of the environment variable to set.
        value: Value to set for the environment variable.

    Usage:
        with TempEnvSet("TRITON_CACHE_DIR", str(Path("/path/to/cache"))):
            # Triton compilation code here
    """

    env_var: str
    value: str | None
    old_value: str | None = None

    def __init__(self, env_var: str, value: str | None) -> None:
        """Initialize the context manager with the desired environment variable and value.

        Parameters:
            env_var: Name of the environment variable to set.
            value: Value to set for the environment variable. If None, the variable will not be set.
        """
        self.env_var = env_var
        self.value = value
        self.old_value = None

    def __enter__(self) -> None:
        """Set the environment variable to the specified value."""
        if self.value is None:
            return
        self.old_value = os.environ.get(self.env_var, None)
        os.environ[self.env_var] = str(self.value)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore the original environment variable after exiting the context."""
        if self.value is None:
            return
        if self.old_value is not None:
            os.environ[self.env_var] = self.old_value
        else:
            del os.environ[self.env_var]


def _get_triton_target(kernel_spec: KernelSpec) -> str:
    """Returns Triton target names for a given KernelSpec architecture.

    Parameters:
        kernel_spec: The KernelSpec containing the architecture information.

    Returns:
        A string representing the Triton target name corresponding to the kernel's architecture.

    Raises:
        ValueError: If the architecture specified in the kernel_spec is not supported.
    """
    mapping = {
        "aie2": "npu1",
        "aie2p": "npu2",
    }
    try:
        return mapping[kernel_spec.arch]
    except KeyError:
        msg = f"Unsupported architecture for Triton kernel: {kernel_spec.arch}"
        raise ValueError(msg) from None


def compile_triton_kernel(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Compile a Triton kernel.

    This function executes the Triton-XDNA compilation pipeline:
    1. Translates the kernel specification into a Triton kernel
    2. Compiles via Triton-XDNA stack to produce PDI and instructions

    Parameters:
        kernel_spec: The KernelSpec containing the Triton kernel function.
        exported_name: Name for the exported kernel files.
        output_directory: Directory for output PDI and instruction files.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.

    """
    from dataclasses import MISSING

    import triton
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver import NPUDriver, get_npu_cache_dir

    # Determine Triton cache directory
    cache_dir = output_directory / f"{exported_name}-triton-artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Triton cache directory: %s", cache_dir)

    # Set active driver based on architecture
    arch = _get_triton_target(kernel_spec)
    if arch in ["npu1", "npu2"]:
        triton.runtime.driver.set_active(NPUDriver())
    else:
        msg = f"Unsupported architecture for Triton kernel: {arch}"
        raise ValueError(msg)

    with (
        TempEnvSet("TRITON_CACHE_DIR", str(cache_dir)),
        config_context(
            compile_only=True,
            transform_tiling_script=kernel_spec.config.get("transform_script", MISSING),
            output_format="xclbin",
            debug=1 if verbose else 0,
            target=arch,
        ),
    ):
        compiled_kernel = kernel_spec.function()
        xclbin_path = Path(get_npu_cache_dir(compiled_kernel))
        logger.info(
            (
                "Triton compilation successful\n"
                "  Metadata:           %s\n"
                "  Metadata Group:     %s\n"
                "  XCLBIN Parent Path: %s"
            ),
            compiled_kernel.metadata,
            compiled_kernel.metadata_group,
            str(xclbin_path),
        )
        with Path(xclbin_path / "tt.shared.mlir").open("w", encoding="utf-8") as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
            logger.info("Triton Shared MLIR written to %s", f.name)

        # Create PDI from Triton cache xclbin
        pdi_path = output_directory / f"{exported_name}.pdi"
        cmd = [
            "/opt/xilinx/xrt/bin/xclbinutil",
            "--dump-section",
            "AIE_PARTITION:JSON:partition.json",
            "--force",
            "--input",
            str(xclbin_path / "aie.xclbin"),
        ]
        subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=str(xclbin_path),
        )
        pdi_src_path = next(xclbin_path.glob("**/*.pdi"))
        shutil.copy(pdi_src_path, pdi_path)

        # Copy instructions file from Triton cache
        insts_path = output_directory / f"{exported_name}_insts.bin"
        shutil.copy(xclbin_path / "insts.bin", insts_path)

        logger.info(
            (
                "Triton compilation successful\n"
                "  PDI Path:          %s\n"
                "  Instructions Path: %s"
            ),
            pdi_path,
            insts_path,
        )
