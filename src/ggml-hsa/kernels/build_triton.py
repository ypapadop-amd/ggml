# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Triton-XDNA backend compiler for GGML HSA kernels."""

import logging
import os
from contextlib import ContextDecorator
from dataclasses import MISSING
from pathlib import Path

import triton
from kernel import KernelSpec
from triton_kernels.spec_utils import NPU_ARCH_MAP, ArchKind, arch_kind


class TempEnvSet(ContextDecorator):
    """Context manager to temporarily set an environment variable.

    Usage:
        with TempEnvSet("TRITON_CACHE_DIR", str(Path("/path/to/cache"))):
            # Triton compilation code here

    Attributes:
        env_var: Name of the environment variable to set.
        value: Value to set; if None, the variable is left untouched.
        old_value: Original value of the variable, restored on exit.
    """

    env_var: str
    value: str | None
    old_value: str | None = None

    def __init__(self, env_var: str, value: str | None) -> None:
        """Initialize the context manager.

        Args:
            env_var: Name of the environment variable to set.
            value: Value to set; if None, the variable is left untouched.
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
        """Restore the original environment variable after exiting the context.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_val: Exception instance raised in the context, if any.
            exc_tb: Traceback of the exception raised, if any.
        """
        if self.value is None:
            return
        if self.old_value is not None:
            os.environ[self.env_var] = self.old_value
        else:
            del os.environ[self.env_var]


def _get_triton_target(kernel_spec: KernelSpec) -> str:
    """Return the Triton target string for a KernelSpec's architecture.

    Maps NPU architecture names to their Triton equivalents and passes GPU
    architectures through unchanged (e.g. "npu1", "npu2", "gfx942").

    Args:
        kernel_spec: The KernelSpec whose architecture to map.

    Raises:
        ValueError: If the architecture is not a known NPU or GPU target.
    """
    if kernel_spec.arch in NPU_ARCH_MAP:
        return NPU_ARCH_MAP[kernel_spec.arch]
    if arch_kind(kernel_spec.arch) is ArchKind.GPU:
        return kernel_spec.arch
    msg = f"Unsupported architecture for Triton kernel: {kernel_spec.arch}"
    raise ValueError(msg)


def _compile_triton_gpu_kernel(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Compile a Triton kernel to a GPU code object (.hsaco) ahead of time.

    Uses the upstream Triton AMD backend to ahead-of-time compile the same @triton.jit kernel
    used by the AIE path. Emits ``<exported_name>.hsaco`` plus a launch-metadata sidecar
    ``<exported_name>.hsaco.json`` = {"grid_size_x", "workgroup_size_x", "group_segment_size"}
    because the launch geometry (workgroup size, number of workgroups) is fixed at compile time
    and the dynamic LDS size is set at launch, and neither is recoverable from the code object
    alone; the C++ dispatch reads them back from the sidecar.

    Parameters:
        kernel_spec: The KernelSpec; config carries triton_fn, signature, constexprs,
            num_warps, num_programs and, optionally, num_stages (Triton's default if absent).
        exported_name: Base name for the output files.
        output_directory: Directory for the .hsaco and sidecar.
        logger: Logger for status messages.
        verbose: If True, enables verbose output.
    """
    import json

    from triton.backends.compiler import GPUTarget

    cfg = kernel_spec.config
    fn = cfg["triton_fn"]
    signature = cfg["signature"]
    constexprs = cfg["constexprs"]
    num_warps = int(cfg.get("num_warps", 4))
    num_programs = int(cfg["num_programs"])
    if num_programs < 1:
        # e.g. an empty tensor: the loader rejects a zero-size grid, so fail before compiling
        msg = (
            f"{exported_name}: a launch needs at least one program, got {num_programs}"
        )
        raise ValueError(msg)

    # gfx11 uses wave32; the warp_size passed here is informational (Triton selects the ISA
    # wavefront). The actual workgroup size is read back from the compiled kernel metadata.
    target = GPUTarget("hip", kernel_spec.arch, 64)
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    options = {"num_warps": num_warps}
    if "num_stages" in cfg:
        options["num_stages"] = int(cfg["num_stages"])
    compiled = triton.compile(src, target=target, options=options)

    hsaco = compiled.asm["hsaco"]
    hsaco_path = output_directory / f"{exported_name}.hsaco"
    hsaco_path.write_bytes(bytes(hsaco))

    block_x = int(compiled.metadata.num_warps) * int(compiled.metadata.warp_size)
    # The AQL dispatch packet holds the grid in 32 bits and the workgroup in 16; the loader
    # rejects a sidecar outside them, so fail here with the reason instead.
    if num_programs * block_x > 0xFFFFFFFF or block_x > 0xFFFF:
        msg = (
            f"{exported_name}: launch of {num_programs} programs x {block_x} work-items does "
            f"not fit an AQL packet (grid at most 2^32 - 1, workgroup at most 2^16 - 1)"
        )
        raise ValueError(msg)
    launch = {
        "grid_size_x": int(num_programs * block_x),  # total work-items (AQL grid_size)
        "workgroup_size_x": int(block_x),  # work-items per workgroup
        # Triton requests its shared memory (e.g. for tl.dot) as dynamic LDS, which the code
        # object's fixed group segment size does not include, so the launcher must reserve it.
        "group_segment_size": int(compiled.metadata.shared),
    }
    sidecar_path = output_directory / f"{exported_name}.hsaco.json"
    sidecar_path.write_text(json.dumps(launch))

    logger.info(
        "Triton GPU compilation successful\n  hsaco: %s\n  launch: %s",
        hsaco_path,
        launch,
    )


def _compile_npu_kernel(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    cache_dir: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Compile a Triton kernel for an NPU through the Triton-XDNA pipeline.

    Runs the kernel's compile function under the NPU driver and packs the result into
    ``<exported_name>.hsaco``: a full ELF on aie2p, an xclbin and instructions binary on aie2.

    Args:
        kernel_spec: The KernelSpec; its function triggers the Triton compilation and its config
            may carry a transform_script.
        exported_name: Base name for the output files.
        output_directory: Directory for the hsaco.
        cache_dir: Triton cache directory for the compilation artifacts.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.
    """
    from aie_hsaco import pack_aie_hsaco
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.driver import NPUDriver, get_npu_cache_dir

    # As for IRON, aie2p kernels are built as full ELFs, which take none of the queue's 32
    # PDI slots, and ROCR rejects a dispatch batch that mixes them with PDI-plus-instructions
    # kernels. aie2 has no full-ELF format and keeps the xclbin.
    full_elf = kernel_spec.arch == "aie2p"

    triton.runtime.driver.set_active(NPUDriver())
    with (
        TempEnvSet("TRITON_CACHE_DIR", str(cache_dir)),
        config_context(
            compile_only=True,
            transform_tiling_script=kernel_spec.config.get("transform_script", MISSING),
            output_format="elf" if full_elf else "xclbin",
            debug=1 if verbose else 0,
            target=_get_triton_target(kernel_spec),
        ),
    ):
        compiled_kernel = kernel_spec.function()
        npu_artifacts_path = Path(get_npu_cache_dir(compiled_kernel))
        logger.info(
            (
                "Triton compilation successful\n"
                "  Metadata:           %s\n"
                "  Metadata Group:     %s\n"
                "  NPU Artifacts Path: %s"
            ),
            compiled_kernel.metadata,
            compiled_kernel.metadata_group,
            str(npu_artifacts_path),
        )
        with Path(npu_artifacts_path / "tt.shared.mlir").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(str(compiled_kernel.asm["ttsharedir"]))
            logger.info("Triton Shared MLIR written to %s", f.name)

        if full_elf:
            # MLIR-AIR's full ELF also holds an internal sequence that loads no PDI,
            # which ROCR refuses to load; pack only the kernel Triton-XDNA dispatches.
            elf_kernel_name_path = npu_artifacts_path / "elf_kernel_name.txt"
            try:
                elf_kernel_name = elf_kernel_name_path.read_text(
                    encoding="utf-8"
                ).strip()
            except OSError as e:
                msg = (
                    f"{exported_name}: could not read {elf_kernel_name_path}: {e}; the "
                    f"Triton-XDNA pipeline did not produce it in {npu_artifacts_path}"
                )
                raise RuntimeError(msg) from e
            hsaco_path = pack_aie_hsaco(
                kernel_spec,
                exported_name,
                output_directory,
                full_elf_path=npu_artifacts_path / "aie.elf",
                elf_kernel_name=elf_kernel_name,
            )
        else:
            hsaco_path = pack_aie_hsaco(
                kernel_spec,
                exported_name,
                output_directory,
                npu_artifacts_path / "aie.xclbin",
                npu_artifacts_path / "insts.bin",
            )

        logger.info("Triton-XDNA compilation successful\n  HSACO Path: %s", hsaco_path)


def compile_triton_kernel(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Compile a Triton kernel for the target architecture in kernel_spec.

    NPU targets run the Triton-XDNA pipeline and pack the result into an hsaco: a
    full ELF on aie2p, an xclbin and instructions binary on aie2. GPU targets
    ahead-of-time compile the kernel to a .hsaco and emit a launch-metadata sidecar.

    Args:
        kernel_spec: The KernelSpec containing the Triton kernel function.
        exported_name: Name for the exported kernel files.
        output_directory: Directory where output files are written.
        logger: Logger for status messages.
        verbose: If True, enables verbose compilation output.

    Raises:
        ValueError: If the architecture is not a supported NPU or GPU target.
    """
    # Determine Triton cache directory
    cache_dir = output_directory / f"{exported_name}-triton-artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Triton cache directory: %s", cache_dir)

    kind = arch_kind(kernel_spec.arch)
    if kind is ArchKind.NPU:
        _compile_npu_kernel(
            kernel_spec, exported_name, output_directory, cache_dir, logger, verbose
        )
    elif kind is ArchKind.GPU:
        # GPU targets ahead-of-time compile to a .hsaco and emit a launch-metadata sidecar
        # (grid/block geometry) required by the C++ dispatch.
        # Keep Triton's own cache under the kernel's artifact directory, as the NPU path does,
        # rather than in ~/.triton/cache: it then follows the kernel cache (and its clearing).
        with TempEnvSet("TRITON_CACHE_DIR", str(cache_dir)):
            _compile_triton_gpu_kernel(
                kernel_spec, exported_name, output_directory, logger, verbose
            )
    else:
        msg = f"Unsupported architecture for Triton kernel: {kernel_spec.arch}"
        logger.error(msg)
        raise ValueError(msg)
