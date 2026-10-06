# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Packs a compiled AIE kernel into an hsaco (HSA code object) for the ROCr loader."""

import contextlib
import io
import tempfile
from pathlib import Path

from aie.compiler.hsaco import pack
from kernel import KernelSpec

# Each kernel argument occupies two uint64 kernarg entries: its device address and its size.
_KERNARG_BYTES_PER_ARG = 16


def pack_aie_hsaco(
    kernel_spec: KernelSpec,
    exported_name: str,
    output_directory: Path,
    xclbin_path: Path | None = None,
    insts_path: Path | None = None,
    full_elf_path: Path | None = None,
    elf_kernel_name: str | None = None,
) -> Path:
    """Write ``<exported_name>.hsaco`` from an xclbin and its instruction sequence, or a full ELF.

    Runs MLIR-AIE's ``aie-hsaco`` packer. Give it either ``xclbin_path`` and
    ``insts_path``, or ``full_elf_path`` (aie2p only):

    - From an xclbin, the packer takes the PDI and the partition's column count from
      the xclbin's ``AIE_PARTITION`` section, and the kernel is named ``exported_name``.
    - From a full ELF, the PDIs, control code and column count all come from the ELF,
      and so does the kernel name (``<device>:<sequence>``, e.g. ``main:sequence``).
      Unlike a PDI-plus-instructions kernel, it takes no PDI slot on the queue. An
      ELF with more than one kernel (MLIR-AIR's carries an internal sequence next to
      the one to dispatch) needs ``elf_kernel_name`` to say which to pack.

    The hsaco has one kernel, in the section for ``kernel_spec.arch``; its arguments
    are the kernel's input tensors followed by its output tensor. The hsaco is
    assembled in a scratch directory and renamed into place only once complete, so a
    failed pack never leaves a partial hsaco that the runtime would find and fail to
    load.

    Args:
        kernel_spec: The compiled kernel's KernelSpec.
        exported_name: Name of the hsaco file, and of the kernel when packing an xclbin.
        output_directory: Directory for the output hsaco.
        xclbin_path: xclbin holding the kernel's PDI.
        insts_path: Instruction sequence (``insts.bin``).
        full_elf_path: Full ELF (``aiecc --get-full-elf``).
        elf_kernel_name: The full ELF's kernel to pack; every kernel in it if None.

    Returns:
        The path of the written hsaco.

    Raises:
        ValueError: If not given exactly one of the two input forms.
        RuntimeError: If the packer rejects the inputs.
    """
    if (full_elf_path is None) == (xclbin_path is None or insts_path is None):
        msg = "pack_aie_hsaco needs either xclbin_path and insts_path, or full_elf_path"
        raise ValueError(msg)
    num_kernargs = len(kernel_spec.input_tensors) + 1  # inputs, then the output
    kernarg_size = _KERNARG_BYTES_PER_ARG * num_kernargs
    hsaco_path = output_directory / f"{exported_name}.hsaco"
    with tempfile.TemporaryDirectory(dir=output_directory) as scratch_dir:
        scratch = Path(scratch_dir) / hsaco_path.name
        if full_elf_path is not None:
            _pack_full_elf(
                scratch,
                kernel_spec.arch,
                full_elf_path,
                kernarg_size,
                elf_kernel_name,
                exported_name,
            )
            scratch.replace(hsaco_path)
            return hsaco_path
        argv = [
            "--hsaco",
            str(scratch),
            "--arch",
            kernel_spec.arch,
            "--kernel-name",
            exported_name,
            "--kernel-xclbin",
            str(xclbin_path),
            "--kernel-insts",
            str(insts_path),
            "--kernel-kernarg",
            str(kernarg_size),
        ]
        # The packer is a command-line tool: it reports bad input on stderr and exits.
        # Keep that report so the exception says why the pack failed.
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                pack.main(argv)
        except SystemExit as e:
            # The report follows argparse's usage text and may span several lines, e.g. when
            # it quotes xclbinutil's or llvm-objcopy's own output.
            report = stderr.getvalue()
            start = report.find("aie-hsaco: error:")
            reason = (report[start:] if start >= 0 else report).strip()
            msg = f"aie-hsaco could not pack kernel {exported_name}: " + (
                reason or f"exit status {e.code}"
            )
            raise RuntimeError(msg) from e
        scratch.replace(hsaco_path)
    return hsaco_path


def _pack_full_elf(
    hsaco_path: Path,
    arch: str,
    full_elf_path: Path,
    kernarg_size: int,
    elf_kernel_name: str | None,
    exported_name: str,
) -> None:
    """Write a full ELF's kernels, or just ``elf_kernel_name``, into a new hsaco.

    Goes through the packer's library functions, the same steps its command line
    takes, because the command line of the pinned MLIR-AIE cannot select one kernel
    of a full ELF. Errors are raised as RuntimeError, like a failed command line.
    """
    error = f"aie-hsaco could not pack kernel {exported_name}"
    try:
        kernels = pack.kernels_from_full_elf(str(full_elf_path), kernarg_size)
    except (OSError, ValueError) as e:
        msg = f"{error}: {e}"
        raise RuntimeError(msg) from e
    if elf_kernel_name is not None:
        kernels = [k for k in kernels if k["name"] == elf_kernel_name]
        if not kernels:
            msg = f"{error}: {full_elf_path} has no kernel named {elf_kernel_name!r}"
            raise RuntimeError(msg)
    try:
        section = pack.build_section(arch, kernels)
        pack.ensure_hsaco(str(hsaco_path))
        pack.inject(str(hsaco_path), arch, section)
    except (OSError, ValueError, RuntimeError) as e:
        msg = f"{error}: {e}"
        raise RuntimeError(msg) from e
