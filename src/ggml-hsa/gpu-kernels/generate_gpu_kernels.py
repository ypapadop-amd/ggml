#!/usr/bin/env python3
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Generate HIP elementwise kernels as AMDGPU code objects for the GGML HSA GPU backend.

The backend searches for a code object in, in order:

1. ``${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_name>.hsaco`` (pregenerated)
2. ``<cache_dir>/<device_name>/<kernel_name>.hsaco`` (cache)

where ``<kernel_name>`` is the sanitized ggml kernel name, e.g. ``add_8f32_8f32_8f32``. The
loader reads the kernel symbol from the code object's metadata, so it does not depend on the
file name; the kernels generated here export the kernel name itself.

By default the kernels are written to the cache directory the backend uses at run time, so they
are found without setting ``GGML_HSA_KERNEL_DIR``. It is resolved as the backend resolves it:
``GGML_HSA_KERNEL_CACHE_DIR``, else ``$XDG_CACHE_HOME/ggml``, else ``$HOME/.cache/ggml``, else
``/tmp/ggml/ggml-hsa``.

Only elementwise ``add_*`` and ``scale_*`` kernels over contiguous, equally shaped tensors are
supported; other names are skipped with a warning. The element count is a kernel argument, so
the shape in a name only identifies the code object.

Environment:
    ROCM_PATH: ROCm installation (default: /opt/rocm).
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ggml dtype tag -> HIP scalar type. Arithmetic is done in float, as ggml's CPU backend does.
CTYPES = {
    "f32": "float",
    "f16": "_Float16",
    "bf16": "__hip_bfloat16",
}

# (src0, src1, dst) dtype combinations ggml's CPU backend implements for ADD
# (ggml-cpu/binary-ops.cpp); a generated kernel is checked against the CPU result.
ADD_DTYPES = {
    ("f32", "f32", "f32"),
    ("f16", "f16", "f16"),
    ("bf16", "bf16", "bf16"),
    ("bf16", "f32", "bf16"),
    ("bf16", "f32", "f32"),
    ("f16", "f32", "f16"),
    ("f16", "f32", "f32"),
}

# (src, dst) dtype combinations ggml's CPU backend implements for SCALE
# (ggml-cpu/ops.cpp, ggml_compute_forward_scale): f32 only, so a generated kernel can be checked
# against the CPU result.
SCALE_DTYPES = {
    ("f32", "f32"),
}

# A tensor in a kernel name: dimensions joined by "x", the dtype, and "n" if non-contiguous.
_TENSOR = r"(\d+(?:x\d+)*)(f32|f16|bf16)(n?)"
# Kernel names list the destination first, then the sources.
_ADD_RE = re.compile(rf"^add_{_TENSOR}_{_TENSOR}_{_TENSOR}$")
# SCALE has op_params (scale, bias), encoded as hex words; all-zero words are omitted.
_SCALE_RE = re.compile(rf"^scale_{_TENSOR}_{_TENSOR}(?:_[0-9a-f]+)*$")


@dataclass(frozen=True)
class Tensor:
    """A tensor as encoded in a kernel name."""

    shape: str
    dtype: str
    contiguous: bool


def _tensors(match: re.Match) -> list[Tensor]:
    groups = match.groups()
    return [
        Tensor(shape=groups[i], dtype=groups[i + 1], contiguous=groups[i + 2] == "")
        for i in range(0, len(groups), 3)
    ]


def _elementwise(tensors: list[Tensor]) -> bool:
    """Return True if a flat one-work-item-per-element kernel computes the op."""
    return all(t.contiguous and t.shape == tensors[0].shape for t in tensors)


def kernel_source(name: str) -> str | None:
    """Return the HIP source of the kernel for name, or None if it is not supported.

    Args:
        name: Sanitized ggml kernel name.

    Returns:
        HIP source that defines an ``extern "C"`` kernel called name, or None.
    """
    if match := _ADD_RE.match(name):
        dst, src0, src1 = _tensors(match)
        if not _elementwise([dst, src0, src1]):
            return None
        if (src0.dtype, src1.dtype, dst.dtype) not in ADD_DTYPES:
            return None
        return f"""extern "C" __global__ void {name}(const {CTYPES[src0.dtype]} * A,
                                       const {CTYPES[src1.dtype]} * B,
                                       {CTYPES[dst.dtype]} * C,
                                       unsigned long long N) {{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {{
        C[i] = static_cast<{CTYPES[dst.dtype]}>(static_cast<float>(A[i]) +
                                                static_cast<float>(B[i]));
    }}
}}
"""
    if match := _SCALE_RE.match(name):
        dst, src = _tensors(match)
        if not _elementwise([dst, src]) or (src.dtype, dst.dtype) not in SCALE_DTYPES:
            return None
        ctype = CTYPES[dst.dtype]
        # The scalars follow N in op_params order, so dispatch copies them in raw.
        return f"""extern "C" __global__ void {name}(const {ctype} * A, {ctype} * C,
                                       unsigned long long N, float scale, float bias) {{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {{
        C[i] = static_cast<{ctype}>(static_cast<float>(A[i]) * scale + bias);
    }}
}}
"""
    return None


def default_cache_dir() -> Path:
    """Return the runtime kernel cache directory, matching ggml_hsa_cached_kernel_dir()."""
    # The backend checks getenv() != nullptr, so a set-but-empty variable still counts; match that
    # with "is not None" rather than a truthiness check.
    if (value := os.environ.get("GGML_HSA_KERNEL_CACHE_DIR")) is not None:
        return Path(value)
    if (value := os.environ.get("XDG_CACHE_HOME")) is not None:
        return Path(value) / "ggml"
    if (value := os.environ.get("HOME")) is not None:
        return Path(value) / ".cache" / "ggml"
    # Must match the backend's hardcoded fallback, or the kernels would not be found.
    return Path("/tmp/ggml/ggml-hsa")  # noqa: S108


def compile_kernel(
    name: str, source: str, arch: str, rocm_path: Path, out: Path, work_dir: Path
) -> None:
    """Compile source to the code object out.

    Raises:
        subprocess.CalledProcessError: If a compiler invocation fails.
    """
    cu = work_dir / f"{name}.cu"
    asm = work_dir / f"{name}.s"
    cu.write_text(
        f"#include <hip/hip_bf16.h>\n#include <hip/hip_runtime.h>\n\n{source}",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(rocm_path / "bin" / "hipcc"),
            "-S",
            "--cuda-device-only",
            "-Wno-unused-command-line-argument",
            f"--offload-arch={arch}",
            str(cu),
            "-o",
            str(asm),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(rocm_path / "llvm" / "bin" / "clang"),
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={arch}",
            str(asm),
            "-o",
            str(out),
        ],
        check=True,
    )


def main() -> int:
    """Generate the kernels named on the command line's input."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "arch", nargs="?", default="gfx1151", help="AMDGPU target (default: gfx1151)"
    )
    parser.add_argument(
        "names",
        nargs="?",
        default="-",
        help='file with one kernel name per line, or "-" for stdin (default: -)',
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=None,
        help="output kernel directory, the <arch> subdirectory is created inside "
        "(default: the runtime cache directory)",
    )
    args = parser.parse_args()

    rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    arch_dir = (args.out_dir or default_cache_dir()) / args.arch
    arch_dir.mkdir(parents=True, exist_ok=True)
    print(f"generating {args.arch} kernels into {arch_dir}", file=sys.stderr)

    if args.names == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(args.names).read_text(encoding="utf-8").splitlines()

    with tempfile.TemporaryDirectory() as work_dir:
        for line in lines:
            name = line.strip()
            if not name:
                continue
            source = kernel_source(name)
            if source is None:
                print(f"skip (unsupported name): {name}", file=sys.stderr)
                continue
            out = arch_dir / f"{name}.hsaco"
            if out.is_file():
                continue
            compile_kernel(name, source, args.arch, rocm_path, out, Path(work_dir))
            print(f"built {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
