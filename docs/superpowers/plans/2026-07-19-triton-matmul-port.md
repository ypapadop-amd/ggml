# Triton matmul port (PoC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a Triton-compiled bf16 256×256×256 matmul as a `GGML_OP_MUL_MAT` kernel on the NPU, selected ahead of the IRON path only for that exact profile.

**Architecture:** Reuse the existing Triton-XDNA compilation path (`build_triton.py`, already used by `GGML_OP_ADD`). Port the `bare_matmul` Triton kernel and its two per-arch transform scripts verbatim from `Triton-XDNA/examples/matmul_bf16_m64_n64_k64/`. Widen `ggml_op_mul_mat` to return `[Triton, IRON]` when a node matches the profile, else `[IRON]`. No C++ backend/runtime changes — the `.pdi`/`_insts.bin` dispatch contract is backend-agnostic. Extend `test-mul-mat-hsa.cpp` to drive and verify the bf16 case on-device.

**Tech Stack:** Python (Triton, `amd_triton_npu` backend), MLIR transform dialect, C++ (ggml test harness), CMake.

## Global Constraints

- Target problem size (the only shape the Triton path accepts): **M = N = K = 256**, verbatim from the example's tuning. The `m64_n64_k64` in the example name is the **L1 tile size**, not the problem size.
- Input dtype **bf16** (both operands); output dtype **f32**.
- IRON `gemm.py` remains the general MUL_MAT path and must be unchanged. The Triton spec is *added*, never replaces IRON in the returned list.
- Follow the existing `vecadd` Triton kernel layout and naming exactly: kernel in `triton_kernels/matmul.py`, transforms in `triton_kernels/matmul_<arch>.mlir`, referenced via `config["transform_script"]`.
- Arch strings: `"aie2"` (npu1) and `"aie2p"` (npu2).
- bf16 numeric tolerance for the on-device check: `atol = 1e1`, `rtol = 1e-1` (the source example's bounds).
- A device is present at `/dev/accel0`; the on-device test is expected to actually run.
- Reference spec: `docs/superpowers/specs/2026-07-19-triton-matmul-port-design.md`.

---

### Task 1: Port the Triton matmul kernel and transform scripts

Adds the three new source files (kernel + two transform scripts) and packages them via CMake. Deliverable: the kernel module imports and exposes a Triton JIT function; the transform scripts are byte-identical copies of the example; CMake installs all three.

**Files:**
- Create: `src/ggml-hsa/kernels/triton_kernels/matmul.py`
- Create: `src/ggml-hsa/kernels/triton_kernels/matmul_aie2.mlir`
- Create: `src/ggml-hsa/kernels/triton_kernels/matmul_aie2p.mlir`
- Modify: `src/ggml-hsa/kernels/triton_kernels/CMakeLists.txt`
- Test: `tests/ggml-hsa/test_triton_matmul_kernel.py` (new, pytest)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `triton_kernels.matmul.bare_matmul` — a `@triton.jit` kernel with signature `bare_matmul(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)` (all scalars after `C` are `tl.constexpr`). Files `matmul_aie2.mlir` / `matmul_aie2p.mlir` next to it, consumed later via `config["transform_script"]`.

- [ ] **Step 1: Copy the transform scripts verbatim**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
cp ../Triton-XDNA/examples/matmul_bf16_m64_n64_k64/transform_aie2.mlir \
   src/ggml-hsa/kernels/triton_kernels/matmul_aie2.mlir
cp ../Triton-XDNA/examples/matmul_bf16_m64_n64_k64/transform_aie2p.mlir \
   src/ggml-hsa/kernels/triton_kernels/matmul_aie2p.mlir
```

- [ ] **Step 2: Verify the copies are byte-identical**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
diff ../Triton-XDNA/examples/matmul_bf16_m64_n64_k64/transform_aie2.mlir \
     src/ggml-hsa/kernels/triton_kernels/matmul_aie2.mlir && \
diff ../Triton-XDNA/examples/matmul_bf16_m64_n64_k64/transform_aie2p.mlir \
     src/ggml-hsa/kernels/triton_kernels/matmul_aie2p.mlir && echo "IDENTICAL"
```
Expected: prints `IDENTICAL`, no diff output.

- [ ] **Step 3: Create the Triton kernel module**

Create `src/ggml-hsa/kernels/triton_kernels/matmul.py` with exactly:
```python
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton kernel for single-block matrix multiplication: C = A @ B.

Ported verbatim from Triton-XDNA examples/matmul_bf16_m64_n64_k64. All shape
and stride parameters are compile-time constants; the NPU tiling/packing lives
in the paired matmul_<arch>.mlir transform scripts, not in this kernel.
"""

import triton
import triton.language as tl


@triton.jit
def bare_matmul(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Compute C = A @ B for a single (BLOCK_SIZE_M, BLOCK_SIZE_N) output block."""
    pid_m = tl.program_id(0)  # block row id
    pid_n = tl.program_id(1)  # block column id

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_block = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_block = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    c_block = tl.dot(a_block, b_block)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_block)
```

- [ ] **Step 4: Add the three files to CMake packaging**

In `src/ggml-hsa/kernels/triton_kernels/CMakeLists.txt`, the `TRITON_FILES` list currently reads:
```cmake
set(TRITON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd.py
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    )
```
Replace it with (adds matmul files; also adds the pre-existing `vecadd_aie2p.mlir` which is currently omitted, keeping both arch scripts symmetric):
```cmake
set(TRITON_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_aie2p.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd.py
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_aie2.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_aie2p.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.py
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
    )
```

- [ ] **Step 5: Write the failing test**

Create `tests/ggml-hsa/test_triton_matmul_kernel.py`:
```python
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Smoke tests for the ported Triton matmul kernel source files."""

from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
TRITON_DIR = KERNELS_DIR / "triton_kernels"


def test_transform_scripts_present():
    assert (TRITON_DIR / "matmul_aie2.mlir").is_file()
    assert (TRITON_DIR / "matmul_aie2p.mlir").is_file()


def test_bare_matmul_is_jit():
    import sys

    sys.path.insert(0, str(KERNELS_DIR))
    triton = pytest.importorskip("triton")
    from triton_kernels.matmul import bare_matmul

    assert isinstance(bare_matmul, triton.runtime.jit.JITFunction)
```

- [ ] **Step 6: Run the test to verify it fails (before the module exists) or passes**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_triton_matmul_kernel.py -v
```
Expected: `test_transform_scripts_present` PASSES (files copied in Step 1). `test_bare_matmul_is_jit` PASSES if `triton` is importable, else is SKIPPED (`importorskip`). Neither should ERROR. If `test_bare_matmul_is_jit` fails with `ModuleNotFoundError: triton_kernels.matmul`, the module in Step 3 was not created correctly — fix it.

- [ ] **Step 7: Commit**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/kernels/triton_kernels/matmul.py \
        src/ggml-hsa/kernels/triton_kernels/matmul_aie2.mlir \
        src/ggml-hsa/kernels/triton_kernels/matmul_aie2p.mlir \
        src/ggml-hsa/kernels/triton_kernels/CMakeLists.txt \
        tests/ggml-hsa/test_triton_matmul_kernel.py
git commit -m "Add ported Triton matmul kernel and transform scripts"
```

---

### Task 2: Wire the Triton matmul KernelSpec and profile guard into dispatch

Widen `ggml_op_mul_mat` to return a list: `[Triton, IRON]` when the node matches the 256³ bf16→f32 profile, else `[IRON]` only. The Triton spec's compile function is deferred and re-checks the profile (defensive, mirroring `_make_triton_add_kernel_spec`). Deliverable: pure-logic dispatch selection, verified without any device or compilation.

**Files:**
- Modify: `src/ggml-hsa/kernels/mul_mat.py` (full rewrite; currently 43 lines)
- Test: `tests/ggml-hsa/test_mul_mat_dispatch.py` (new, pytest)

**Interfaces:**
- Consumes: `triton_kernels.matmul.bare_matmul` (Task 1); `triton_kernels.utils.{numpy_dtype_to_torch, triton_device}`; `kernel.{Backend, KernelSpec}`; `iron_kernels.gemm.gemm`.
- Produces: `ggml_op_mul_mat(arch, input_tensors, output_tensor, op_params) -> list[KernelSpec]`. First element is the Triton spec iff `_matches_triton_matmul_profile(input_tensors, output_tensor)` is True. Helper `_matches_triton_matmul_profile(input_tensors, output_tensor) -> bool` is module-level and importable by the test.

- [ ] **Step 1: Write the failing test**

Create `tests/ggml-hsa/test_mul_mat_dispatch.py`:
```python
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""Dispatch-selection tests for GGML_OP_MUL_MAT (Triton vs IRON)."""

import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

from kernel import Backend  # noqa: E402
from mul_mat import ggml_op_mul_mat  # noqa: E402
from tensor_desc import TensorDesc  # noqa: E402


def _td(dtype, dim=256):
    return TensorDesc(dtype=dtype, shape=(dim, dim, 1, 1))


def test_triton_first_for_256_bf16():
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16"), _td("bf16")], _td("f32"), bytearray()
    )
    assert isinstance(specs, list)
    assert [s.backend for s in specs] == [Backend.TRITON, Backend.IRON]
    assert specs[0].config["transform_script"].endswith("matmul_aie2.mlir")


def test_iron_only_for_wrong_dtype():
    specs = ggml_op_mul_mat(
        "aie2", [_td("f32"), _td("f32")], _td("f32"), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]


def test_iron_only_for_wrong_shape():
    specs = ggml_op_mul_mat(
        "aie2", [_td("bf16", 128), _td("bf16", 128)], _td("f32", 128), bytearray()
    )
    assert [s.backend for s in specs] == [Backend.IRON]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_mul_mat_dispatch.py -v
```
Expected: FAIL — `test_triton_first_for_256_bf16` fails because `ggml_op_mul_mat` currently returns a single `KernelSpec`, not a list.

- [ ] **Step 3: Rewrite mul_mat.py**

Replace the entire contents of `src/ggml-hsa/kernels/mul_mat.py` with:
```python
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

"""Top-level entry point for the matrix multiplication operation (GGML_OP_MUL_MAT)."""

from functools import partial
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from .kernel import Backend, KernelSpec

# The ported Triton matmul is specialised to a single square problem size,
# matching the verbatim transform scripts (see the design doc). Only nodes of
# exactly this shape/dtype are routed to Triton; everything else uses IRON.
_TRITON_MATMUL_DIM = 256


def _matches_triton_matmul_profile(input_tensors: list, output_tensor) -> bool:
    """Return True if the node is the exact profile the Triton matmul supports.

    The profile is: two bf16 inputs, one f32 output, all operands square with
    leading two dims equal to _TRITON_MATMUL_DIM, higher dims trivial (== 1),
    and all operands contiguous.
    """
    if len(input_tensors) != 2:
        return False
    tensors = [*input_tensors, output_tensor]
    if not all(getattr(t, "contiguous", True) for t in tensors):
        return False
    if any(np.dtype(t.dtype) != np.dtype(bfloat16) for t in input_tensors):
        return False
    if np.dtype(output_tensor.dtype) != np.dtype(np.float32):
        return False
    d = _TRITON_MATMUL_DIM
    for t in tensors:
        shape = tuple(t.shape)
        if shape[0] != d or shape[1] != d:
            return False
        if any(s != 1 for s in shape[2:]):
            return False
    return True


def _make_iron_matmul_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create the IRON-backend KernelSpec for MUL_MAT (the general path)."""
    from .iron_kernels.gemm import gemm

    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=partial(
            gemm, arch=arch, input_tensors=input_tensors, output_tensor=output_tensor
        ),
    )


def _make_triton_matmul_kernel_spec(
    arch: str, input_tensors: list, output_tensor
) -> KernelSpec:
    """Create the TRITON-backend KernelSpec for the fixed 256x256x256 bf16 matmul."""
    dim = _TRITON_MATMUL_DIM

    def _compile(arch=arch, input_tensors=input_tensors, output_tensor=output_tensor):
        # Imports and tensor creation are deferred so any failure is caught by
        # the try/except fallback in build.py, mirroring the ADD Triton spec.
        import torch
        import triton

        from .triton_kernels.matmul import bare_matmul
        from .triton_kernels.utils import numpy_dtype_to_torch, triton_device

        if not _matches_triton_matmul_profile(input_tensors, output_tensor):
            msg = "Triton matmul supports only 256x256x256 bf16->f32 contiguous nodes."
            raise ValueError(msg)

        m = n = k = dim
        device = triton_device(arch)
        a = torch.randn(
            (m, k), device=device, dtype=numpy_dtype_to_torch(input_tensors[0].dtype)
        )
        b = torch.randn(
            (k, n), device=device, dtype=numpy_dtype_to_torch(input_tensors[1].dtype)
        )
        c = torch.empty(
            (m, n), device=device, dtype=numpy_dtype_to_torch(output_tensor.dtype)
        )
        grid = (triton.cdiv(m, dim), triton.cdiv(n, dim))
        return bare_matmul[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=dim,
            BLOCK_SIZE_N=dim,
            BLOCK_SIZE_K=k,
        )

    return KernelSpec(
        backend=Backend.TRITON,
        op_name="GGML_OP_MUL_MAT",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        function=_compile,
        config={
            "transform_script": str(
                Path(__file__).parent / "triton_kernels" / f"matmul_{arch}.mlir"
            ),
        },
    )


def ggml_op_mul_mat(
    arch: str, input_tensors: list, output_tensor, op_params: bytearray
) -> list[KernelSpec]:
    """Return KernelSpecs for GGML_OP_MUL_MAT.

    IRON is always available (the general path). For nodes matching the fixed
    Triton profile (256x256x256 bf16->f32), the Triton spec is returned first so
    the build system tries it before falling back to IRON.

    Args:
        arch: Target architecture.
        input_tensors: Input tensors A and B.
        output_tensor: Output tensor C.
        op_params: Operation parameters (unused; shape/dtype come from tensors).

    Returns:
        List of KernelSpecs; Triton first iff the profile matches, else IRON only.
    """
    iron = _make_iron_matmul_kernel_spec(arch, input_tensors, output_tensor)
    if _matches_triton_matmul_profile(input_tensors, output_tensor):
        return [
            _make_triton_matmul_kernel_spec(arch, input_tensors, output_tensor),
            iron,
        ]
    return [iron]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_mul_mat_dispatch.py -v
```
Expected: PASS (all three tests). `mul_mat.py` imports `bfloat16` from `ml_dtypes` directly, so the dtype comparison does not depend on numpy string registration.

- [ ] **Step 5: Verify build.py consumes a list here (no regression)**

`build.py`'s `ggml_compile_op` already handles a dispatch function returning either a `KernelSpec` or a `list[KernelSpec]` (the ADD path returns a list). Confirm no other caller assumes `ggml_op_mul_mat` returns a scalar:
```bash
cd /home/ypapadop/workspace-raiders/ggml
grep -rn "ggml_op_mul_mat" src/ggml-hsa/ | grep -v "def ggml_op_mul_mat"
```
Expected: only references are the `_OP_KERNEL_MAP` registration in `build.py` (which calls it generically). If any caller indexes the result as a single spec, stop and report it.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/kernels/mul_mat.py tests/ggml-hsa/test_mul_mat_dispatch.py
git commit -m "Route 256^3 bf16 MUL_MAT to Triton, IRON fallback"
```

---

### Task 3: Verify the Triton matmul compiles to loadable artifacts

Drive `compile_triton_kernel` end-to-end for the 256³ node and confirm it produces `<name>.pdi` + `<name>_insts.bin` (the artifacts the runtime loads). This is the compile-only gate from success criterion 1; it needs the Triton/`amd_triton_npu` toolchain but not the device. Deliverable: a reproducible compile check.

**Files:**
- Test: `tests/ggml-hsa/test_triton_matmul_compile.py` (new, pytest)

**Interfaces:**
- Consumes: `build_triton.compile_triton_kernel(kernel_spec, exported_name, output_directory, logger, verbose)`; `mul_mat._make_triton_matmul_kernel_spec` (Task 2).
- Produces: nothing consumed downstream (verification only).

- [ ] **Step 1: Write the compile test**

Create `tests/ggml-hsa/test_triton_matmul_compile.py`:
```python
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""End-to-end compile check for the Triton matmul (no device required)."""

import logging
import sys
from pathlib import Path

import pytest

KERNELS_DIR = Path(__file__).resolve().parents[2] / "src" / "ggml-hsa" / "kernels"
sys.path.insert(0, str(KERNELS_DIR))

pytest.importorskip("triton")


def _has_npu_backend():
    try:
        import triton.backends.amd_triton_npu.driver  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _has_npu_backend(), reason="amd_triton_npu backend unavailable")
@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_matmul_compiles_to_pdi(tmp_path, arch):
    from build_triton import compile_triton_kernel
    from mul_mat import _make_triton_matmul_kernel_spec
    from tensor_desc import TensorDesc

    def _td(dtype):
        return TensorDesc(dtype=dtype, shape=(256, 256, 1, 1))

    spec = _make_triton_matmul_kernel_spec(
        arch, [_td("bf16"), _td("bf16")], _td("f32")
    )
    name = f"mul_mat_{arch}"
    compile_triton_kernel(spec, name, tmp_path, logging.getLogger("test"), verbose=False)

    assert (tmp_path / f"{name}.pdi").is_file()
    assert (tmp_path / f"{name}_insts.bin").is_file()
```

- [ ] **Step 2: Run the compile test**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_triton_matmul_compile.py -v
```
Expected: PASS for whichever arch(es) the local toolchain targets. If the `amd_triton_npu` backend is not installed in this environment, both are SKIPPED — record that and defer the compile gate to the on-device test in Task 4 (which compiles as a side effect of running). If it FAILS inside `compile_triton_kernel` (e.g. transform-script error), this is the real transform-compatibility risk from the spec: capture the full error and stop for review before Task 4.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
git add tests/ggml-hsa/test_triton_matmul_compile.py
git commit -m "Add compile-only check for Triton matmul artifacts"
```

---

### Task 4: Extend test-mul-mat-hsa.cpp with a bf16 256³ on-device case

Add bf16 support and a parametrized 256³ matmul case that runs on `/dev/accel0`, compares against a CPU f32 reference within tolerance, and thereby validates the stride mapping (Mapping 1 in the spec — the highest-risk detail). Deliverable: `test-mul-mat-hsa` reports PASSED for the bf16 case.

**Files:**
- Modify: `tests/ggml-hsa/test-mul-mat-hsa.cpp`

**Interfaces:**
- Consumes: the dispatch + kernel from Tasks 1–2 (exercised at graph-compute time via the HSA backend); `ggml_fp32_to_bf16` (declared in `ggml.h`).
- Produces: nothing downstream (end-to-end verification).

- [ ] **Step 1: Add a bf16 mul_mat graph helper and a bf16 test routine**

The existing test drives an i16 path via `ggml_mul_mat_i16` (forces an I16 output) and a templated `load_model`/`build_graph`/`compute` pipeline. Rather than force bf16 through the i16 template, add a dedicated self-contained routine (surgical, one fixed shape). Insert the following just above `int main()` in `tests/ggml-hsa/test-mul-mat-hsa.cpp`:

```cpp
#ifdef GGML_USE_HSA
// Build an MUL_MAT node with an explicit F32 output (bf16 inputs, f32 accumulate).
static struct ggml_tensor * ggml_mul_mat_bf16_f32(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));
    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    result->op     = GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;
    return result;
}

// Deterministic pseudo-random float in [-1, 1) from a linear index.
static float pseudo_rand(int i) {
    const uint32_t x = static_cast<uint32_t>(i) * 2654435761u + 1013904223u;
    return (static_cast<float>(x >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
}

// Run the Triton-routed bf16 256x256x256 matmul on the NPU and compare to a
// CPU f32 reference within tolerance. Returns true on pass.
static bool run_bf16_matmul_test(int64_t M, int64_t N, int64_t K) {
    // Host f32 operands.
    std::vector<float> A(M * K), B(K * N);
    for (int64_t i = 0; i < M * K; ++i) A[i] = pseudo_rand((int) i);
    for (int64_t i = 0; i < K * N; ++i) B[i] = pseudo_rand((int) (i + 7919));

    // CPU f32 reference: C[m,n] = sum_k A[m,k] * B[k,n].
    std::vector<float> C_ref(M * N, 0.0f);
    gemm((int) M, (int) N, (int) K, A.data(), B.data(), C_ref.data());

    // bf16 copies of the operands (round-trip through the ggml bf16 encoding).
    std::vector<ggml_bf16_t> A_bf16(M * K), B_bf16(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_bf16[i] = ggml_fp32_to_bf16(A[i]);
    for (int64_t i = 0; i < K * N; ++i) B_bf16[i] = ggml_fp32_to_bf16(B[i]);

    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_hsa_init() failed\n", __func__);
        return false;
    }

    const size_t buf_size = (M * K + K * N) * sizeof(ggml_bf16_t) + 1024;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buf_size);

    ggml_init_params tparams { ggml_tensor_overhead() * 2, NULL, true };
    ggml_context * tctx = ggml_init(tparams);

    // GGML MUL_MAT convention: a = [K, M], b = [K, N]; result = [M, N].
    ggml_tensor * a = ggml_new_tensor_2d(tctx, GGML_TYPE_BF16, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(tctx, GGML_TYPE_BF16, K, N);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tallocr alloc = ggml_tallocr_new(buffer);
    ggml_tallocr_alloc(&alloc, a);
    ggml_tallocr_alloc(&alloc, b);

    // a stores rows of length K (a[m,k]); A is already [M,K] row-major -> direct.
    ggml_backend_tensor_set(a, A_bf16.data(), 0, ggml_nbytes(a));
    // b stores rows of length K (b[n,k]); we need B[k,n], so transpose into [N,K].
    std::vector<ggml_bf16_t> B_nk(N * K);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k)
            B_nk[n * K + k] = B_bf16[k * N + n];
    ggml_backend_tensor_set(b, B_nk.data(), 0, ggml_nbytes(b));

    // Build the graph.
    std::vector<uint8_t> gbuf(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
    ggml_init_params gparams { gbuf.size(), gbuf.data(), true };
    ggml_context * gctx = ggml_init(gparams);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * c = ggml_mul_mat_bf16_f32(gctx, a, b);
    ggml_set_name(c, "c");
    ggml_build_forward_expand(gf, c);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "%s: ggml_gallocr_alloc_graph() failed\n", __func__);
        return false;
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return false;
    }

    // Read back the f32 result and compare within tolerance.
    ggml_tensor * result = ggml_graph_node(gf, -1);
    std::vector<float> C(ggml_nelements(result));
    ggml_backend_tensor_get(result, C.data(), 0, ggml_nbytes(result));

    const float atol = 1e1f, rtol = 1e-1f;
    bool passed = ((int64_t) C.size() == M * N);
    float max_abs = 0.0f;
    for (int64_t i = 0; passed && i < M * N; ++i) {
        const float diff = std::fabs(C[i] - C_ref[i]);
        max_abs = std::max(max_abs, diff);
        if (diff > atol + rtol * std::fabs(C_ref[i])) passed = false;
    }
    printf("bf16 matmul [%ld,%ld,%ld]: max_abs_err=%g -> %s\n",
           M, N, K, max_abs,
           passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    ggml_free(tctx);
    ggml_free(gctx);
    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return passed;
}
#endif // GGML_USE_HSA
```

- [ ] **Step 2: Call the bf16 routine from main()**

In `int main()`, immediately after `ggml_time_init();`, add:
```cpp
#ifdef GGML_USE_HSA
    {
        const bool bf16_ok = run_bf16_matmul_test(256, 256, 256);
        if (!bf16_ok) {
            fprintf(stderr, "bf16 Triton matmul test FAILED\n");
            return 1;
        }
    }
#endif
```

- [ ] **Step 3: Build the test**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml/build
cmake --build . --target test-mul-mat-hsa -j
```
Expected: compiles cleanly. If `ggml_fp32_to_bf16` is undefined at link time, confirm the test links the ggml core lib (it already uses `ggml_backend_*`, so the symbol is in the same lib); if missing, add `#include "ggml.h"` is already present — the declaration is at `ggml.h:378`.

- [ ] **Step 4: Run the test on-device and verify the stride mapping**

Run (use the on-device build/venv from project memory if the plain invocation cannot reach `/dev/accel0`):
```bash
cd /home/ypapadop/workspace-raiders/ggml/build
./bin/test-mul-mat-hsa
```
Expected: `bf16 matmul [256,256,256]: max_abs_err=... -> PASSED`.

If it FAILS with a large `max_abs_err` (result numerically wrong, not a crash), the stride mapping is inverted. The fallback is the other layout choice for B: in Step 1, the code transposes B into `[N,K]` on the host so the device tensor `b` holds `b[n,k]`. If wrong, remove that transpose (set `b` directly from `B_bf16` as `[K,N]` and create `b` as `ggml_new_tensor_2d(tctx, GGML_TYPE_BF16, N, K)` accordingly), rebuild, and re-run. Exactly one of the two layouts matches how the Triton artifact reads B. Record which one worked in the commit message.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
git add tests/ggml-hsa/test-mul-mat-hsa.cpp
git commit -m "Add bf16 256^3 Triton matmul on-device test with tolerance check"
```

---

### Task 5: Final verification against success criteria

Confirm all three success criteria hold together and nothing regressed. Deliverable: a green run of the new tests plus the on-device case.

**Files:** none (verification only).

- [ ] **Step 1: Run the Python test suite for the new files**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_triton_matmul_kernel.py \
                 tests/ggml-hsa/test_mul_mat_dispatch.py \
                 tests/ggml-hsa/test_triton_matmul_compile.py -v
```
Expected: all pass or skip (compile test may skip if toolchain absent). No failures.

- [ ] **Step 2: Run the on-device matmul test**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml/build
./bin/test-mul-mat-hsa
```
Expected: `bf16 matmul [256,256,256]: ... -> PASSED` and the pre-existing i16 case still `PASSED`.

- [ ] **Step 3: Confirm no IRON regression for a non-matching shape**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
python -m pytest tests/ggml-hsa/test_mul_mat_dispatch.py::test_iron_only_for_wrong_shape \
                 tests/ggml-hsa/test_mul_mat_dispatch.py::test_iron_only_for_wrong_dtype -v
```
Expected: PASS — non-256³/non-bf16 nodes still return IRON only.

- [ ] **Step 4: Map results to success criteria and report**

Confirm and report:
1. Criterion 1 (artifacts): Task 3 compile test passed, or Task 4 produced artifacts as a side effect.
2. Criterion 2 (selection): `test_mul_mat_dispatch.py` all green.
3. Criterion 3 (on-device numerics): `test-mul-mat-hsa` bf16 case PASSED within tolerance.
```

