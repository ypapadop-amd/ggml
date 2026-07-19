# Triton matmul port (PoC) — design

Date: 2026-07-19
Status: approved (design), pending implementation plan

## Summary

Port the single-tile bf16 matmul from the Triton-XDNA project into the
`ggml-hsa` backend as a `GGML_OP_MUL_MAT` kernel compiled through the existing
Triton compilation flow (`build_triton.py`). This is a **functional
proof-of-concept**: the goal is to prove a Triton-compiled matmul runs
end-to-end on the NPU. The hand-tuned IRON `gemm.py` remains the general-purpose
matmul path; the Triton kernel is selected only for one fixed, supported profile.

## Goals / non-goals

**Goal:** Prove a Triton-compiled matmul executes correctly as a MUL_MAT kernel
on the NPU, reusing the Triton plumbing that already exists (the same path
`GGML_OP_ADD` uses today).

**Non-goals (explicitly out of scope):**
- Arbitrary (M, N, K) shapes; K-tiling / whole-array multi-column dataflow.
- int8 or f32-native matmul.
- Performance tuning or benchmarking against IRON.
- Retiring or modifying the IRON `gemm.py` dataflow.

## Supported profile

The ported kernel is a single Triton block, shape-specialized (mirrors
`Triton-XDNA/examples/matmul_bf16_m64_n64_k64/`):

- Input dtype: **bf16** (both operands)
- Output dtype: **f32** (bf16 inputs, f32 accumulate) — matches ggml-hsa's
  bf16 MUL_MAT → f32 cap.
- Shape: **M = N = K = 256**, 2D, contiguous operands.

**Note on the example's naming:** the directory `matmul_bf16_m64_n64_k64` and
the `l1_m=64, l1_n=64, l2_k=64` header in the transform scripts refer to the
**L1 tile sizes**, not the GEMM problem size. The example kernel sets
`BLOCK_SIZE_M = BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = K` and its benchmark runs
problem sizes of 256 and up; PHASE 5 of the transform tiles `[16, 16]` for
multi-core herd distribution, which requires the packed problem to exceed a
single 64-wide L1 tile. The PoC therefore computes a **256×256×256** GEMM —
the shape the verbatim transform is actually tuned for — with 64-sized L1
tiling handled internally by the transform.

Any node not matching this profile is handled exactly as today (IRON only).

## Success criteria

1. `build_triton` produces `<name>.pdi` + `<name>_insts.bin` for the 256³ bf16
   node (the same artifacts the runtime already loads).
2. For a matching node, the Triton `KernelSpec` is **first** in the list
   returned by `ggml_op_mul_mat`; for any non-matching node it is **absent**
   (list contains IRON only).
3. `test-mul-mat-hsa` runs a 256³ bf16 matmul on-device (`/dev/accel0` is
   present) and matches a CPU f32 reference within bf16 tolerance
   (`atol ≈ 1e1`, `rtol ≈ 1e-1`, matching the source example's bounds).

## Approach

**Transform script: port verbatim (chosen over generalizing or relying on the
default recipe).** The Triton `@triton.jit` kernel is trivial; all NPU-specific
tiling / packing / vectorization lives in the paired transform-dialect MLIR.
The example's `transform_aie2.mlir` / `transform_aie2p.mlir` are already tuned
for exactly this 256³ bf16 shape, so they are copied verbatim and referenced via
`config["transform_script"]`, matching the existing `vecadd_<arch>.mlir`
convention. Generalizing the transform to arbitrary shapes is the hard,
uncertain part and is out of scope for a PoC.

## Components & file changes

Layout mirrors the existing `vecadd` Triton kernel.

**New files:**
- `src/ggml-hsa/kernels/triton_kernels/matmul.py` — `bare_matmul` `@triton.jit`
  kernel, ported verbatim from the Triton-XDNA example.
- `src/ggml-hsa/kernels/triton_kernels/matmul_aie2.mlir` — transform script,
  copied verbatim from the example's `transform_aie2.mlir`.
- `src/ggml-hsa/kernels/triton_kernels/matmul_aie2p.mlir` — transform script,
  copied verbatim from the example's `transform_aie2p.mlir`.

**Edited files:**
- `src/ggml-hsa/kernels/mul_mat.py` — add `_make_triton_matmul_kernel_spec`
  plus a profile-match guard. `ggml_op_mul_mat` returns `[Triton, IRON]` when
  the node matches the supported profile, else `[IRON]` (return type widens
  from `KernelSpec` to `list[KernelSpec]`). This is the only dispatch change.
- `src/ggml-hsa/kernels/triton_kernels/CMakeLists.txt` — add the three new
  files to `TRITON_FILES` (copy + install).
- `tests/ggml-hsa/test-mul-mat-hsa.cpp` — add bf16 support to
  `fundamental_to_ggml_type`, parametrize dims, add a 256³ bf16 case with a
  tolerance check.

**No C++ backend/runtime changes.** The `.pdi` / `_insts.bin` dispatch and
launch contract is backend-agnostic; it is keyed only on the produced artifacts.

## Data flow

Dispatch reuses existing infrastructure:

```
ggml_op_mul_mat  ->  [Triton spec, IRON spec]   (Triton first iff profile matches)
build.py         ->  tries each backend in order, falls back on exception
Triton _compile  ->  builds torch tensors, launches bare_matmul[grid]
build_triton.py  ->  amd_triton_npu pipeline, config["transform_script"]=matmul_<arch>.mlir
                 ->  xclbin -> PDI extracted -> <name>.pdi + <name>_insts.bin
runtime          ->  loads artifacts, dispatches kernarg [srcA, srcB, dst, sizeA, sizeB, sizeC]
```

### Mapping 1 — ggml layout → standard A@B via strides

ggml MUL_MAT computes `dst[m,n] = Σ_k src0[k,m] · src1[k,n]` — both operands
share K as `ne0`. The Triton `bare_matmul` is a standard `A[M,K] @ B[K,N]` but
takes **all strides as `constexpr` args**. We therefore pass strides that
present `src0` as `A[M,K]` and `src1` (stride-transposed) as `B[K,N]` — no
kernel edit and no data movement. This stride mapping is the one place that can
be silently wrong, so it is verified against a CPU reference in the test using
non-trivial (non-identity) input data.

### Mapping 2 — profile guard

The guard in `mul_mat.py` emits the Triton spec (first) only when: src0/src1
dtype == bf16, output dtype == f32, and the problem is exactly
M == N == K == 256 (i.e. `src0->ne == [256, 256]`, `src1->ne == [256, 256]`,
`dst->ne == [256, 256]`), with operands contiguous. The deferred `_compile`
**re-checks**
the same profile and raises on mismatch (defensive, matching the ADD fallback
pattern), so any drift between guard and compile degrades to IRON rather than
miscompiling.

## Testing & verification

**In-test (`test-mul-mat-hsa.cpp`):**
- Add bf16 to `fundamental_to_ggml_type` (`GGML_TYPE_BF16`) and a bf16
  print/compare path.
- Parametrize M/N/K; add a 256³ bf16 case alongside the existing i16 path.
- Fill A/B with non-trivial values (not identity) so the stride mapping is
  exercised; compute the CPU f32 reference with the existing naive `gemm`;
  compare NPU output within tolerance.
- Pass condition: max abs/rel error within tolerance **and** element count
  == M·N.

**Manual verification (before marking done):**
1. `build_triton` on a 256³ bf16 node emits both artifacts (compile-only,
   no device needed).
2. `ggml_op_mul_mat` returns Triton-first for 256³ bf16 and IRON-only otherwise
   (quick Python check).
3. Run `test-mul-mat-hsa` on `/dev/accel0`; confirm the bf16 case passes
   tolerance.

## Risks

- **Stride mapping (Mapping 1)** is the highest-risk detail; mitigated by the
  non-identity CPU-reference tolerance check.
- **Transform-script arch coverage:** verbatim scripts are tuned for 256³ bf16;
  behavior on the other arch variant is confirmed by running on the actual
  device present in this environment.
- Low integration risk overall: the Triton path is already wired for ADD and
  the runtime contract is unchanged.
