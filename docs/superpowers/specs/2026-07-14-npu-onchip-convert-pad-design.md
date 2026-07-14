# On-device convert/pad/depad kernels for f32 MUL_MAT (ggml-hsa)

Status: **design approved, not yet implemented**
Date: 2026-07-14
Branch: `ypapadop-amd/matmul`

## Goal

Remove the two per-op host synchronization points around every f32 `MUL_MAT` on the HSA
(AIE/NPU) backend by moving the f32→bf16 conversion, zero-padding, and result de-padding from
the **host** onto **on-device kernels** dispatched on the same in-order queue. This lets the
already-landed packet-batching machinery (PR #199) span the matmul instead of being flushed at
every one, collapsing the MNIST graph toward a single doorbell run.

**Success criterion (scope 2 — visible speedup):**
1. `test-backend-ops-mnist test -o MUL_MAT` still passes 3/3 at the strict 5e-4 tolerance.
2. `mnist-eval ... HSA0` still reports 98.00% accuracy AND shows a measurable us/image improvement
   over the ~353 us/image baseline, with a reduced doorbell count.
3. New standalone per-kernel tests pass; existing hsa tests (`test-batch-dispatch-hsa`, etc.)
   remain green.

## Background

### Current f32 MUL_MAT path

`ggml_hsa_prepare_mul_mat_f32` (`src/ggml-hsa/ggml-hsa.cpp`) rewrites an f32 MUL_MAT node so the
GEMM sees **bf16 operands zero-padded** to the per-architecture tile multiples (aie2 tile=32,
aie2p tile=16; see the gemm tiling notes). It sets `convert_dtype`/`depad` flags and sizes
internal device buffers. `allocate_internal_storage` pre-zeroes the whole internal block so pad
gaps (interior K gap + trailing M/N rows/cols) read as zero.

At graph-compute time (`ggml_backend_hsa_graph_compute`, ggml-hsa.cpp:1450-1494), each such node
does, on the **host**:

```
drain (wait_dispatches) → host convert+scatter src0,src1 (ggml_hsa_copy_subblock)
→ dispatch GEMM
→ drain (wait_dispatches) → host gather+depad dst (ggml_hsa_copy_subblock)
```

The convert/scatter/depad are CPU loops (`ggml_hsa_copy_subblock_f` in `host-ops.cpp:77`,
`nb[]`-stride driven, dtype-templated).

### Why this matters (from `docs/ggml-hsa-dispatch-overhead.md`)

- Packet batching (Optimization #1) **already landed** in PR #199: `dispatch` writes packets and
  only rings the doorbell every `dispatch_batch_size` packets (`aie-kernel.cpp:90`).
- The queue is `HSA_QUEUE_TYPE_SINGLE` (**in-order**); same-queue work is implicitly ordered
  (see the `event_wait` comment, ggml-hsa.cpp:1576). No host barrier is required for correctness
  between consecutive on-queue kernels.
- The drains' wall-clock is negligible (~0.1 ms), but each drain **flushes the batch**, so every
  f32 MUL_MAT fragments the graph and defeats #199's batching. Removing the drains is the unlock
  for the visible speedup.

This design implements Optimization #3 (on-device pad/convert), which the doc identifies as the
enabler that lets #1 realize its value.

## Approach (chosen: A — standalone pre/post-amble kernels)

Replace the host copies for the `depad` path with on-device kernels dispatched on the same queue:

```
dispatch CONVERT_PAD(src0 f32 → src0' bf16 padded)
dispatch CONVERT_PAD(src1 f32 → src1' bf16 padded)
dispatch GEMM(src0', src1' → dst' f32 padded)     [UNCHANGED]
dispatch DEPAD(dst' f32 padded → dst f32)
```

No `wait_dispatches`, no flush between these. All four packets chain in-order and batch with
surrounding elementwise ops.

Rejected alternatives:
- **B — fold convert+pad into `gemm.py` data movement:** touches the rigid upstream whole-array
  GEMM and `mm.cc` (no f32-input microkernel combo exists); high risk against the 5e-4 tolerance
  and the known tile-size landmines. Larger change for no additional structural benefit.
- **C — A + cached converted weights:** valid follow-up (weights are constant across batches, so
  their conversion could be done once), but caps at the ~5% `fill` bucket on its own and adds
  lifetime/caching bookkeeping. Out of scope here; revisit after A shows the batching win.

### Key enablers (verified in current tree)

- `ggml_hsa_aie_kernel::dispatch` (`aie-kernel.cpp:13`) operates on arbitrary `ggml_tensor*` with
  `.data` set — **not** graph nodes. Convert/depad kernels dispatch directly against the internal
  device buffers `allocate_internal_storage` already allocates.
- The internal block is pre-zeroed (memset in `allocate_internal_storage`), so CONVERT_PAD writes
  only valid regions and pad gaps stay zero. **This memset must remain.**
- `ggml_hsa_copy_subblock_f` (`host-ops.cpp:77`) is a pure `nb[]`-strided nested copy, identical in
  both directions (convert+pad and depad differ only by src/dst swap and dtypes). It is already
  proven correct at 5e-4, so it is the reference the kernels' DMA access patterns replicate.

## Components

### 1. Two new IRON kernels (synthetic internal ops)

These are **not** GGML ops. They compile through the existing `ggml_compile_op` → IRON → PDI
pipeline, keyed by `ggml_hsa_create_kernel_name` using two synthetic op-names introduced for this
purpose: `HSA_CONVERT_PAD` and `HSA_DEPAD`. Each is registered in `_OP_KERNEL_MAP` (`build.py`)
with a dispatch wrapper and IRON design, mirroring `scale`:

- `kernels/convert_pad.py` / `kernels/depad.py` — dispatch wrappers returning a `KernelSpec`.
- `kernels/iron_kernels/convert_pad.py` / `depad.py` — IRON designs (ObjectFifo data movement).
- `kernels/iron_kernels/convert_pad.cc` / `depad.cc` — C++ compute kernels (`extern "C"`).
- Register `.py`/`.cc` in both `kernels/CMakeLists.txt` and
  `kernels/iron_kernels/CMakeLists.txt` (files are enumerated and copied to the build tree).

The kernel name encodes op-name + in/out dtypes + padded/unpadded shapes, so each distinct
(M,N,K,pad,dtype) combination caches its own PDI.

**CONVERT_PAD** (`f32 [d0,d1] → bf16 [d0pad,d1pad]`):
- Compute (`.cc`): elementwise f32→bf16 cast, vectorized like `scale.cc`
  (`aie::load_v` f32 → `.to_vector<bfloat16>()` → `aie::store_v`), scalar tail. bf16 rounding
  must match `ggml_hsa_type_traits<BF16>::from_fp32`.
- Data movement (`.py`): output DMA tap scatters each valid row of `d1` elements into the padded
  buffer using the padded row stride (`d0pad`), leaving the interior gap and trailing rows/cols
  untouched (pre-zeroed). Input tap contiguous over logical `[d0,d1]`. The tap sizes/strides
  replicate `ggml_hsa_copy_subblock_f`'s `nb[]`-derived indexing.

**DEPAD** (`f32 [Mpad,Npad] → f32 [M,N]`, no cast):
- Gather: input tap reads the `[M,N]` sub-block out of the padded buffer with the padded stride;
  output contiguous. Same `nb[]`-strided pattern as CONVERT_PAD with src/dst swapped and identical
  dtypes. Whatever layout convention (incl. the GEMM's column-major C) makes the host subblock
  copy correct carries over directly through the shared stride math — no separate col-major
  derivation.

### 2. C++ integration

- **`ggml_backend_hsa_tensor_extra`** (`common.hpp:338`): add `std::shared_ptr<ggml_hsa_kernel>`
  members for the per-source convert kernels and the depad kernel (e.g.
  `src_convert_kernel[GGML_MAX_SRC]`, `depad_kernel`). Build and cache them in the ctor alongside
  the GEMM kernel (same `ggml_hsa_create_kernel` path with the synthetic op-names), so they are
  ready at dispatch time with no per-graph-compute compilation.
- **`ggml_backend_hsa_graph_compute`** (`ggml-hsa.cpp:1450-1494`), `depad` branch only:
  - Pre-block: replace `wait_dispatches` + per-source `ggml_hsa_copy_subblock` with a
    CONVERT_PAD `dispatch` per f32 source (parent src → internal src'). No drain.
  - Post-block: replace `wait_dispatches` + `ggml_hsa_copy_subblock` with a DEPAD `dispatch`
    (internal dst' → parent dst). No drain.
  - Non-`depad` paths (plain `convert_dtype`, layout copies, other ops) are unchanged.
- **Fallback:** keep the host copy path intact. If a convert/depad kernel fails to
  compile/load (JIT miss), fall back to the existing host `ggml_hsa_copy_subblock` + drain for
  that node so the op degrades gracefully rather than failing.

## Correctness invariants

- CONVERT_PAD writes only valid regions; pad gaps rely on the existing pre-zero memset → keep it.
- bf16 rounding in the AIE kernel must match the host `from_fp32` path; verify in numpy before
  hardware.
- In-order queue provides src'→GEMM→dst ordering with no host barrier. Only the MUL_MAT `depad`
  drains are removed; no other consumer's drain is touched.
- Column-major B/C layout constraints of the whole-array GEMM are preserved (GEMM is unchanged;
  depad replicates the proven host stride math).

## Testing

1. **Numpy pre-check:** both kernels' index math validated against `ggml_hsa_copy_subblock_f`
   semantics before touching hardware (diff 0.0 on scatter/gather; bf16 ULP on the cast).
2. **`test-backend-ops-mnist test -o MUL_MAT`:** still 3/3 at 5e-4.
3. **End-to-end `mnist-eval ... HSA0`:** 98.00% accuracy AND measurable us/image improvement +
   reduced doorbell count vs the ~353 baseline (reproduction steps in
   `docs/ggml-hsa-dispatch-overhead.md`).
4. **`test-batch-dispatch-hsa`** and other existing hsa tests remain green.

### New standalone per-kernel tests

Live next to the other hsa tests (`tests/ggml-hsa/`), each its own executable added to
`tests/ggml-hsa/CMakeLists.txt` mirroring the existing pattern (`add_executable` +
`target_link_libraries(... ggml)` + `add_test` + `LLVM_PROFILE_FILE` property):

- **`test-convert-pad-hsa.cpp`:** allocate an f32 source on the HSA backend, dispatch CONVERT_PAD
  into a padded bf16 device buffer, read back, compare against a host reference computed with
  `ggml_hsa_copy_subblock_f` semantics (valid region bf16-rounded; pad gaps 0). Cases: exact tile
  multiple (no pad), K-only pad, M/N-only pad, all three padded.
- **`test-depad-hsa.cpp`:** populate a padded f32 `[Mpad,Npad]` device buffer with a known
  pattern, dispatch DEPAD into `[M,N]`, read back, compare to the sliced sub-block. Same
  pad-shape matrix.

**Test hook (internal, not public API):** because these kernels are not reachable through
`ggml_backend_graph_compute`, add a narrow internal test-support header under `src/ggml-hsa/`
(e.g. `ggml-hsa-test-support.hpp`) exposing a function that, given two device-buffer
`ggml_tensor`s, builds the kernel via the existing `ggml_hsa_create_kernel` path (with the
synthetic op-name) and calls `dispatch` + `wait`. Production `graph_compute` uses the same builder,
so the tests exercise the real dispatch path, not a mock. This keeps the hook out of the public
`ggml-hsa.h` surface.

## Files touched

- `src/ggml-hsa/ggml-hsa.cpp` — build/cache convert+depad kernels in `tensor_extra` ctor; replace
  host copies in the `depad` branch of `graph_compute` with dispatches; keep host fallback.
- `src/ggml-hsa/common.hpp` — new kernel shared_ptr members on `ggml_backend_hsa_tensor_extra`.
- `src/ggml-hsa/host-ops.cpp` — reference for the on-device index math (kept as fallback).
- `src/ggml-hsa/kernels/build.py` — register `HSA_CONVERT_PAD`, `HSA_DEPAD` in `_OP_KERNEL_MAP`.
- `src/ggml-hsa/kernels/convert_pad.py`, `kernels/depad.py` — dispatch wrappers (new).
- `src/ggml-hsa/kernels/iron_kernels/convert_pad.py|.cc`, `depad.py|.cc` — IRON designs +
  compute (new).
- `src/ggml-hsa/kernels/CMakeLists.txt`, `kernels/iron_kernels/CMakeLists.txt` — enumerate new
  files.
- `src/ggml-hsa/ggml-hsa-test-support.hpp` — internal test hook (new).
- `tests/ggml-hsa/test-convert-pad-hsa.cpp`, `test-depad-hsa.cpp` — new tests.
- `tests/ggml-hsa/CMakeLists.txt` — register new test executables.

## Out of scope

- Weight-conversion caching (Optimization #3 alternative / C) — follow-up after the batching win.
- Any change to the whole-array GEMM or `mm.cc` microkernel.
- Generalizing the pre/post-amble mechanism to other ops (the convert/depad kernels are written as
  reusable primitives, but wiring other ops through them is future work).
```
