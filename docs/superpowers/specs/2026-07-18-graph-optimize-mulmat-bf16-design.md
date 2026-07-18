# Move MUL_MAT f32→bf16 cast fusion into a graph_optimize hook

**Date:** 2026-07-18
**Branch:** ypapadop-amd/matmul
**Status:** Design approved, pending implementation plan

## Problem

`ggml_mul_mat` always produces an f32 result (hardcoded `GGML_TYPE_F32` in
`ggml.c`), regardless of input dtype. The bf16 MNIST graph
(`mnist_model_build_bf16`) therefore wraps every GEMM in
`ggml_cast(ggml_mul_mat(...), BF16)` to get back onto the bf16 chain for the
following ADD/RELU. That cast is dtype plumbing, not math — the model has no
real f32 intermediate.

Today the HSA backend hides this cost with a graph_compute-time fusion
(`ggml_hsa_fuse_mul_mat_narrow`): a whole-graph O(N²) scan finds each padded
MUL_MAT whose sole consumer is that cast, redirects the MUL_MAT's de-pad
post-amble to narrow f32→bf16 straight into the cast's buffer, and marks the
cast skipped. It works (~62 µs/image, bit-identical) but carries a lot of
machinery: the `fusion_t` struct (`narrow_dst` / `skip_dispatch` / `analyzed`),
the O(N²) consumer scan, and a `post_dst` branch in the postprocess dispatch.

`graph_optimize` is a backend iface hook (`ggml-backend-impl.h:139`) that runs
per-split **before** allocation and **before** `tensor_extra` construction
(`ggml-backend.cpp:1417`). It is the sanctioned place to rewrite the graph. The
HSA backend currently sets it to `nullptr` (`ggml-hsa.cpp:1920`). We can use it
to retype the MUL_MAT to bf16 up front, so the existing de-pad (which already
converts f32→bf16 when the destination is bf16) and the framework's existing
empty-op skip do all the work — deleting the custom fusion machinery.

## Goal

Replace the graph_compute-time fusion with a `graph_optimize` hook that:

1. retypes a qualifying MUL_MAT node to bf16,
2. rewires its cast consumers' downstream readers onto the MUL_MAT, and
3. blanks each orphaned cast to `GGML_OP_NONE`.

Success is **perf-neutral and bit-identical** to today's fused path, with the
custom fusion code deleted. This is a simplification / correctness-robustness
change, not a speedup.

## Non-goals

- CPU-runnable bf16 graph. CPU `ggml_compute_forward_mul_mat` asserts
  `nb0 == sizeof(float)`, so a bf16-dst MUL_MAT cannot fall back to CPU. The
  bf16 example becomes NPU-only. Accepted: it is an NPU benchmark, and in
  practice HSA supports the op so the fallback never fires.
- Changing core `ggml_mul_mat`.
- Touching the stock f32 `mnist-eval` / `mnist_model_build`.
- Removing the `ggml_cast` from the source graph. It stays for expressivity;
  the backend collapses it, the author does not delete it.

## Why graph_optimize can retype but cannot truly delete a node

Verified in the scheduler:

- **Order.** `graph_optimize` (`ggml-backend.cpp:1417`) runs before
  `ggml_backend_sched_alloc_splits` (:1489+). Mutations are seen by allocation.
- **`src` rewires and type changes persist.** `ggml_graph_view` shares the
  tensor objects (`nodes = cgraph0->nodes + i0`); the sched copy-back re-reads
  the same tensor structs, so field mutations stick.
- **Node removal does NOT persist.** The copy-back rebuilds `graph_copy` from
  the original `[i_start, i_end)` range (`ggml-backend.cpp:1438-1441`),
  ignoring the view's `n_nodes`. Splicing a node out of the array and
  decrementing `n_nodes` is undone. Every in-tree backend that implements
  `graph_optimize` (metal, hexagon) only **reorders**; none deletes.

So the orphaned cast node still enters `graph_copy` and is still allocated /
walked. To make it do nothing we set its `op = GGML_OP_NONE`; the framework's
`ggml_op_is_empty` check at the top of the compute loop then skips it, and
gallocr gives it no buffer (0 children).

**Allocation correctness after rewire.** `n_children` / `n_views` are
recomputed at alloc time by walking `node->src[j]` (`ggml-alloc.c:753`), which
runs after `graph_optimize`, so the rewired topology is counted correctly. The
graph-level `use_counts` (`ggml.c:6955`) are frozen at build time and NOT
recomputed, but the gallocr path used here relies on `n_children`, so this is
benign for a NONE'd, zero-reader node.

## Architecture

### New shared predicate

`ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor & mm) -> bool`

Eligibility for the padded bf16 GEMM path, over a raw `ggml_tensor` (so it can
be called from `graph_optimize`, before `tensor_extra` exists):

- `mm.op == GGML_OP_MUL_MAT` and exactly 2 sources;
- both sources are `F32` or `BF16`;
- trivial layout on both sources and the destination;
- no batch / broadcast (`ne[2] == 1 && ne[3] == 1` on both sources).

Deliberately **excludes** the `dst.type` check — that is the one field the two
callers disagree on. `prepare_mul_mat` and `graph_optimize` both call this, so
their eligibility cannot diverge.

### New hook: `ggml_backend_hsa_graph_optimize(backend, cgraph)`

Wired into the HSA iface (replaces the `nullptr` at `ggml-hsa.cpp:1920`).
For each node `mm` in the split graph:

1. **Qualify** (all must hold):
   - `ggml_hsa_mul_mat_is_padded_gemm(*mm)`;
   - `mm` is not `GGML_TENSOR_FLAG_OUTPUT`;
   - `mm` has **≥1 consumer**, and **every** consumer is a pure f32→bf16
     convert-cast that is **not** `GGML_TENSOR_FLAG_OUTPUT`. A convert-cast is
     `ggml_hsa_is_convert_copy(*c) && c->src[0]->type == F32 && c->type == BF16`.
   - If any consumer reads `mm` as f32 directly (matmul, add, f32 output, or a
     cast to a different dtype), **disqualify entirely** — retyping would feed
     bf16 where f32 is expected. All-or-nothing.

2. **Rewrite:**
   - `mm->type = GGML_TYPE_BF16`; recompute `mm->nb[]` as bf16 contiguous
     strides.
   - For **each** cast consumer `cast`: for every node `c` in the graph with
     `c->src[k] == cast`, set `c->src[k] = mm`; then `cast->op = GGML_OP_NONE`.

Because all consumers were casts and each cast's readers are rewired onto `mm`,
every orphaned cast ends with zero readers and is a true dead node.

### Modified: `ggml_hsa_prepare_mul_mat_f32`

- Replace the inline entry gate (`ggml-hsa.cpp:663-682`) with a call to
  `ggml_hsa_mul_mat_is_padded_gemm`, then its own dtype rule: accept
  `dst.type == F32` **or** `BF16` (was F32-only at :675).
- Everything downstream is unchanged: the function already builds the de-pad
  from the destination's dtype, so a bf16 dst yields the f32→bf16 convert
  de-pad automatically. `retype ⟹ prepare accepts`, because both use the same
  predicate.

### Deleted

- `ggml_hsa_fuse_mul_mat_narrow` (whole function) and its call site in
  graph_compute (`ggml-hsa.cpp:1654`).
- `fusion_t` struct and the `fusion` member in `common.hpp` (:391-407).
- The `skip_dispatch` early-continue in graph_compute (:1669) — replaced by the
  framework's existing `ggml_op_is_empty` skip already at the top of the loop.
- The `post_dst` / `narrow_dst` branch in the postprocess dispatch
  (:1761-1763) → reverts to dispatching into `*node`.

### Unchanged

`depad.cc` / `depad.py` (already f32→bf16 capable), `convert_pad`, constant
weight/bias caching, `use_device_transforms`, and the source preprocess path.

## Data flow (bf16 MNIST, per batch, after the change)

```
graph_optimize:  mm1.type f32→BF16; add1.src[0] cast1→mm1; cast1.op→NONE
                 (same for mm2 / cast2)
alloc:           cast1/cast2 have 0 children → no buffer;
                 mm1/mm2 own bf16 buffers
graph_compute:   mul_mat → depad(f32 padded temp → BF16 into mm's buffer)
                 → add reads mm directly
                 cast nodes hit ggml_op_is_empty → skipped, no dispatch
```

Identical on-device work to today's fused path: one GEMM + one convert-de-pad
per layer, no separate cast dispatch, no f32 [M, N] round trip.

## Error handling / fallbacks

- **Non-qualifying MUL_MAT** (f32 consumer, mixed-dtype casts, output cast,
  batched) → predicate false, node untouched, runs the normal f32 de-pad path.
  The stock f32 `mnist-eval` graph is unaffected (its MUL_MATs feed f32
  ADD/RELU → disqualified).
- **prepare_mul_mat stays the correctness gate.** graph_optimize only retypes
  when `is_padded_gemm` is true — the same predicate prepare uses — so a retype
  implies prepare will accept the bf16 dst. They cannot diverge (one function).
- **No opportunistic kernel-build fallback anymore.** Today's fusion falls back
  to de-pad + separate cast if the fused kernel fails to build. Under this
  design the convert-de-pad is the only path; a build failure is a hard failure
  surfaced via the normal dispatch error. Acceptable: that kernel already ships
  and is exercised by the current fused path.

## Testing / success criteria

1. **Bit-exact:** `mnist-eval-bf16 … HSA0` → loss `0.066342`, acc `98.00%`,
   unchanged from today's fused numbers.
2. **Perf-neutral:** ~62 µs/image (regression is a bug, not an expected cost).
3. **f32 untouched:** `mnist-eval … HSA0` → ~48 µs, loss `0.066372`.
4. **Op gate:** `test-backend-ops-mnist -o MUL_MAT` 3/3 — standalone MUL_MAT
   has no following cast, so graph_optimize does not fire; confirms selectivity.
5. **Depad kernel gate:** `test-depad-hsa` PASS (f32→f32 path intact).
6. **CPU sanity:** `mnist-eval CPU` unaffected (~2.8 µs) — the f32 graph never
   hits the hook.

All runs on a **Release** build with the venv active (JIT compiles kernels in
embedded Python). After editing kernels, move `~/.cache/ggml/aie2/` aside
(`mv`, not `rm`) to force recompile.

## Files touched

- `src/ggml-hsa/ggml-hsa.cpp` — new predicate, new hook, prepare gate relax,
  delete fusion function + call site + post_dst branch + skip_dispatch check,
  wire hook into iface.
- `src/ggml-hsa/common.hpp` — delete `fusion_t` struct and `fusion` member.
