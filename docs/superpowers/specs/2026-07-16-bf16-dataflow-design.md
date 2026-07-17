# bf16 dataflow for ggml-hsa MUL_MAT chains (dtype-collapse)

Date: 2026-07-16
Status: approved (design), revised after code review

## Problem

The AMD XDNA NPU (ggml-hsa backend) has no native f32 GEMM. Today
`ggml_hsa_prepare_mul_mat_f32` (`src/ggml-hsa/ggml-hsa.cpp:638`) rewrites every f32
`MUL_MAT` into a bf16 padded GEMM wrapped by two conversions:

- a **pre-amble** `convert_pad` on *each* operand: f32 -> bf16 + zero-pad to the GEMM
  tile multiples;
- the GEMM itself: bf16 x bf16, **f32 accumulate**, padded **f32** output;
- a **post-amble** `depad`: f32 padded -> f32 dense, gathered back into the parent.

The MNIST fully-connected forward graph (built f32 in
`examples/mnist/mnist-common.cpp:319-323`) is:

```
images(f32) -> MUL_MAT(fc1_w) -> ADD(fc1_bias) -> RELU -> MUL_MAT(fc2_w) -> ADD(fc2_bias) -> logits(f32)
```

Because each MUL_MAT depads back to f32 and the next one re-converts, the data
round-trips through f32 between the two GEMMs: `depad -> f32`, `ADD`+`RELU` in f32,
`convert_pad -> bf16` again. NPUs are more efficient in bf16 (half the bytes moved,
native compute), so these mid-graph conversions are pure overhead.

Two distinct costs are bundled in the current path:

1. **Constants are re-converted every dispatch.** `convert_dtype = true` is set on *both*
   MUL_MAT sources (`ggml-hsa.cpp:691, 699`), and `graph_compute` re-dispatches the
   convert_pad pre-amble on every image (`ggml-hsa.cpp:1539`). The weight
   (`[784 x 500]` for fc1) is a **constant** (`ggml_set_param`, a graph leaf) yet is
   converted+padded once per image (10,000x over the eval set). Likewise the biases.
2. **The activation round-trip** (depad -> f32 -> ADD/RELU -> convert_pad -> bf16)
   between the GEMMs.

## Goal

Eliminate the redundant re-conversion of constants and the mid-graph f32<->bf16
**dtype** conversions in f32 MUL_MAT chains on the ggml-hsa backend. After this work the
only dtype conversions per dispatch are:

- **image in**: f32 -> bf16, once at the chain entry;
- **logits out**: bf16 -> f32, once at the chain exit (logits are asserted f32 at
  `mnist-common.cpp:378`);
- **constants**: converted exactly **once** (first dispatch), then reused.

Scope decision (**A1, dtype-collapse**): the pad/depad **shape** steps are retained but
carry bf16 (no dtype math). Making ADD/RELU padding-aware to remove pad/depad entirely
(A2) is **out of scope**.

## Key lifecycle facts (established by code review)

These correct the first draft, which assumed a `graph_optimize`-driven rewrite:

- **`graph_optimize` runs too late to drive the rewrite.** The dtype/shape rewrite lives
  in `ggml_hsa_prepare_mul_mat_f32`, called from the `tensor_extra` constructor at
  **`buffer_init_tensor`** time (`ggml-hsa.cpp:1051`), i.e. when weights are loaded.
  `graph_optimize` is called by the scheduler at **compute time, per split**, on a
  `ggml_graph_view` (`ggml-backend.cpp:1417`) -- long after `tensor_extra` and its
  internal buffers already exist. It **cannot** retroactively change the per-node dtype
  decision or re-allocate buffers. It is therefore **not** the vehicle for this work.
- **A node sees only its sources, never its consumers,** in the `tensor_extra`
  constructor. Any cross-node "emit bf16 because a GEMM consumes me" decision needs a
  pre-pass over the whole graph. Phase 2 (below) uses a lightweight consumer-lookup at
  init time, not `graph_optimize`.
- **The internal bf16 buffer is persistent.** `allocate_internal_storage`
  (`ggml-hsa.cpp:844`) allocates a `unique_ptr buffer` owned by the `tensor_extra`
  (stored in the buffer context, `ggml-hsa.cpp:1057`), memset-zeroed **once**
  (`ggml-hsa.cpp:875`). It survives across all dispatches -- this is what makes the
  constant cache (Phase 1) possible without any new allocation.
- **Constants are distinguishable from activations.** Weights/biases are `ggml_set_param`
  leaves (`op == GGML_OP_NONE`, not `GGML_TENSOR_FLAG_INPUT`); `images` is
  `ggml_set_input` (`mnist-common.cpp:233`). Predicate for "reuse the converted buffer":
  the source is a leaf whose `data` pointer is stable and is not flagged INPUT.

## Phased plan

The review split the work cleanly: Phase 1 needs no new kernels and is numerically free;
Phase 2 needs net-new bf16 kernels and drifts numerics. Land and measure Phase 1 first.

### Phase 1 -- Cache converted+padded constants (safe, dominant win)

**Mechanism:** the persistent internal buffer already holds the converted+padded bf16
weight after the first dispatch. For a constant source, skip the convert_pad pre-amble on
every subsequent dispatch instead of recomputing identical bits.

**Changes (all in `ggml-hsa.cpp`):**
1. Add a per-source flag on `tensor_extra` marking a source as a **cacheable constant**
   (leaf, `op == GGML_OP_NONE`, not `GGML_TENSOR_FLAG_INPUT`, stable `data` pointer).
   Set it in the `tensor_extra` constructor where the padded MUL_MAT path is chosen.
2. Add a per-source runtime **latch** ("already converted") on `tensor_extra`.
3. In `graph_compute` (`ggml-hsa.cpp:1518-1555`), when a source is a cacheable constant
   and the latch is set, **skip** its preprocess dispatch (device path) / sub-block copy
   (host path). On the first dispatch, run it and set the latch.

**Numerics:** identical bits every dispatch -> **zero** numeric change. Bit-identity to
the current baseline (loss 0.066372) is preserved.

**Risk:** if a constant's `data` were mutated between runs the cache would be stale. Guard
by keying the latch on the observed `data` pointer and clearing it if the pointer changes;
`ggml_set_param` weights in eval are static, so this is belt-and-suspenders.

### Phase 2 -- bf16 intermediates (drifts numerics; gated on accuracy)

Only after Phase 1 is measured. Threads bf16 through the middle of the chain.

**New kernels required (the review found these do NOT exist today):**
- **bf16 pad-only** (dense bf16 -> padded bf16) for the MM2 input. `convert_pad` is
  f32->bf16 by definition and `convert_pad.py:71` rejects non-bf16 output but requires f32
  **input**; a pad-only bf16 variant is new.
- **bf16 depad** (padded bf16 -> dense bf16) for the bf16 GEMM output. `depad.py:66`
  hard-rejects any non-f32 src/dst; a bf16 instantiation is new.
  (Both are shape-only shuffles; the depad kernel body is dtype-agnostic, but the Python
  design's dtype guard and the C++ transform-kernel wiring must accept bf16.)

**Constant handling:** biases must also be cached as **bf16** (Phase 1 mechanism), because
the ADD bias fast-path is gated on all three operands sharing a dtype
(`binary_ops.py:506, 514`); a bf16 activation + f32 bias would silently fall through to the
scalar per-element broadcast path, discarding the 2.9x vectorization win.

**GEMM:** emit `dtype_out = bf16` (listed valid at `gemm.py:70`; **verify** a built
`matmul_bf16_bf16` object exists before relying on it -- open item).

**Final boundary:** after MM2, `ADD(fc2_bias)` must produce **f32** logits. Chosen
ordering: bf16 depad -> bf16 `ADD` -> a single bf16 -> f32 convert into the logits tensor.
(The alternative, a mixed bf16+bf16->f32 ADD, drops out of the same-dtype fast-path, so it
is rejected.)

**Cross-node decision:** at `tensor_extra` init, a MUL_MAT whose consumer chain is
`ADD -> RELU -> MUL_MAT` keeps its output bf16 and skips the depad-to-f32. Implemented via
a consumer lookup over the graph at init, not `graph_optimize`.

**RELU:** runs bf16 (dtype-generic via `-DINPUT_DTYPE`/`-DOUTPUT_DTYPE`,
`unary_ops.py:144-145`); uses aligned `load_v`/`store_v` (`unary_ops.cc:188`). **Verify**
the aligned bf16 store path is correct for RELU's tiling (the convert_pad memory notes the
*unaligned* 16-bit store is broken in this aie_api version; aligned is expected fine --
open item).

## Acceptance gate

- **Phase 1**: MNIST 10k FC f32 e2e stays **bit-identical** (loss 0.066372, acc 98.00%);
  measured throughput improves (per-image weight convert_pad dispatch gone).
- **Phase 2**: test accuracy **>= 98.00%**; bit-identity dropped (bf16 intermediates drift
  by design); record loss drift for the record (not a gate).
- Kernel-level bit-exact gates (`test-convert-pad-hsa`, `test-depad-hsa`, GEMM tests) must
  still pass -- existing kernels are unchanged (Phase 2 adds new bf16 instantiations).

## Verification (per the aie-kernel-opt methodology)

1. **One change at a time**: Phase 1 fully landed + measured before Phase 2 starts.
2. **Confirm the mechanism, not just wall time**: ablation / dispatch trace showing the
   per-image weight convert_pad dispatch disappears after the first dispatch (Phase 1);
   per-op NPU profile before/after (baseline ~47.8 us/image).
3. **Eligibility fallback**: any MUL_MAT failing the padded-bf16 predicate (batched,
   permuted, non-contiguous) cleanly falls back to the existing f32 path.

## Risks

- **Phase 1 stale cache**: guarded by keying the latch on the source `data` pointer.
- **Phase 2 numeric drift**: accepted, gated on accuracy only.
- **Phase 2 new kernels**: bf16 pad/depad are net-new; must pass their own bit-exact
  gates (shape-only, so a straightforward extension of the f32 gates).
- **Unverified assumptions (open items to close in the plan)**: built `matmul_bf16_bf16`
  object exists; RELU aligned bf16 store correct.

## Out of scope

- A2 (padding-aware ADD/RELU to remove pad/depad entirely).
- Any change to the ggml graph, the CPU backend, training, or the CNN MNIST path.
- SUB/MUL/DIV binary ops and general 4D broadcast (still scalar; unrelated).
- `graph_optimize` implementation (established above to be the wrong vehicle here).
