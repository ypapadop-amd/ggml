# bf16 dataflow for ggml-hsa MUL_MAT chains (dtype-collapse)

Date: 2026-07-16
Status: approved (design), pending implementation plan

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

1. **Constant weights are re-converted every dispatch.** `convert_dtype = true` is set
   on *both* MUL_MAT sources (`ggml-hsa.cpp:691, 699`). The weight
   (`[784 x 500]` for fc1) dwarfs the single-image activation (`[784 x 1]`) and never
   changes, yet it is converted+padded once per image (10,000x over the eval set).
2. **The activation round-trip** (depad -> f32 -> ADD/RELU -> convert_pad -> bf16)
   between the GEMMs.

## Goal

Eliminate all mid-graph f32<->bf16 **dtype** conversions in f32 MUL_MAT chains on the
ggml-hsa backend. After this change, the only dtype conversions are:

- **image in**: f32 -> bf16, once at the chain entry;
- **logits out**: bf16 -> f32, once at the chain exit (logits are asserted f32 at
  `mnist-common.cpp:378`).

Everything between (MM1 output, ADD, RELU, MM2 input/output, final ADD) stays bf16.

Scope decision (**A1, dtype-collapse**): the pad/depad **shape** steps are retained but
become bf16 -> bf16 shuffles (no dtype math, half the bytes). Making ADD/RELU
padding-aware to remove pad/depad entirely (A2) is explicitly **out of scope** for this
work; the `graph_optimize` infrastructure landed here is the foundation A2 would build on.

## Acceptance gate

- **MNIST 10k FC f32 e2e**: test accuracy **>= 98.00%** (current baseline 98.00%).
- Bit-identity to the f32-intermediate baseline is **dropped** (bf16 intermediates drift
  numerically by design). Record the loss drift from the baseline 0.066372 for the record,
  but it is not a gate.
- Kernel-level bit-exact gates (`test-convert-pad-hsa`, `test-depad-hsa`, GEMM tests) must
  still pass, since the compute kernels themselves are unchanged.

## Approach: backend-level, three coordinated changes

All changes are internal to `src/ggml-hsa/ggml-hsa.cpp`. The ggml graph in
`mnist-common.cpp` stays f32; the CPU backend and training paths are untouched. This
generalizes to any contiguous non-batched f32 MUL_MAT chain, not just MNIST.

### 1. Implement `graph_optimize` (the vehicle)

`graph_optimize` is currently `nullptr` (`ggml-hsa.cpp:1732`); its signature is
`void(ggml_backend_t, ggml_cgraph *)` (`src/ggml-backend-impl.h:139`). Per-node
`tensor_extra` is built in isolation at buffer-init (`ggml-hsa.cpp:1051`) where a node can
see its *sources* but not its *consumers*, so a per-node rewrite cannot decide "emit bf16
because a GEMM consumes me." A whole-graph pass can.

The pass walks the graph, identifies f32 MUL_MAT chains eligible for the padded bf16 GEMM
path (reusing the predicate in `ggml_hsa_prepare_mul_mat_f32`), and records, per edge,
whether it stays bf16. This informs the per-node rewrite that follows.

Open item for the plan: confirm the lifecycle ordering between `graph_optimize` and
`tensor_extra` construction, and decide where the cross-node decisions are stored so the
per-node path can read them.

### 2. Cache converted+padded bf16 weights (dominant win, numerically free)

A MUL_MAT weight operand is a graph leaf (`op == GGML_OP_NONE`) with resident data and is
not the graph input tensor -- i.e. a constant. Convert+pad it to bf16 **once** and cache the
resulting bf16 buffer (on its `tensor_extra` / buffer context), rather than re-running
`convert_pad` on every dispatch. The cached bits are identical every time, so this is a
pure throughput win with **zero numeric change**.

Open item for the plan: buffer lifetime -- the cached bf16 weight must outlive all 10k
dispatches; confirm the owning allocation and that it is not reclaimed between graph runs.

### 3. Thread bf16 through the intermediates

- GEMM emits `dtype_out = bf16` (already supported: `gemm.py:70`).
- The activation operand of MM1 converts f32 -> bf16 once at entry.
- The bf16 GEMM output flows through ADD and RELU, both already dtype-generic via
  `-DINPUT_DTYPE`/`-DOUTPUT_DTYPE` (`binary_ops.py:169`, `unary_ops.py:144-145`) and
  their bf16 bias/input operands.
- Pad/depad are retained but bf16 -> bf16 (no dtype conversion).
- The final logits are depadded and converted bf16 -> f32 for the f32 output tensor.

## Components and data flow (after)

```
images(f32)
  --[convert f32->bf16 + pad, once at entry]--> A0(bf16, padded)
fc1_w(f32 const) --[convert+pad ONCE, cached]--> W1(bf16, padded)
  GEMM(W1, A0) -> C1(bf16, padded)
  --[depad bf16->bf16]--> h1(bf16, dense)
  ADD(h1, fc1_bias bf16) -> RELU -> a1(bf16, dense)
  --[pad bf16->bf16]--> A1(bf16, padded)
fc2_w(f32 const) --[convert+pad ONCE, cached]--> W2(bf16, padded)
  GEMM(W2, A1) -> C2(bf16, padded)
  --[depad bf16->bf16]--> h2(bf16, dense)
  ADD(h2, fc2_bias bf16)
  --[convert bf16->f32, once at exit]--> logits(f32)
```

## Testing / verification

1. **Kernel gates unchanged**: `test-convert-pad-hsa`, `test-depad-hsa`, GEMM tests still
   pass (compute kernels are not modified).
2. **e2e accuracy gate**: run the MNIST 10k FC f32 eval on HSA0; assert acc >= 98.00%;
   record loss drift from 0.066372.
3. **Confirm the mechanism**, not just the wall time (per the aie-kernel-opt methodology):
   - ablation / dispatch trace showing the per-image weight `convert_pad` dispatch is gone;
   - per-op NPU profile before/after to attribute the win (baseline ~47.8 us/image).
4. **One change at a time**: land the weight cache first (safe, numerically free, big win),
   measure; then bf16 intermediates; measure again.

## Risks

- **Lifecycle ordering**: `graph_optimize` vs `tensor_extra` init -- must confirm the pass
  runs early enough to influence per-node rewrite and buffer allocation.
- **Numeric drift**: bf16 intermediates change logits; accepted, gated on accuracy only.
- **Weight-cache lifetime**: cached bf16 buffers must persist across all dispatches and not
  be double-freed or reclaimed between graph runs.
- **Eligibility fallback**: any MUL_MAT that fails the padded-bf16 predicate (batched,
  permuted, non-contiguous) must cleanly fall back to the existing f32 path.

## Out of scope

- A2 (padding-aware ADD/RELU to remove pad/depad entirely).
- Any change to the ggml graph, the CPU backend, training, or the CNN MNIST path.
- SUB/MUL/DIV binary ops and general 4D broadcast (still scalar; unrelated).
