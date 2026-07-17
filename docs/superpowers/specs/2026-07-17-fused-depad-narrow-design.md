# Fused MUL_MAT depad + f32→bf16 narrow (ggml-hsa)

## Problem

In the bf16 MNIST FC graph on the ggml-hsa NPU backend, every layer is:

```
... -> MUL_MAT (bf16 x bf16, f32-accumulate) -> f32 result
     -> CPY (f32 -> bf16 cast)               -> bf16
     -> ADD/RELU (bf16) -> ...
```

`ggml_mul_mat` hardcodes an f32 destination, so the padded GEMM path writes an
f32 padded temporary, HSA_DEPAD strips the padding into the f32 parent tensor,
and then a separate convert-CPY re-reads that f32 parent and writes a bf16 copy
for the next op to consume.

Both the depad and the CPY already run on the device queue (the CPY via
`convert_copy_kernel` / HSA_CONVERT), so this is **not** a queue-drain cost. It is
one extra kernel dispatch per layer plus a full `[M, N]` f32 round trip through
memory: depad writes f32, HSA_CONVERT immediately re-reads that same f32 and
writes bf16. This is the last structural difference between the bf16 and f32
paths and the bulk of the residual ~1.4x gap.

## Approach

Fuse the f32→bf16 narrow into the depad post-amble. The depad already streams the
`[M, N]` result row by row; narrowing each element as it is written costs nothing
extra and eliminates the separate CPY dispatch and its memory round trip.

The graph stays idiom-legal — the ggml `mul_mat` tensor keeps its f32 type and
the `ggml_cast`/CPY node stays in the graph. The backend recognizes the
`MUL_MAT -> convert-CPY(f32->bf16)` pair and redirects the depad to write bf16
straight into the CPY's output buffer, then skips the CPY dispatch. No tensor
type is mutated; nothing outside the backend changes.

This is done as a **graph pre-pass** rather than at `buffer_init_tensor` time
because `tensor_extra` is built per node and sees a node's sources, never its
consumers. Only a whole-graph scan can see that a given MUL_MAT is consumed by a
convert-CPY. The pre-pass runs in `graph_compute` after all `tensor_extra`
objects exist, so there is no node-ordering assumption.

## Trigger (graph pre-pass)

Runs once per `graph_compute`, in the existing node loop that fixes up source
pointers (ggml-hsa.cpp ~1519), or a sibling loop right after it. Guarded by a
`fusion_analyzed` latch on the MUL_MAT extra so the detection + kernel lookup do
not repeat every forward pass.

For each node `mm`, fuse only if ALL hold:

1. `mm->op == GGML_OP_MUL_MAT`.
2. `mm->extra` has `node.depad == true` and `postprocess_kernel != nullptr`
   (it is on the padded-GEMM device-transform path).
3. `mm` is not a graph output: `(mm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0`.
4. `mm` has **exactly one consumer** in the cgraph. Consumers are found by an
   O(N^2) scan over `ggml_graph_n_nodes` (graphs here are ~7 nodes). A node `c`
   is a consumer if any `c->src[j] == mm`.
5. That sole consumer `c` is a convert-CPY with an f32 source and bf16 dst:
   `ggml_hsa_is_convert_copy(*c)` and `c->src[0]->type == GGML_TYPE_F32` and
   `c->type == GGML_TYPE_BF16`.

When all hold, build the fused depad kernel:

```cpp
auto fused = ggml_hsa_build_transform_kernel(
    dev_info, "HSA_DEPAD", mm_extra.node.tensor /* padded f32 in */, *c /* bf16 dst */);
```

The distinct (f32 in, bf16 out) dtype pair produces its own cached PDI, separate
from the plain f32→f32 depad.

On success:
- `mm_extra.postprocess_kernel = fused;`
- `mm_extra.fused_narrow_dst = c;`
- `c_extra.skip_dispatch = true;`

`fusion_analyzed` is set regardless of success. If the kernel fails to build, no
fields change and the graph runs exactly as today (separate depad + CPY).

## tensor_extra fields (common.hpp)

Three new members on `ggml_backend_hsa_tensor_extra`:

```cpp
/// When set (fused MUL_MAT depad+narrow), the post-processing kernel writes the
/// narrowed bf16 result into this consumer tensor instead of the f32 parent.
ggml_tensor * fused_narrow_dst = nullptr;
/// When true (a convert-CPY fused into its producer's depad), graph_compute
/// skips this node entirely — the producer already wrote its bf16 output.
bool skip_dispatch = false;
/// Latches the one-time fusion pre-pass so detection does not repeat per forward pass.
bool fusion_analyzed = false;
```

## Dispatch wiring (graph_compute)

Two edits in `ggml_backend_hsa_graph_compute`:

1. **CPY/DUP case (~1544):** at the top, `if (extra.skip_dispatch) continue;`
   before the existing convert/host branching. A skipped convert-CPY must NOT
   run — its f32 source parent is never written (the producer wrote bf16 to the
   CPY dst directly), so re-reading it would clobber the correct result with
   garbage.

2. **Postprocess dispatch (~1642):** the fused kernel writes to the consumer,
   not the f32 parent:

   ```cpp
   ggml_tensor & post_dst =
       tensor_extra.fused_narrow_dst ? *tensor_extra.fused_narrow_dst : *node;
   status = tensor_extra.postprocess_kernel->dispatch(ctx, &postprocess_src, 1, post_dst);
   ```

The `use_device_transforms` gate, the preprocess loop, and the constant cache are
untouched. The fused path is only reachable when `use_device_transforms` is
already true (it requires `postprocess_kernel != nullptr`, which fusion sets).

## Kernel (depad.cc / depad.py)

Add an f32→bf16 mode to the existing depad kernel, selected by a compile flag,
alongside the current f32→f32 copy.

**depad.py:**
- Accept a bf16 `output_tensor` in addition to f32; the source is always f32
  (the padded GEMM temp).
- When `output_tensor.dtype == bfloat16`, append `-DDEPAD_CONVERT_F32_TO_BF16=1`
  and set `OUTPUT_DTYPE=bf16`; the fifo/row types follow the dtypes as they
  already do. f32→f32 path unchanged.

**depad.cc:**
- Default (`#ifndef DEPAD_CONVERT_F32_TO_BF16`): current f32 copy, unchanged.
- New mode: reuse convert.cc's exact-RNE f32→bf16 vector body (integer
  arithmetic on `aie::vector<uint32_t, V>`, `V = 512/32 = 16`), aligned
  `load_v`/`store_v` on the vector-aligned per-row fifo buffers, scalar-RNE tail
  for the `[vend, d0)` remainder. The `[d0, d0pad)` padding is simply not read
  (depad only copies the first `d0`), so there is no tail-zeroing to do.
- `AIE_LOOP_RANGE` guarded `#if defined(DEPAD_D0) && (DEPAD_D0) >= 16` exactly as
  the f32 path already is (fc2 d0=10 < V falls to the scalar tail).

Shapes exercised: fc1 output row d0 = 128 = 8*V (fully vectorized); fc2 output
row d0 = 10 < V (all scalar tail). Both must be bit-identical to the current
two-step depad-then-CPY.

## Numerics

The narrow applies the same round-to-nearest-even the CPY (HSA_CONVERT) applied,
one pass earlier and against the same f32 accumulate values. Result is
bit-identical to the current two-step path, so accuracy stays 98.00% / loss
~0.0663. Gate: `mnist-eval-bf16` accuracy >= 98.00%.

## Safety / fallback

- Sole-consumer + not-a-graph-output guards guarantee nothing else reads the f32
  parent that fusion leaves unwritten.
- Kernel-build failure leaves all fields default → today's separate depad + CPY.
- `dispatch(..., *fused_narrow_dst)` resolves the consumer's `data` pointer at
  compute time; it is a real allocated graph tensor.
- f32 (non-bf16) MNIST path never matches guard 5 (dst is f32), so it is wholly
  unaffected.

## Verification

1. Build; move any stale `~/.cache/ggml/aie2/*depad*` PDIs aside (mv, not rm).
2. `mnist-eval-bf16`: accuracy >= 98.00%, record us/image (expect a drop from
   ~67 toward the f32 ~47.6).
3. `mnist-eval` (f32): unchanged (bit-identical loss, timing).
4. Depad kernel gate (f32→f32 and the new f32→bf16) passes.
5. Confirm the CPY dispatch count drops (the fused layers no longer dispatch
   HSA_CONVERT) — one-line instrumentation or trace.
```
