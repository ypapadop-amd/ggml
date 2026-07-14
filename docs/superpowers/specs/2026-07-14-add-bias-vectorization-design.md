# Vectorized ADD bias path (co-design `.py` + `.cc`)

Status: **approved design**, pre-implementation. Date: 2026-07-14. Tiling revised from "replicated
super-row" to "plain row-tiling" during planning (the super-row needed an unproven on-core scratch
replication with DMA risk; row-tiling reaches ~99.2% vectorization on fc1 via the proven `scale.cc`
pattern with no replication).

## Motivation

Per `docs/ggml-hsa-optimization-tracking.md` §4, **ADD dominates the MNIST FC eval at ~55% of
serialized dispatch time** (~59 ms/dispatch, ~9× the GEMM it feeds). The expensive ADDs are the two
**bias adds** — a broadcast of a bias row over the batch dimension.

Both MNIST bias adds have the same structure:

```
out[row, i] = src0[row, i] + src1[i]
```

where `src1` is a **single contiguous row** (`src1_ne1 == src1_ne2 == src1_ne3 == 1`,
`src1_ne0 == dst_ne0`) repeated down every row of `dst`.

- **fc1 bias:** dst `[500, 500]`, src1 `[500]` → 250,000 elements (~98% of ADD volume).
- **fc2 bias:** dst `[10, 500]`, src1 `[10]` → 5,000 elements (~2% of ADD volume).

The current kernel routes these through `transform_binary_broadcast_n` (`binary_ops.cc`), a **fully
scalar** loop that computes a 4D coordinate decomposition **per element**: 3 signed `/` + 4 signed
`%` against runtime `int` args. Signed power-of-two divide/modulo lower to `__divsi3` external calls,
which (a) cost cycles directly and (b) act as a **vectorization barrier for the whole function** — so
the loop never vectorizes. That is the root cause of ADD being ~9× the GEMM.

For this broadcast pattern the entire index computation collapses to `src1[i % ne0]`, and because
`src1` is one repeated row, no genuine gather is needed at all.

## Scope

Per user decision: **ADD entry points only.** Only the ADD single-row-bias broadcast gains the
optimized path. SUB / MUL / DIV and every non-single-row-broadcast case keep calling the existing
scalar templates unchanged. Multi-core is a **separate follow-up phase**, out of scope here (this
pass is single-core vectorization).

## Architecture & dispatch

Add a **new bias entry point** in `binary_ops.cc`, selected in `binary_op()` (Python) **only when**
all of:

1. `op_name == "GGML_OP_ADD"`, and
2. src1 is a single row broadcast over rows: `src1_ne0 == dst_ne0` and
   `src1_ne1 == src1_ne2 == src1_ne3 == 1` and `src0.shape == dst.shape`.

```
binary_op(op_name, ...)
├─ ADD AND single-row-bias  → NEW row-tiled bias design (.cc + .py)
└─ else (SUB/MUL/DIV, general 4D broadcast, elementwise) → existing scalar path (UNTOUCHED)
```

Anything failing the predicate (SUB/MUL/DIV, true multi-dim broadcasts, plain element-wise) falls to
the existing `transform_binary_broadcast_n` / `transform_binary_n`. No behavior change for those.

Both MNIST bias adds match the predicate, so **both** use the new path — fc2 included (it runs the
same kernel; it just doesn't vectorize because its row is shorter than one vector — see below).

## The row-tiling scheme

Tile on **one dst row** so `src1` (one contiguous row, `ne0` elements) is loaded **once** and reused
for every row — no replication, no strided DMA (avoids the known shim-DMA strided-transfer limit that
silently drops data), no per-element index math.

- `tile_size = ne0` (one full dst row). One `ObjectFifo` object == one row.
- `num_tiles = num_rows = total_elements / ne0` (always integer; `total = num_rows * ne0`, and f32
  arch-alignment is a no-op so `num_elements == numel` exactly).
- Three fifos: `src0` (tile = ne0), `out` (tile = ne0), and `src1` (`depth=1`, full ne0 row, acquired
  **once** outside the tile loop — mirrors the existing broadcast path's src1 handling).
- The kernel adds one row per call: `out[i] = src0[i] + src1[i]` for `i in [0, ne0)`, vectorized over
  a `V`-wide interior with a scalar tail (the `scale.cc` idiom).

**Why the tail is cheap:** `V = 16` for f32. `vend = (ne0 / V) * V` is the vector interior; `[vend,
ne0)` is a scalar tail. Because the object is exactly `ne0` long, `aie::load_v<V>` at any interior
offset `i` (with `i + V <= vend <= ne0`) never reads past the object boundary — this sidesteps the
known `load_v` row-boundary over-read gotcha.

- **fc1** `ne0 = 500`: `vend = 496`, tail = 4 → **99.2% of elements vectorized, 0.8% scalar.** This
  is the ~98%-of-ADD-cost case.
- **fc2** `ne0 = 10 < V`: `vend = 0` → entire row scalar. Acceptable: ~2% of ADD volume, and still
  **strictly better than today** because the new kernel has **no `__divsi3`** (a plain row add), vs.
  the current 7 signed div/mod per element.

## Kernel loop (follows in-tree `scale.cc` convention)

```cpp
void ggml_op_add_bias(const T * __restrict src0,   // one row: ne0 elements
                      const T * __restrict src1,   // one row: ne0 elements (loaded once, reused)
                      T * __restrict out,           // one row: ne0 elements
                      int32_t N) {                  // N == ne0
    event0();
    constexpr int32_t V = 512 / (sizeof(T) * 8);
    const int32_t vend = (N / V) * V;               // division by CONSTEXPR V → inline shift, once (not __divsi3)

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<T, V> a = aie::load_v<V>(src0 + i);
        aie::vector<T, V> b = aie::load_v<V>(src1 + i);
        aie::store_v(out + i, aie::add(a, b));
    }

    for (int32_t i = vend; i < N; ++i) {
        out[i] = src0[i] + src1[i];
    }
    event1();
}
```

- `__restrict` on every pointer (pipelining lever), `constexpr V` (folds to shifts),
  `AIE_PREPARE_FOR_PIPELINING` + `AIE_LOOP_MIN_ITERATION_COUNT` (Peano pipelining).
- **No `__divsi3` on any path:** the only division is `N / V` by a compile-time-constant `V`, once,
  outside the loop — the compiler lowers constant-pow2 division to an inline shift sequence, not a
  libcall. (The current kernel's `__divsi3` storm was division by *runtime* `src1_ne`/`dst_ne` values,
  per element.)
- `#include "aie_kernel_utils.h"` (not currently included by `binary_ops.cc`).
- Preserve numerics: f32 add is exact vs the scalar path; correctness gate is the existing 1e-7 /
  bf16 1e-4 tolerance in `test-backend-ops-mnist`.

The exact `extern "C"` symbol is `ggml_op_add_bias`, gated by a new `#ifdef GGML_OP_ADD_BIAS` block;
the Python `ExternalFunction` name/signature must match it.

## Verification (full profile)

1. **Correctness gate:** `test-backend-ops-mnist -o ADD` must stay **3/3 OK** (1e-7 / bf16 1e-4).
2. **Isolated ADD win:** temporarily re-add the `GGML_HSA_PROFILE_DISPATCH` §4 per-dispatch probe;
   measure ADD µs/dispatch before/after; **revert the probe** after measuring.
3. **End-to-end:** MNIST §1 benchmark before/after for net wall-clock and 98.00% accuracy
   (must not regress).
4. **Cache hygiene:** move stale ADD kernel artifacts out of `~/.cache/ggml/aie2/` (`mv`, not `rm` —
   `rm` is blocked in this env) so the new kernel recompiles.

## Explicitly NOT touched

- `transform_binary_n` (elementwise) and `transform_binary_broadcast_n` (general 4D broadcast).
- SUB / MUL / DIV entry points and their (element-wise + broadcast) paths.
- The Triton ADD fallback.
- Multi-core fan-out (the designated next optimization phase).

## Risks / open items

- **fc2 not vectorized:** accepted; ~2% of ADD volume, `ne0=10 < V`. Still `__divsi3`-free, so still
  faster than the current scalar broadcast path.
- **DMA linearity:** all three fifos stream contiguous rows (one object == one `ne0` row); fill and
  drain stay linear, respecting the known strided-DMA limit.
- **On-tile footprint:** three `ne0`-element objects (fc1: 3 × 500 f32 = 6 KB, well within ~64 KB L1;
  fifo double-buffering at most doubles the src0/out portion). No large replicated buffer.
