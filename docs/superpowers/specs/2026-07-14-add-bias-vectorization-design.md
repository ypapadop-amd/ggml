# Vectorized ADD bias path (co-design `.py` + `.cc`)

Status: **approved design**, pre-implementation. Date: 2026-07-14.

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

Per user decision: **ADD entry points only.** Only `ggml_op_add` / the ADD broadcast dispatch gains
the optimized path. SUB / MUL / DIV and every non-row-broadcast case keep calling the existing scalar
templates unchanged. Multi-core is a **separate follow-up phase**, out of scope here (this pass is
single-core vectorization).

## Architecture & dispatch

Add a **new vectorized bias entry point** in `binary_ops.cc`, selected in `binary_op()` (Python)
**only when** all of:

1. `op_name == "GGML_OP_ADD"`, and
2. src1 is a single row broadcast over rows: `src1_ne0 == dst_ne0` and
   `src1_ne1 == src1_ne2 == src1_ne3 == 1` and `dst_ne0 == src0_ne0`, and
3. a clean "replicated super-row" tiling exists (predicate below).

```
binary_op(op_name, ...)
├─ ADD AND single-row-bias AND super-row tiling exists → NEW vectorized bias design (.cc + .py)
└─ else (SUB/MUL/DIV, general 4D broadcast, elementwise) → existing scalar path (UNTOUCHED)
```

Anything failing the predicate (including fc2, SUB/MUL/DIV, true multi-dim broadcasts) falls to the
existing `transform_binary_broadcast_n` / `transform_binary_n`. No behavior change for those.

## The "replicated super-row" tiling

To keep the kernel hot loop a clean, tail-free vector add even when `ne0` is not a multiple of the
vector width `V`, replicate `src1` and tile `src0`/`out` on the least common multiple:

- `V = 512 / (8 * sizeof(dtype))` — 16 for f32.
- `g = gcd(ne0, V)`, `R = V / g` (rows after which the src1 phase realigns to a vector boundary).
- `L = R * ne0 = lcm(ne0, V)` — the super-row tile size (a multiple of `V`).
- Pre-replicate `src1` `R` times into an `L`-element buffer, loaded **once** (`ObjectFifo depth=1`).
- Stream `src0`/`out` in tiles of `L`. Every tile aligns to the same replicated src1 buffer, so all
  src1 vector loads are aligned and the loop is a pure `aie::add` with **zero scalar tail**.

**Tiling-existence predicate:** `L` must divide the padded total, i.e. `num_rows % R == 0`
(equivalently `total_elements % L == 0`).

- **fc1** `[500, 500]`: `g = gcd(500,16) = 4`, `R = 4`, `L = 2000`. `num_rows = 500`,
  `500 % 4 == 0` ✓ → **125 tiles of 2000, zero tail, pure `aie::add`.** This is the 98%-of-cost case.
- **fc2** `[10, 500]`: `g = gcd(10,16) = 2`, `R = 8`, `L = 80`. `num_rows = 500`,
  `500 % 8 != 0` ✗ → **no clean super-row tiling; falls back to the existing scalar path.**

fc2 falling back is acceptable and explicitly accepted: it is ~2% of ADD volume and already cheap.
This pass targets the fc1 bias that dominates.

## Kernel hot loop (follows in-tree `scale.cc` convention)

```cpp
void ggml_op_add_bias(const T * __restrict src0,   // L-element tile
                      const T * __restrict src1,   // L-element pre-replicated row (loaded once)
                      T * __restrict out,          // L-element tile
                      int32_t L) {
    event0();
    constexpr int32_t V = 512 / (sizeof(T) * 8);
    // L is a multiple of V by construction (super-row tiling).
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < L; i += V) {
        aie::vector<T, V> a = aie::load_v<V>(src0 + i);
        aie::vector<T, V> b = aie::load_v<V>(src1 + i);
        aie::store_v(out + i, aie::add(a, b));
    }
    event1();
}
```

- `__restrict` on every pointer (pipelining lever), `constexpr V` (folds to shifts),
  `AIE_PREPARE_FOR_PIPELINING` + `AIE_LOOP_MIN_ITERATION_COUNT` (Peano pipelining).
- No `__divsi3`, no per-element index math, no scalar tail (L % V == 0 by construction).
- `#include "aie_kernel_utils.h"` (not currently included by `binary_ops.cc`).
- Preserve numerics: f32 add is exact vs the scalar path; correctness gate is the existing 1e-7 /
  bf16 1e-4 tolerance in `test-backend-ops-mnist`.

The exact entry-point name and whether it is a new `#ifdef GGML_OP_ADD_BIAS` block vs. a variant of
the existing `ggml_op_add_broadcast` is an implementation detail for the plan; the Python
`ExternalFunction` name/signature and the `extern "C"` symbol must match whatever the new design
dispatches.

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

- **fc2 unchanged:** accepted; ~2% of ADD volume, no clean super-row tiling.
- **Replicated src1 buffer size:** `L` elements held on-tile (fc1: 2000 f32 = 8 KB, well within the
  ~64 KB L1). Confirm for any larger future ne0 before reuse.
- **DMA linearity:** src1 replication is a linear fill of a contiguous buffer — respects the known
  strided-DMA limit (fill/drain stay linear).
