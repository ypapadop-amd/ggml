# RELU tile-granularity design

**Date:** 2026-07-15
**Status:** design, pending review
**Scope:** RELU only (port to other unary/elementwise ops is a separate follow-up)

## Goal

Cut the per-dispatch call count of the ggml-hsa RELU IRON design by streaming far
fewer, larger objectfifo tiles, amortizing the per-call acquire/release overhead that
currently dominates RELU's device time.

## Problem

MNIST RELU processes N = 250000 f32 elements per dispatch. The tile size comes from
`max_tile_size` (`utils.py`), which returns the largest **power-of-two** tile that divides
N. Since 250000 = 2^4 * 5^6, the largest power-of-2 divisor is **16** — exactly the vector
width V. Consequences:

- The worker loops `num_tiles = 250000 / 16 = 15625` times per dispatch, each iteration an
  `acquire(1) / function(...) / release(1)` cycle on a 16-element tile.
- The kernel body is one 512-bit `load / max / store` (~5 ns), but measured cost is
  ~388 ns/call — **~98% is per-call overhead** (verified earlier via `.o` inspection and
  arithmetic: 6059 us/dispatch / 15625 calls).

The pow2 restriction is the artificial ceiling: 250000 has multiple-of-V divisors
(80, 400, 2000, 10000, 50000) that would cut the call count 5x-3125x while keeping every
tile a whole number of vectors.

## Non-goals

- Changing the RELU kernel's compute (`ggml_unary_op_relu` in `unary_ops.cc`) — it is
  already optimally vectorized (one vector op, no `__divsi3`).
- Multi-core / worker fan-out for RELU (separate lever).
- Porting to abs/sgn/neg/sqr/step/... or to binary_ops/clamp/scale/count_equal (separate
  follow-up; this design is structured to make that port a gate removal).

## Constraints (hard)

1. **32 unique functions per queue.** The device queue admits at most 32 distinct compiled
   kernels. The design MUST NOT mint a new kernel per tile size. → Tile size stays a
   **runtime `N` argument** to the kernel (not a compile-time define / not part of the
   kernel name), so all RELU calls of all shapes share one kernel identity per dtype pair.
   This is the reason the remainder tail (below) reuses the same kernel rather than
   compiling a second specialization.
2. **L1 data memory per compute tile.** aie2 and aie2p core tiles have 64 KB data memory,
   shared by the in + out fifos (double-buffered), the worker stack, and kernel locals. The
   fifo footprint must fit with margin.
3. **DMA stays linear.** `rt.fill` / `rt.drain` move the whole tensor contiguously; no
   strided 2D shim descriptors (avoids the known BD wrap-size limit — see the DMA
   strided-transfer note). Larger tiles = fewer, larger linear objects; still linear.
4. **Bit-exact correctness.** RELU output must be unchanged: op test 3/3 at 5e-4, MNIST
   accuracy 98.00%, loss bit-identical (0.066372).

## Architecture

### Arch-parameter dict (utils.py)

Arch-tied constants move into one dict so adding an NPU generation is a single entry:

```python
# Per-architecture on-tile resources. Add a new NPU generation by adding one entry.
_ARCH_PARAMS = {
    "aie2":  {"core_data_mem_bytes": 64 * 1024, "vector_reg_bits": 512},  # NPU1/Phoenix (AIE-ML)
    "aie2p": {"core_data_mem_bytes": 64 * 1024, "vector_reg_bits": 512},  # NPU2/Strix (XDNA2)
}
```

`max_tile_size` (utils.py) currently hardcodes `vector_register_width = 512` inside its own
`if arch in {"aie2", "aie2p"}`. Route it through `_ARCH_PARAMS[arch]["vector_reg_bits"]` so
there is one source of truth. This is a value-preserving refactor (same 512), but it touches
a function shared by binary_ops, clamp, scale, count_equal, so each is re-verified
independently (see Testing). An unknown arch raises `ValueError` (as today).

### Tile-size selector (utils.py)

```python
def tiled_tile_size(arch, dtype, num_elements):
    """Largest multiple-of-V tile whose in+out double-buffered fifos fit ~half the
    core data memory, capped at num_elements. V = vector width in elements."""
    params = _ARCH_PARAMS[arch]                      # ValueError on unknown arch
    V = params["vector_reg_bits"] // (8 * dtype.itemsize)
    budget = params["core_data_mem_bytes"] // 2      # half DM: leave room for stack + locals
    # in + out fifos, each double-buffered (depth 2) => 4 buffers of tile*itemsize bytes.
    max_by_mem = (budget // (4 * dtype.itemsize) // V) * V
    max_by_n = (num_elements // V) * V
    return max(V, min(max_by_mem, max_by_n))
```

For aie2 f32: V=16, budget=32768, `max_by_mem = (32768//16//16)*16 = 2048`. So tile = 2048
(for N >= 2048). This is the fraction-of-DM approach: a future arch with a larger tile DM
automatically gets larger tiles without editing the selector.

### Tiled dataflow (`_unary_op_tiled` in unary_ops.py)

A new function parallel to the existing `_unary_op`; `unary_op()` routes only
`op_name == "GGML_UNARY_OP_RELU"` to it, all other unary ops stay on `_unary_op` unchanged.

- `tile = tiled_tile_size(arch, dtype, N)`; `n_full = N // tile`; `rem = N % tile`.
- Fifos `of_in` / `of_out` sized at `tile` elements (default depth => double-buffered).
- Worker core loop:
  ```
  for _ in range_(n_full):
      ein = of_in.acquire(1); eout = of_out.acquire(1)
      function(ein, eout, tile)          # runtime N = tile
      of_in.release(1); of_out.release(1)
  # remainder (Python-level if, unrolled at build time; 0 or 1 extra call)
  if rem:
      ein = of_in.acquire(1); eout = of_out.acquire(1)
      function(ein, eout, rem)           # runtime N = rem; kernel touches only rem elems
      of_in.release(1); of_out.release(1)
  ```
- `rt.fill(of_in.prod(), a_in)` / `rt.drain(of_out.cons(), b_out, wait=True)` over the full
  N elements. With a tile that does not divide N, the fill/drain still move N elements
  linearly across `n_full + 1` objects; the last object's valid region is `rem` elements
  and the buffer is otherwise unused. (fill/drain move by element count, contiguous.)

For MNIST: tile=2048, n_full=122, rem=44 => **123 calls/dispatch** (was 15625, ~127x fewer).
The rem=44 call exercises the kernel's own `vend=(N/V)*V` interior (32) + scalar tail (12).

### Kernel (unary_ops.cc) — no change needed

`ggml_unary_op_relu` already has a runtime-N path (the `#else` of the `GGML_TILE_SIZE`
fold). The tiled design does **not** pass `-DGGML_TILE_SIZE`, so the runtime-N branch is
used — it handles both full tiles (N=2048) and the remainder (N=44) correctly, including the
sub-V scalar tail. The compile-time fold stays available for the non-tiled unary path.

## Data flow

```
DRAM src (N f32, contiguous)
  -> rt.fill -> of_in (tile-sized, depth 2) --.
                                              acquire(1)
                                     ggml_unary_op_relu(in, out, N=tile or rem)
                                              release(1)
  of_out (tile-sized, depth 2) -> rt.drain -> DRAM dst (N f32, contiguous)
```

n_full full-tile iterations + at most one remainder iteration per dispatch.

## Testing / verification (each independent)

**RELU (the feature):**
- `test-backend-ops-mnist test -o RELU` => 3/3 at 5e-4 (covers 3 shapes incl. 14x14x16x500).
- MNIST eval: accuracy 98.00%, loss bit-identical 0.066372.
- e2e us/image before/after (median of 5); expect a drop from ~51.5.
- Per-op profiler: RELU share drops from ~11.5%.
- Confirm no new kernels: RELU still one compiled kernel per shape (32-function budget
  unaffected) — check the JIT log's "generated kernel" lines.

**max_tile_size refactor (value-preserving) — verify each consumer independently.**
The `test-backend-ops-mnist` harness registers cases for RELU (test_unary), ADD/SUB/MUL/DIV
(test_bin_bcast), and MUL_MAT (test_mul_mat) — these directly exercise the max_tile_size
consumers unary_ops and binary_ops. Run each op filter separately so a regression is
attributable to one consumer:
- `test-backend-ops-mnist test -o RELU` => 3/3
- `test-backend-ops-mnist test -o ADD` => pass (binary_ops)
- `test-backend-ops-mnist test -o MUL_MAT` => 3/3
The remaining max_tile_size consumers — clamp, scale, count_equal — have no case in this
MNIST harness. Verify them via the standalone `test-backend-ops` binary
(`test-backend-ops test -o SCALE` / `-o CLAMP` / `-o COUNT_EQUAL`, filtered to the HSA0
backend). If a consumer has no runnable HSA test at all, note that in the plan rather than
claim a gate that does not exist. Because the refactor is value-preserving (same 512), a
pass everywhere confirms no behavior change.

## Risks & mitigations

- **Fifo depth-2 at 2048 must fit L1.** Budgeted at half DM (32 KB of 64). If aiecc reports
  memory pressure, halve the budget fraction (=> tile 1024, 245 calls, still ~64x fewer).
- **Remainder call on a partially-filled buffer.** The op test's non-pow2 shapes exercise
  the sub-tile path; the kernel's existing scalar tail handles rem % V != 0.
- **32-function limit.** Explicitly protected: tile size is a runtime arg, so no per-tile
  kernel proliferation. Noted so the later port keeps this property.
- **Refactor blast radius.** `max_tile_size` change is same-value; independent per-op tests
  above catch any accidental behavior change.

## Follow-up (out of scope)

- Port `_unary_op_tiled` to the other unary ops (drop the RELU gate).
- Consider the same granularity lever for binary_ops (ADD), clamp, scale.
