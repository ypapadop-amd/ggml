# ggml-hsa: Optimization Tracking & Benchmark Log

Status: **living document**. Records profiling and benchmarking data for the HSA (AIE/NPU) backend
so optimization work has a concrete, dated baseline to measure against. Append new measurements as
sections; do not overwrite old ones (keep the history).

Machine (all numbers below unless stated otherwise): AMD Ryzen 9 7940HS w/ Radeon 780M, NPU1
(`aie2`, 4 columns). Kernels pre-compiled and cached (`~/.cache/ggml/aie2`) so timings exclude JIT
compilation. Model: MNIST FC f32 (`mnist-fc-f32.gguf`), 10,000 test images, physical batch = 500
(20 graph_compute calls per run).

## Optimization history

Per-op dispatch-overhead work, tracked as three optimizations. All three have landed:

- **#1 — batch packets per doorbell** (merged, PR #199). `ggml_hsa_aie_kernel::dispatch` enqueues a
  packet without ringing the doorbell; `ggml_hsa_flush_dispatches` rings once for the whole pending
  run (`aie-kernel.cpp` / `ggml-hsa.cpp`). Cut doorbell rings ~9× (see §2).
- **#2 — kernarg bump allocator** (merged, PR #198). `ggml_hsa_bump_allocator` in `common.hpp`
  replaced per-dispatch `hsa_amd_memory_pool_allocate` + `std::vector<unique_ptr>` tracking. No
  measurable end-to-end change (removed cost was ~0.2% of runtime); value is a cleaner hot path and
  an allocation-free foundation for #1. Landed with allocate-after-wait ordering and power-of-two
  alignment enforced at construction.
- **#3 — on-device pad/convert, keep intermediates on-device** (implemented, this branch). f32→bf16
  convert + zero-pad and result de-pad moved from host copies to on-device IRON kernels dispatched
  on the queue (`HSA_CONVERT_PAD` / `HSA_DEPAD`; `src/ggml-hsa/kernels/iron_kernels/convert_pad.*`,
  `depad.*`). Removed the per-op host sync points that previously fragmented batching. See §5 for
  the outcome — it did **not** produce the large end-to-end win originally predicted.

### Reconciliation: the doorbell was not the bottleneck

The original dispatch-overhead analysis (2026-07-12, §5 below) concluded that ~94% of wall-clock was
the blocking doorbell store, and that the lever was doorbelling less often (#1). That conclusion did
**not hold up** once #1 landed:

- #1 cut doorbell rings ~9× (180/run → 20/run, 0.002/image; §2), yet end-to-end barely moved
  (~350 → ~333 µs/image, ~5%; §1). If the doorbell were truly 94% of runtime, batching would have
  collapsed HSA0 toward the CPU baseline. It did not.
- The old breakdown's own arithmetic hints why: 3282 ms ÷ 180 dispatches ≈ **18 ms per doorbell
  store**, far above the "~24 µs driver round-trip" it assumed. The blocking store was absorbing
  **device kernel-execution time**, not pure dispatch overhead. Batching removed the per-op blocking
  but the device still executes all the work, so total runtime held.

Corrected cost model: **the bottleneck is device kernel execution, not dispatch overhead.** The
per-dispatch profile (§4) confirms this and identifies the real hot kernels (ADD dominates, convert
costs more than the GEMM). Dispatch overhead (#1) and host copies (#3) are both largely solved and
neither was the primary cost. Do not chase doorbell count further on this model.

## How to reproduce

```
source build/.venv/bin/activate
cd examples/mnist
BIN=$PWD/../../build/bin/mnist-eval
IMG=data/MNIST/raw/t10k-images-idx3-ubyte
LBL=data/MNIST/raw/t10k-labels-idx1-ubyte
"$BIN" mnist-fc-f32.gguf "$IMG" "$LBL" HSA0   # or CPU
```

Instrumentation used to gather the data below was **temporary and env-gated**, reverted after
measuring (not in the tree). To re-measure, re-add the specific probe described in each section.

---

## 1. End-to-end: CPU vs NPU (2026-07-14)

MNIST FC f32, 10k images, on-device convert/pad + de-pad transforms active.

| Metric | CPU | NPU (HSA0 / aie2) |
| --- | ---: | ---: |
| Accuracy | 98.01% ±0.14 | 98.00% ±0.14 |
| Test loss | 0.066352 ±0.008763 | 0.066372 ±0.008765 |
| Latency | ~30 µs/image | ~333 µs/image |
| Total (10k) | ~302 ms | ~3317–3350 ms |

Stability (repeat runs, µs/image): CPU ~30–34; NPU ~332–335.

Takeaways:
- Correctness matches: NPU accuracy within noise of CPU; loss differs only at the 5th decimal
  (expected bf16 rounding of the matmul operands).
- NPU is ~11× slower on this model. Expected: MNIST FC is tiny, so wall-clock is dominated by
  host-side per-op dispatch latency and device execution of small kernels, not AIE compute
  throughput. This is a small-graph latency floor, not a compute-throughput limit; it would invert
  on large models where AIE compute dominates.

---

## 1b. End-to-end re-benchmark (2026-07-14, confirmation run)

Repeat of §1 later the same day to confirm the baseline is stable and reproducible. Same machine,
model, batch, and on-device convert/pad + de-pad transforms active. MUL_MAT prerequisite re-checked
first: **3/3 OK at strict 5e-4**.

| Metric | CPU | NPU (HSA0 / aie2) |
| --- | ---: | ---: |
| Accuracy | 98.01% ±0.14 | 98.00% ±0.14 |
| Test loss | 0.066352 ±0.008763 | 0.066372 ±0.008765 |
| Latency | ~29.2 µs/image | ~333.1 µs/image |
| Total (10k) | ~291–294 ms | ~3330 ms |

Stability (repeat runs, µs/image): CPU 29.13 / 29.35 / 29.13 (3 runs); NPU 333.04 / 333.05 / 333.09 /
333.10 (4 clean runs). NPU is now noticeably *tighter* than the §1 spread (~332–335) — steady to the
0.1 µs. NPU ~11.4× slower than CPU, unchanged.

Note: the first cold run of the session read 854 µs/image and dropped the accuracy/loss lines — the
known intermittent cold-cache JIT hitch (see the on-device convert/pad memory note); discarded as a
warmup outlier. All subsequent warm runs were steady. Numbers match §1 within noise; no regression.

---

## 7. Post-vectorization re-benchmark + corrected CPU baseline (2026-07-15)

After vectorizing all inference-relevant AIE kernels (ADD §6, then RELU, depad, convert_pad — the
last three this session), and — critically — **rebuilding the host with `-DCMAKE_BUILD_TYPE=Release`
and OpenMP enabled**. The §1/§1b CPU numbers (~30 µs/image) were taken from a **Debug (`-O0`),
effectively single-threaded** build and are NOT a fair CPU baseline. Corrected below.

Build note (OpenMP with clang): the host compiler is clang-22, which ships no bundled `omp.h` and no
`-lomp` dev symlink; only llvm-20's libomp is installed. Configure with an isolated include dir
holding just `omp.h` (copied from llvm-20) so it does not shadow clang-22's own `stdint.h`:
`-DOpenMP_CXX_FLAGS="-fopenmp=libomp -isystem <dir-with-only-omp.h>"`,
`-DOpenMP_omp_LIBRARY=/usr/lib/llvm-20/lib/libomp.so`, `OpenMP_{C,CXX}_LIB_NAMES=omp`.

### End-to-end (10k images, warm cache)

| Config | µs/image | notes |
| --- | ---: | --- |
| NPU (HSA0), start of session | ~333 | before RELU/depad/convert_pad vectorization |
| NPU (HSA0), **now** | **~54** | all inference kernels vectorized; **6.2× vs session start** |
| CPU 1 core (`taskset -c 0`, `OMP_NUM_THREADS=1`), Release | ~16 | fair 1-core compute baseline |
| CPU all 16 cores, Release+OpenMP | ~2.6 | full-chip |

Accuracy 98.00% (NPU) / 98.01% (CPU); loss 0.066372 / 0.066352 (5th-decimal bf16 diff), unchanged.

**Corrected gap:** NPU is **~3.4× slower than 1 CPU core**, **~21× slower than the full 16-core CPU**.
The §1 "~11.4×" was against the crippled Debug CPU and is superseded. The kernel work was real (NPU
333→54) but earlier gap framing compared it to an -O0 CPU.

### Per-op comparison, CPU vs NPU (per-node eval-callback profiler, 2026-07-15)

Backend-agnostic probe: `ggml_backend_sched_set_eval_callback` with a `ggml_backend_sched_synchronize`
after each node (env `GGML_MNIST_PROFILE_LAYERS`). **Serializes execution** (defeats batching), so
absolute totals are inflated — use per-op SHARE, not throughput. Temporary; reverted after measuring.
Eval-only ops (CROSS_ENTROPY_LOSS / ARGMAX / COUNT_EQUAL) are the loss/accuracy harness and vanish in
real inference. `UNARY` = RELU; `NONE` = views/reshape (no compute).

| op | NPU µs/disp | NPU % | CPU-1core µs/disp | CPU-1c % | NPU/CPU-1c ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| **MUL_MAT** | 8,665 | **33.0%** | 3,713 | 92.9% | **2.3×** |
| ADD (bias) | 4,634 | 17.6% | 116 | 2.9% | 40× |
| UNARY (RELU) | 6,059 | 11.5% | 52 | 0.6% | 117× |
| CROSS_ENTROPY_LOSS | 12,227 | 23.3% | 132 | 1.7% | 93× (eval-only) |
| ARGMAX | 1,968 | 7.5% | 2.2 | 0.1% | 894× (eval-only) |
| COUNT_EQUAL | 3,762 | 7.2% | 107 | 1.3% | 35× (eval-only) |

(NPU total serialized ~1.05 s/run; CPU-1core ~0.16 s/run. CPU-16core MUL_MAT drops to ~530 µs/disp,
88% — MUL_MAT dominates on every backend.)

### Next big NPU target: MUL_MAT (the GEMM path)

- **MUL_MAT is now the top NPU cost at 33%** (8,665 µs/disp), and unlike the eval-only ops it is
  inference-relevant. It is also the only op where the NPU is *close* to the CPU (2.3× a single core
  vs 40–117× on the elementwise ops), i.e. the AIE is being used well but there is still headroom.
- The 8,665 µs/disp still bundles the GEMM proper plus its convert_pad pre-amble and depad post-amble
  (attributed to the MUL_MAT node). Now that convert_pad/depad are vectorized, the remaining cost is
  more genuinely the GEMM + DMA. Split the pre/gemm/post with the §4 per-dispatch probe to confirm the
  mix before optimizing.
- Candidate levers (in rough priority): (a) **cache converted+padded weights** — weights are constant
  across batches but re-converted every batch (§ candidate #3); (b) larger GEMM tiles / better AIE
  utilization in `gemm.py`; (c) multi-column fan-out. The elementwise ops (ADD/RELU), though 40–117×
  the CPU per-op, are small absolute shares now and dominated by per-dispatch/DMA overhead, not
  compute — a design-level tile-granularity change, not more kernel vectorization.
- The eval-only ops (CROSS_ENTROPY_LOSS 23%, ARGMAX, COUNT_EQUAL) are the largest serialized shares
  but are NOT worth optimizing for inference — they only run in the accuracy/loss harness.

---

## 6. ADD bias-add optimization — the §4 #1 target, resolved (2026-07-14)

Acting on §4's ranking (ADD = 55% of serialized dispatch, the top candidate). The expensive ADDs
are the two FC **bias adds** (`out[row,i] = src0[row,i] + src1[i]`, src1 a single row broadcast
over the batch dim). They routed through `transform_binary_broadcast_n`, a scalar loop doing a **4D
coordinate decomposition per element**: 3 signed `/` + 4 signed `%` against runtime ints = ~7
`__divsi3` libcalls/element, which also block vectorization. **The `__divsi3` storm, not the add,
was the cost.**

Fix (design + kernel co-change, `binary_ops.py` + `binary_ops.cc`, commit a2da7898): an ADD-only
dispatch predicate routes the single-row bias to a new **row-tiled** design — tile = one dst row,
src1 loaded once (`depth=1` fifo) and reused across rows, one row added per kernel call
(`ggml_op_add_bias`). Row-tiling handles the broadcast structurally, so the kernel has **zero index
math and zero `__divsi3`**. Scope was ADD only; SUB/MUL/DIV and general 4D broadcast still use the
scalar path.

Measured (10k MNIST FC f32, warm cache; ADD µs/dispatch via the §4 serialized profiler — relative
only; e2e via §1 batched clean run):

| Variant | e2e µs/image | ADD µs/dispatch | ADD vs GEMM | accuracy |
| --- | ---: | ---: | ---: | ---: |
| baseline (scalar 4D broadcast) | ~333 | 59,430 | ~9× | 98.00% |
| row-tiled **scalar** | ~124 | 8,578 | ~0.5× | 98.00% |
| row-tiled **vectorized** (shipped) | **~115** | **6,051** | ~0.35× | 98.00% |

- **2.9× end-to-end** (333 → 115 µs/image), accuracy unchanged, ADD/MUL_MAT tests 3/3 at 5e-4.
- **The big lever was removing `__divsi3` (scalar row-tiling alone = 2.7×), not vectorization.**
  Vectorization added ~7% more e2e on top — worthwhile but secondary, and by then ADD is already
  well below the GEMM. This refines §4's framing: ADD's device time was dominated by per-element
  scalar libcalls, so eliminating them cut real wall-clock hard even though device execution (not
  dispatch) was the wall (per the Reconciliation).
- ADD is **no longer the bottleneck** — it fell from ~9× the GEMM to ~0.35×. The §4 ranking now
  leads with the MUL_MAT machinery (see §4: pre/gemm/post ≈ 27.5%) and RELU.

Vectorization gotchas (both cost a debug cycle; see the binary-op vectorization memory note):
- Use `aie::load_unaligned_v`/`store_unaligned_v`, not the aligned forms — the double-buffered fifo
  streams consecutive rows at a non-vector-aligned stride, so aligned load/store corrupt alternate
  (ping-pong) rows.
- Do **not** put `AIE_LOOP_MIN_ITERATION_COUNT(1)` on the `vend=(N/V)*V` loop: the fc2 bias row is
  ne0=10 < V=16 → 0 trip count, and promising ≥1 makes the pipelined prologue run the body on too
  few elements (deterministic wrong result, not the cold-cache flake).

Next per §4/candidates: multi-core fan-out for the binary ops (separate follow-up), then the
convert+pad/depad and MUL_MAT path. The profiler probe used here was temporary (reverted).

---

## 2. Doorbell flushes per image (2026-07-14)

Probe: env-gated counter (`GGML_HSA_COUNT_FLUSHES`) in `ggml_hsa_flush_dispatches`, incremented only
when a doorbell is actually rung (`n_batched != 0`).

Result over the 10k-image run:
- **20 doorbell rings total** (one per graph_compute / 500-image batch).
- Each flush rings **15 packets** at once (`n_batched == 15` every time).
- **Flushes per image = 0.002** (1 per 500 images).

Interpretation: packet batching (dispatch-overhead doc, opt #1, PR #199) is fully effective here.
The entire 13-node FC graph — 2 MUL_MATs with their on-device convert/pad + depad kernels plus the
adds/relu — lowers to 15 dispatch packets that ride a **single doorbell ring at end of graph**. No
mid-graph flushes. At 500 images/batch the doorbell round-trip is fully amortized, so per-image
runtime is NOT doorbell-bound at this batch size.

---

## 3. Mid-graph queue drains (`wait_dispatches`) (2026-07-14)

Probe: env-gated counters (`GGML_HSA_COUNT_WAITS`) in `ggml_hsa_wait_dispatches`, plus labels at
`ggml_backend_hsa_synchronize` and the top of `ggml_backend_hsa_graph_compute`.

Result over the 10k-image run (20 batches):
- **44 total `wait_dispatches` calls**; 42 of them are `wait@synchronize`.
- **2 synchronizes per graph_compute**, and **0 drains from inside the backend's graph_compute loop**
  (the host-fallback drains never fire when on-device transforms are active).

Root cause of the 2/graph: they come from the **ggml scheduler** (`src/ggml-backend.cpp`), not the
HSA backend:
- Input-copy sync before a split (line ~1565), and
- unconditional `ggml_backend_synchronize` after each split's `graph_compute_async` (line ~1706).

`GGML_SCHED_DEBUG=2` confirms the FC graph is a **single split, entirely on HSA0, 0 copied inputs**,
so the graph runs drain-free start→finish; the scheduler then does one post-compute sync to hand
results back (necessary — the caller reads the logits).

Why the scheduler uses full drains instead of events: `mnist-eval` builds the scheduler in
**non-parallel** mode (`n_copies == 1`), so `sched->events[]` are NULL and it falls back to
`ggml_backend_synchronize` instead of the cheaper event path. The HSA backend already implements
`event_record`/`event_wait`; using them would require a parallel scheduler or different event wiring
(upstream / example change, not a backend change).

Conclusion: **the backend already does not drain until end-of-graph.** Remaining per-graph syncs are
scheduler-owned.

---

## 4. Per-dispatch latency profile (2026-07-14)

Probe: env-gated wrapper (`GGML_HSA_PROFILE_DISPATCH`) around every kernel dispatch in
graph_compute (source pre-process, main kernel, result post-process), which **drains the queue
before stopping the timer** so each measurement is that kernel's isolated submit+execute latency.

IMPORTANT: this **serializes execution and defeats batching**, so absolute totals are inflated
(~215 ms/batch measured here vs ~166 ms batched). Use these numbers for **relative** per-kernel cost
and ranking, NOT absolute throughput. Each dispatch pays a full doorbell round-trip that batching
normally hides.

MNIST FC eval graph (includes eval-only ops CROSS_ENTROPY_LOSS / ARGMAX / COUNT_EQUAL). Per
500-image batch:

| kernel | µs/dispatch | dispatches/batch | share of serialized total |
| --- | ---: | ---: | ---: |
| **ADD** | 59,430 | 2 | **55.1%** |
| RELU | 15,242 | 1 | 7.1% |
| CROSS_ENTROPY_LOSS | 12,754 | 1 | 5.9% |
| pre:MUL_MAT (convert+pad) | 8,909 | 4 | 16.5% |
| MUL_MAT (gemm) | 6,558 | 2 | 6.1% |
| post:MUL_MAT (depad) | 5,320 | 2 | 4.9% |
| COUNT_EQUAL | 4,258 | 1 | 2.0% |
| ARGMAX | 2,564 | 2 | 2.4% |

Raw totals (µs, summed over the whole 10k run / 20 batches, count = dispatches over the run):
`ADD 2377185/40`, `pre:MUL_MAT 712693/80`, `RELU 304845/20`, `MUL_MAT 262331/40`,
`CROSS_ENTROPY_LOSS 255086/20`, `post:MUL_MAT 212806/40`, `ARGMAX 102559/40`,
`COUNT_EQUAL 85160/20`; TOTAL `4312665/300`.

Key observations:
- **ADD dominates at ~55%** — ~59 ms/dispatch, ~9× the GEMM (~6.6 ms) it feeds. These are the FC
  bias adds (broadcast `[N] + [M,N]`), wildly disproportionate to their FLOP count. **Prime
  optimization target**, ahead of the matmul path.
- MUL_MAT machinery combined (pre + gemm + post) ≈ 27.5%; the GEMM itself is only ~6%.
- **convert+pad (~8.9 ms) costs more than the GEMM (~6.6 ms) it prepares.** The convert and depad
  kernels are scalar per-row loops on a single worker (unvectorized, un-parallelized) — obvious
  lever if the MUL_MAT path becomes a priority.

---

## 5. Host-phase breakdown — historical baseline (2026-07-12)

Recorded after #2 (kernarg allocator) landed but **before** #3 (on-device convert), when convert/pad
still ran on the host. Kept for history; superseded by §1–§4 for the current (post-#3) state. This is
the measurement the "doorbell is 94%" conclusion (since corrected — see Reconciliation above) came
from.

Probe: env-gated (`GGML_HSA_PROFILE=1`) accumulating `ggml_time_us` timers around each host phase of
`graph_compute`. Averaged over 3 runs, 10k images.

| Phase | Time | Share |
| --- | ---: | ---: |
| dispatch (doorbell etc.) | ~3300 ms | ~94.0% |
| fill (host f32→bf16 convert + zero-pad of sources) | ~177 ms | ~5.0% |
| copyback (host de-pad of result) | ~33 ms | ~0.9% |
| wait_pre / wait_post (queue drains around padded ops) | ~0.1 / ~0.1 ms | <0.01% |
| **host-ops total** (fill + copyback + waits) | **~210 ms** | **~6.0%** |
| grand total | ~3510 ms | |

Per-run phase call counts: dispatch=180, fill=40, copyback=40.

End-to-end at the time: HSA0 ~350–357 µs/image (median ~353), 98.00% acc; CPU ~24–26 µs/image,
98.01%.

What it told us then (with post-hoc correction):
- Host-ops were ~6% of backend time, essentially all in `fill` (convert+pad scatter of the two
  MUL_MAT operands). This is why moving them on-device (#3) was expected to be structural, not a
  direct time win — borne out by §1 (no end-to-end change from #3).
- The queue drains themselves cost ~nothing (~0.1 ms). Their damage was structural: each padded
  MUL_MAT forced drain → host fill → dispatch → drain → host copyback, fragmenting batching. #3
  removed that fragmentation (§2/§3 confirm 0 backend-internal drains now).
- "Dispatch is ~94%" was read as doorbell overhead; per the Reconciliation it is mostly device
  execution time counted inside the blocking doorbell store.

### Prerequisite for any NPU MNIST profiling: F32 MUL_MAT must run on the AIE

The ~333–353 µs/image baselines assume both MNIST matmuls run on the NPU. F32 `MUL_MAT` support
(f32→bf16 convert + zero-pad to the GEMM tile multiples, de-pad on the way out) is required; without
it the matmuls fall back to CPU (`KeyError: 'f32'`) and the profile is meaningless. Confirm before
profiling:

```
source build/.venv/bin/activate
"$PWD/build/bin/test-backend-ops-mnist" test -o MUL_MAT   # expect 3/3 OK at strict 5e-4
```

### Re-adding the host-phase timers (recipe)

The timers are intentionally not committed. To re-measure the host-phase breakdown:

1. Add a file-scope `static const bool g_ggml_hsa_profile = <getenv("GGML_HSA_PROFILE") truthy>;` and
   a small totals struct whose destructor `fprintf`s the table to `stderr` (needs `<cstdio>`; use
   `ggml_time_us()` from `ggml.h`).
2. In `graph_compute`, wrap and accumulate `ggml_time_us()` deltas for: (a) the `requires_sync` drain
   → `wait_pre`; (b) the source-prepare loop (convert/pad dispatch or host copy) → `fill`; (c)
   `kernel->dispatch(...)` → `dispatch`; (d) the post drain → `wait_post`; (e) the copy-back /
   post-process → `copyback`.
3. Build, run `GGML_HSA_PROFILE=1 mnist-eval ... HSA0` (table prints at exit). Run 3×; variance <1%.
4. Revert when done.

---

## Candidate optimizations (prioritized by the data above)

1. **ADD / bias-add kernel** — 55% of serialized dispatch time; investigate why the broadcast add is
   so expensive relative to its work (kernel vectorization, worker fan-out, data movement).
2. **Vectorize / multi-core the convert+pad and depad kernels** — currently scalar single-worker;
   convert alone costs more than the GEMM. (Kernels: `src/ggml-hsa/kernels/iron_kernels/convert_pad.*`,
   `depad.*`.) Note the DMA strided-transfer limit: keep fill/drain linear (see the on-device
   convert/pad memory note).
3. **Cache converted+padded weights** — model weights are constant across batches; their f32→bf16
   pad conversion is currently redone every batch (half of the pre:MUL_MAT dispatches). Cache the
   converted buffer keyed on the source data pointer to remove that half. Bigger lever now that
   convert is on-device (§4) than when it was ~5% host fill (§5); still a complement to kernel-level
   speedups, not a structural change.
4. **Event-based scheduler syncs** — replace the 2 full drains/graph with the event path (requires
   parallel scheduler / event wiring; upstream or example change).

## Notes on methodology

- Always run with kernels pre-cached (run once to populate `~/.cache/ggml/aie2`), or the first
  run's JIT compile time pollutes the totals.
- Move `~/.cache/ggml/aie2` aside to force recompile when a kernel changes. If you change `gemm.py`
  tile sizes or the padding factors, move stale `mul_mat-*` artifacts out of the cache dir
  (`find ... -delete`; `rm` is blocked in this env) so the kernel recompiles.
- Activate the build venv and use the absolute path to `mnist-eval`.
- Compare against the CPU baseline (`... CPU`) for correctness (98.0% accuracy) after any change.
- The `us/image` figure is on `stderr` from `mnist_model_eval`; capture with `2>&1`.
- The per-dispatch profiler (§4) is the only probe that changes execution semantics (serializes);
  §1–§3 and §5 probes are observational and do not alter timing materially.
- Absolute µs from §4 are upper bounds; cross-check headline speedups with the batched end-to-end
  number (§1).
