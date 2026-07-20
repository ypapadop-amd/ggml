# matmul benchmark (`bench-mul-mat-hsa`)

Google Benchmark microbenchmark for `GGML_OP_MUL_MAT` (`C = A * Bᵀ`, with
A: `[K, M]`, B: `[K, N]`, C: `[M, N]`) across the CPU, GPU (HIP) and NPU (HSA)
backends, for `F32` and `BF16` inputs at `M=N=K ∈ {512, 1024, 2048, 4096}`.

## Important: never compile HIP and HSA together

The HIP and HSA backends **must not be compiled into the same binary** — doing
so segfaults at graph-compute time. Each backend therefore gets its own isolated
build directory, and the benchmark is run once per backend. The benchmark's
per-backend cases self-skip (`SkipWithError`) when their backend is not compiled
in, so a CPU-only or HIP-only build still builds and runs the benchmark.

## Build (one isolated directory per backend)

All builds are fully optimized: `Release` (`-O3 -DNDEBUG`) + `-march=native`
(`GGML_NATIVE`) + LTO (`GGML_LTO`), with tests enabled (needed to build the
benchmark), run from the repo root.

Note: `tests/` enables `-UNDEBUG` to keep asserts on in the test suite. The
benchmark's `CMakeLists.txt` re-asserts `-DNDEBUG` so the benchmark and the
Google Benchmark harness are genuinely optimized (asserts and the harness's
internal `CHECK`s compiled out; `library_build_type` reports `release`).

CPU only:

```bash
cmake -S . -B build-bench-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=OFF -DGGML_HSA=OFF -DGGML_BUILD_TESTS=ON
cmake --build build-bench-cpu --target bench-mul-mat-hsa -j"$(nproc)"
```

GPU (HIP — no HSA):

Set `GPU_TARGETS` to the arch of the GPU you are building for. Find it with
`rocm_agent_enumerator` (e.g. `gfx1150`, `gfx1103`, `gfx1100`). Note that
rocBLAS must ship a matching `TensileLibrary_lazy_<arch>.dat` under
`/opt/rocm-*/lib/rocblas/library/` or the run aborts (see the run notes below
for the gfx1103 case on this box).

```bash
GPU_ARCH=$(rocm_agent_enumerator | awk 'NR==1')   # or hard-code e.g. gfx1150
cmake -S . -B build-bench-gpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=ON -DGGML_HSA=OFF -DGGML_BUILD_TESTS=ON \
  -DGPU_TARGETS="${GPU_ARCH}" \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake"
cmake --build build-bench-gpu --target bench-mul-mat-hsa -j"$(nproc)"
```

NPU (HSA/aie2p — no HIP):

Point `hsa-runtime64_DIR` / `CMAKE_PREFIX_PATH` at the HSA install that
contains the AIE headers (`include/hsa/hsa_ext_amd_aie.h`) — on this box that
is `/home/ypapadop/workspace-raiders/opt/rocm`. If the header is missing the
build fails with `fatal error: hsa/hsa_ext_amd_aie.h: No such file or
directory`; adjust the two paths below to wherever your HSA-with-AIE tree
lives.

```bash
cmake -S . -B build-bench-npu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=OFF -DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON -DGGML_BUILD_TESTS=ON \
  -Dhsa-runtime64_DIR=/home/ypapadop/workspace-raiders/opt/rocm/lib/cmake/hsa-runtime64 \
  -DCMAKE_PREFIX_PATH="/home/ypapadop/workspace-raiders/opt/rocm/lib/cmake/hsa-runtime64;/opt/rocm/lib/cmake"
cmake --build build-bench-npu --target bench-mul-mat-hsa -j"$(nproc)"
```

## Run

`repro-matmul.sh` isolates one backend per invocation (`--benchmark_filter`),
writes a Google Benchmark JSON, and generates a markdown report from it. It
defaults `BUILD_DIR` to `build-bench-<target>` and, for the NPU, activates the
repo-root `.venv` (IRON / mlir_aie toolchain) needed for AIE JIT.

Output files are named after the hardware architecture actually running the
benchmark, detected at run time (CPU: `gcc -march=native`; GPU:
`rocm_agent_enumerator`; NPU: the `aie2`/`aie2p` agent name from `rocminfo`) —
not just the generic `cpu`/`gpu`/`npu` target. This keeps results from
different machines/architectures from overwriting each other. For the GPU the
detection strips `HSA_OVERRIDE_GFX_VERSION` first, so the file is named after
the *physical* device (e.g. `gfx1103`) even when the run borrows another arch's
rocBLAS via the override — the name reflects the real hardware, not the
override target.

```bash
cd tests/ggml-hsa/benchmarks
./repro-matmul.sh cpu     # -> results-cpu-<arch>.json + results-cpu-<arch>.md, e.g. results-cpu-znver4.*
./repro-matmul.sh gpu     # -> results-gpu-<arch>.json + results-gpu-<arch>.md, e.g. results-gpu-gfx1103.*
./repro-matmul.sh npu     # -> results-npu-<arch>.json + results-npu-<arch>.md, e.g. results-npu-aie2.*
```

The NPU MUL_MAT kernel defaults to the IRON path. Set
`GGML_HSA_PREFER_TRITON=1` to benchmark the Triton path instead — it
flips the kernel-spec order (Triton primary, IRON fallback) and tags the output
files with `-triton` so they don't overwrite the IRON results:

```bash
GGML_HSA_PREFER_TRITON=1 ./repro-matmul.sh npu   # -> results-npu-aie2-triton.*
```

Env vars: `BUILD_DIR`, `REPS` (default 10), `MIN_TIME` (default `0.5s`),
`OUTDIR` (default: this directory), `GGML_HSA_PREFER_TRITON` (NPU only:
`1` = use the Triton kernel and tag output `-triton`).

Notes:
- The script activates `${REPO_ROOT}/.venv` (IRON / mlir_aie toolchain) for the
  NPU path. This `.venv` must exist — if the working IRON venv lives elsewhere
  (e.g. `build/.venv`), symlink it: `ln -s build/.venv .venv` from the repo
  root.
- The first NPU run JIT-compiles the AIE kernels into `$XDG_CACHE_HOME/ggml/aie2p`
  (slow); Google Benchmark's warmup runs before timing, so JIT is not measured,
  but the wall-clock of the first run is long. Later runs reuse the cache.
- The HSA backend currently segfaults during process teardown *after* all
  results are written. `repro-matmul.sh` tolerates this: it proceeds to report
  generation as long as the JSON was produced.
- GPU arch vs. rocBLAS: the GPU build must target the actual device arch
  (`GPU_TARGETS`), and rocBLAS must have a matching
  `TensileLibrary_lazy_<arch>.dat`. On this box the iGPU is **gfx1103** (Radeon
  780M), for which the installed rocBLAS ships **no** TensileLibrary, so a
  native gfx1103 run aborts with `rocBLAS error: Cannot read
  .../TensileLibrary.dat ... for GPU arch : gfx1103`. Workaround: build for a
  supported near-arch (gfx1150) and force the runtime to match with
  `HSA_OVERRIDE_GFX_VERSION=11.5.0 ./repro-matmul.sh gpu`. On a GPU whose arch
  rocBLAS supports natively (e.g. gfx1100/gfx1150 hardware), no override is
  needed.

## RELU benchmark (`bench-relu-hsa`)

Google Benchmark microbenchmark for `GGML_UNARY_OP_RELU` (`out = relu(a)`,
element-wise and memory-bound) across the CPU, GPU (HIP) and NPU (HSA) backends,
for `F32` at the realistic MNIST activation shapes (`ne0×ne1×ne2×ne3`):

| shape | elements | origin |
|---|---:|---|
| `500×500×1×1` | 250 000 | FC1 hidden activation |
| `14×14×16×500` | 1 568 000 | conv2 output (post-bias) × batch 500 |
| `28×28×8×500` | 3 136 000 | conv1 output (post-bias) × batch 500 |

Same build/run model as matmul (one isolated build dir per backend; the target is
`bench-relu-hsa`). The reported metric is memory bandwidth (`GB/s`, counting one
read + one write per element) since RELU does negligible compute. Reports use
`report_relu_benchmarks.py` (not the matmul reporter — different name/shape regex).

```bash
cd tests/ggml-hsa/benchmarks
BUILD_DIR=build-bench-cpu ./repro-relu.sh cpu   # -> results-relu-cpu-<arch>.*
BUILD_DIR=build-bench-gpu ./repro-relu.sh gpu   # -> results-relu-gpu-<arch>.*
BUILD_DIR=build-bench-npu ./repro-relu.sh npu   # -> results-relu-npu-<arch>.*
```

### IRON vs. Triton on the NPU

The NPU RELU op ships both an IRON and a Triton kernel, selected by
`GGML_HSA_PREFER_TRITON` (see [`../../../src/ggml-hsa/kernels`], `order_kernel_specs`).
The intent is to run the benchmark twice and diff:

```bash
BUILD_DIR=build-bench-npu ./repro-relu.sh npu                          # IRON   (default)
GGML_HSA_PREFER_TRITON=1 BUILD_DIR=build-bench-npu ./repro-relu.sh npu # Triton (tag -triton)
```

**Result on this box (aie2 / Phoenix NPU, 2026-07-20): a true head-to-head is not
currently possible — the Triton RELU kernel fails to compile.** With
`GGML_HSA_PREFER_TRITON=1`, Triton is tried first, its `aircc` step fails
(`'transform.structured.pad' op expects a padding value of type 'f32', got
0.000000e+00 : bf16` in `relu_aie2.mlir`, plus `'aie.tile' op allocated buffers
exceeded available memory`), and the dispatch **falls back to IRON**. So both the
default and the `-triton`-tagged run measure the *same* IRON kernel and report
identical times. To get a real Triton number, the `relu_aie2.mlir` transform
script needs the bf16/f32 pad-value fix first. Always confirm which backend
actually ran by checking the cached artifact: a successful Triton compile leaves
`~/.cache/ggml/aie2/relu-<n>f32-<n>f32.pdi` sourced from `*-triton-artifacts`; on
this box only `*-iron-artifacts` produces the `.pdi`.

Cache caveat: the kernel cache key (`relu-<nelem>f32-<nelem>f32`) does **not**
encode the backend, so a `.pdi` compiled by one backend is reused by the other.
When comparing, clear the relevant `~/.cache/ggml/aie2/relu-*` entries between runs
(move them aside; `rm` is blocked in this env) and use `GGML_HSA_ENABLE_LOG=1` to
see which backend compiles.

### Measured numbers (aie2 NPU, znver4 CPU, IRON kernel; REPS=5, MIN_TIME=0.3s)

| shape | CPU µs (GB/s) | NPU-IRON µs (GB/s) |
|---|---:|---:|
| `500×500×1×1`   | 19.0 (105) | 602 (3.3) |
| `14×14×16×500`  | 1073 (11.7) | 3061 (4.1) |
| `28×28×8×500`   | 1072 (23.4) | 5968 (4.2) |

RELU alone is memory-bound and small, so the NPU's DMA-in/compute/DMA-out round
trip loses to the CPU on this op in isolation — the NPU win comes from fusing RELU
into a larger on-device graph (no host round-trip), not from RELU standalone.

## Plotting

Per-backend reports are generated automatically by `repro-matmul.sh`.
`plot_benchmarks.py` produces a bar chart across backends (requires `matplotlib`):

```bash
./plot_benchmarks.py --labels cpu,gpu,npu \
  results-cpu-znver4.json results-gpu-gfx1103.json results-npu-aie2.json -o comparison.png
```

Explicit `--labels` are used verbatim as the legend entries (when omitted, the
legend falls back to `<filename-stem>: <backend>`), so use them to distinguish
two runs of the *same* backend — e.g. an IRON vs. Triton NPU comparison (both
report backend `HSA`), keeping each in its own result file:

```bash
./plot_benchmarks.py --labels "CPU","GPU (HIP)","NPU (HSA/IRON)","NPU (HSA/Triton)" \
  results-cpu-znver4.json results-gpu-gfx1103.json \
  results-npu-aie2.json results-npu-aie2-triton.json -o comparison.png
```

By default the y-axis is throughput (GFLOP/s, higher is better). Pass
`--metric time` to plot wall time in milliseconds instead (log-scaled, lower is
better):

```bash
./plot_benchmarks.py --metric time --labels "CPU","GPU (HIP)","NPU (HSA/IRON)","NPU (HSA/Triton)" \
  results-cpu-znver4.json results-gpu-gfx1103.json \
  results-npu-aie2.json results-npu-aie2-triton.json -o comparison-time.png
```

Pass `--exclude-sizes` (comma-separated square dims M=N=K) to drop shapes from
the chart — e.g. omit the small 512 case where fixed overhead dominates:

```bash
./plot_benchmarks.py --exclude-sizes 512 --labels "CPU","GPU (HIP)","NPU (HSA/IRON)","NPU (HSA/Triton)" \
  results-cpu-znver4.json results-gpu-gfx1103.json \
  results-npu-aie2.json results-npu-aie2-triton.json -o comparison.png
```
