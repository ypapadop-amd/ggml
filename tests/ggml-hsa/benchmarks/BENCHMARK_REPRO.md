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

GPU (HIP, gfx1150 — no HSA):

```bash
cmake -S . -B build-bench-gpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=ON -DGGML_HSA=OFF -DGGML_BUILD_TESTS=ON \
  -DGPU_TARGETS=gfx1150 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake"
cmake --build build-bench-gpu --target bench-mul-mat-hsa -j"$(nproc)"
```

NPU (HSA/aie2p — no HIP):

```bash
cmake -S . -B build-bench-npu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=OFF -DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON -DGGML_BUILD_TESTS=ON \
  -Dhsa-runtime64_DIR=/scratch/ypapadop/opt/rocm/lib/cmake/hsa-runtime64 \
  -DCMAKE_PREFIX_PATH="/scratch/ypapadop/opt/rocm/lib/cmake/hsa-runtime64;/opt/rocm/lib/cmake"
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
different machines/architectures from overwriting each other.

```bash
cd tests/ggml-hsa/benchmarks
./repro-matmul.sh cpu     # -> results-cpu-<arch>.json + results-cpu-<arch>.md, e.g. results-cpu-znver4.*
./repro-matmul.sh gpu     # -> results-gpu-<arch>.json + results-gpu-<arch>.md, e.g. results-gpu-gfx1150.*
./repro-matmul.sh npu     # -> results-npu-<arch>.json + results-npu-<arch>.md, e.g. results-npu-aie2p.*
```

Env vars: `BUILD_DIR`, `REPS` (default 10), `MIN_TIME` (default `0.5s`),
`OUTDIR` (default: this directory).

Notes:
- The first NPU run JIT-compiles the AIE kernels into `$XDG_CACHE_HOME/ggml/aie2p`
  (slow); Google Benchmark's warmup runs before timing, so JIT is not measured,
  but the wall-clock of the first run is long. Later runs reuse the cache.
- The HSA backend currently segfaults during process teardown *after* all
  results are written. `repro-matmul.sh` tolerates this: it proceeds to report
  generation as long as the JSON was produced.

## Plotting

Per-backend reports are generated automatically by `repro-matmul.sh`.
`plot_benchmarks.py` produces a bar chart across backends (requires `matplotlib`):

```bash
./plot_benchmarks.py --labels cpu,gpu,npu \
  results-cpu-znver4.json results-gpu-gfx1150.json results-npu-aie2p.json -o comparison.png
```
