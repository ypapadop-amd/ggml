# MNIST eval reproduction (`mnist-eval`)

Reproduces the MNIST FC (784->500->10) eval numbers (`test_acc`, `test_loss`,
`us/image`) across the CPU, GPU (HIP) and NPU (HSA) backends, evaluating the
trained model over the 10000-image MNIST test set.

## Important: never compile HIP and HSA together

The HIP and HSA backends **must not be compiled into the same binary** — doing
so segfaults at graph-compute time. Each backend therefore gets its own
isolated build directory, and the eval is run once per backend.

## Build (one isolated directory per backend)

All builds are `Release` + `-march=native` (`GGML_NATIVE`), with examples
enabled (needed to build `mnist-eval`), run from the repo root.

CPU only:

```bash
cmake -S . -B build-mnist-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=OFF -DGGML_HSA=OFF -DGGML_BUILD_EXAMPLES=ON
cmake --build build-mnist-cpu --target mnist-eval -j"$(nproc)"
```

GPU (HIP, gfx1150 — no HSA):

```bash
cmake -S . -B build-mnist-gpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=ON -DGGML_HSA=OFF -DGGML_BUILD_EXAMPLES=ON \
  -DGPU_TARGETS=gfx1150 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/lib/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake"
cmake --build build-mnist-gpu --target mnist-eval -j"$(nproc)"
```

NPU (HSA/aie2p — no HIP):

```bash
cmake -S . -B build-mnist-npu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_CPU=ON -DGGML_OPENMP=ON \
  -DGGML_HIP=OFF -DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON -DGGML_BUILD_EXAMPLES=ON \
  -Dhsa-runtime64_DIR=/scratch/ypapadop/opt/rocm/lib/cmake/hsa-runtime64 \
  -DCMAKE_PREFIX_PATH="/scratch/ypapadop/opt/rocm/lib/cmake/hsa-runtime64;/opt/rocm/lib/cmake"
cmake --build build-mnist-npu --target mnist-eval -j"$(nproc)"
```

## Run

`repro-mnist.sh` runs `mnist-eval` over the MNIST test set for one backend (or
`both` = NPU + CPU), stripping the noisy per-image ASCII digit dump and
printing the summary lines (accuracy, loss, timing). It defaults `BUILD_DIR`
to `build-release`, so pass the isolated per-backend directory built above.

Output files are named after the hardware architecture actually running the
eval, detected at run time (CPU: `gcc -march=native`; GPU:
`rocm_agent_enumerator`; NPU: the `aie2`/`aie2p` agent name from `rocminfo`) —
not just the generic `cpu`/`gpu`/`npu` target. This keeps results from
different machines/architectures from overwriting each other; `both` writes
one file per sub-run (NPU and CPU separately).

```bash
cd examples/mnist
BUILD_DIR=build-mnist-cpu ./repro-mnist.sh cpu   # -> results-cpu-<arch>.json + .md, e.g. results-cpu-znver4.*
BUILD_DIR=build-mnist-gpu ./repro-mnist.sh gpu   # -> results-gpu-<arch>.json + .md, e.g. results-gpu-gfx1150.*
BUILD_DIR=build-mnist-npu ./repro-mnist.sh npu   # -> results-npu-<arch>.json + .md, e.g. results-npu-aie2p.*
```

Env vars: `BUILD_DIR` (default `build-release`), `MODEL` (default
`mnist-fc-f32.gguf`), `RUNS` (default 1), `WARMUP=1` (drop the first run from
the timing summary), `REGEN=1` (regenerate the model + download the dataset
before running), `OUTDIR` (default: this directory).

Notes:
- For the `cpu`/`npu` targets, `HIP_VISIBLE_DEVICES=-1` hides the gfx1150 iGPU
  from the ROCm backend, which otherwise SIGSEGVs when synchronized as a
  fallback during scheduler alloc. The `gpu` target leaves HIP visible.
- The repo-root `.venv` is activated so the HSA AIE-kernel JIT (embedded
  pybind11 interpreter) can find the IRON toolchain. The first NPU run
  JIT-compiles kernels into `$XDG_CACHE_HOME/ggml/aie2p` (slow); later runs
  reuse the cache. Use `WARMUP=1` with `RUNS>1` to exclude it from the timing
  summary.
- Expected: `test_acc` ~98.17%, NPU ~36 us/image, CPU ~2.6 us/image, GPU ~5.8
  us/image. (Device availability depends on the build.)
