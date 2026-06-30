# GGML HSA GPU kernels

Precompiled HIP kernels for the GGML HSA **GPU** backend (`GGML_HSA_GPU`). The
backend searches for AMDGPU code objects in, in order:

```
1. ${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_symbol>.hsaco   (pregenerated)
2. <cache_dir>/<device_name>/<kernel_symbol>.hsaco             (cache)
```

resolving the kernel descriptor symbol `<kernel_symbol>.kd`. `<device_name>` is
the HSA agent name (e.g. `gfx1151`) and `<kernel_symbol>` is the sanitized ggml
kernel name (e.g. `add_8f32_8f32_8f32`).

The cache directory mirrors the AIE JIT cache and is resolved as:
`GGML_HSA_KERNEL_CACHE_DIR`, else `$XDG_CACHE_HOME/ggml`, else
`$HOME/.cache/ggml`, else `/tmp/ggml/ggml-hsa`. Generated kernels are **not**
placed in the CMake build folder.

## Building the backend

```bash
cmake -S . -B build \
  -DGGML_HSA=ON \
  -DGGML_HSA_AIE=OFF \
  -DGGML_HSA_GPU=ON \
  -DGGML_HSA_JIT_COMPILE=OFF \
  -Dhsa-runtime64_DIR=/opt/rocm/lib/cmake/hsa-runtime64 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target ggml-hsa test-backend-ops
```

## Generating kernels

`generate_gpu_kernels.sh` emits a shape-specialized `.hsaco` per kernel name. It
reads kernel names (one per line) from a file or stdin and supports elementwise
`add_*` and `scale_*` (f32, f16). By default it writes into the runtime cache
directory above, so kernels are found automatically — no `GGML_HSA_KERNEL_DIR`
and no dependence on the build folder.

```
generate_gpu_kernels.sh <arch> [names_file] [out_dir]
```

The exact kernel names a test needs can be captured by running with logging on;
misses are reported as `could not find code object for kernel <name>`:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib GGML_HSA_ENABLE_LOG=1 \
  ./build/bin/test-backend-ops -o ADD 2>&1 \
  | grep -oE 'kernel (add|scale)_[0-9a-zA-Z_]+' | sed 's/kernel //' | sort -u > names.txt

# Writes to the runtime cache dir (default); no GGML_HSA_KERNEL_DIR needed.
ROCM_PATH=/opt/rocm ./generate_gpu_kernels.sh gfx1151 names.txt
```

To stage kernels in a fixed, checked-in location instead of the cache, pass an
explicit `out_dir` and point `GGML_HSA_KERNEL_DIR` at it.

## Running

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./build/bin/test-backend-ops -o ADD
```

Shapes without a matching `.hsaco` (e.g. broadcast / permuted adds) report
`not supported` and are skipped; they are not failures.
