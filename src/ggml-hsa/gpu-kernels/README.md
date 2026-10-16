# GGML HSA GPU kernels

Precompiled HIP kernels for the GGML HSA **GPU** backend (`GGML_HSA_GPU`). The
backend searches for AMDGPU code objects in, in order:

```
1. ${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_name>.hsaco   (pregenerated)
2. <cache_dir>/<device_name>/<kernel_name>.hsaco             (cache)
```

`<device_name>` is the HSA agent name (e.g. `gfx1151`) and `<kernel_name>` is the
sanitized ggml kernel name (e.g. `add_8f32_8f32_8f32`). The kernel descriptor
symbol is not derived from the file name: the loader reads it from the code
object's metadata. Kernels from `generate_gpu_kernels.py` export the kernel name
itself (`add_8f32_8f32_8f32.kd`), while a Triton-compiled kernel exports its
`@triton.jit` function name (e.g. `vecadd.kd`).

A code object without HIP hidden arguments, such as a Triton kernel, bakes its
launch geometry in at compile time and needs a `<kernel_name>.hsaco.json`
sidecar holding `grid_size_x` and `workgroup_size_x`, plus an optional
`group_segment_size`: the dynamic LDS in bytes the kernel requests at launch
(Triton's shared memory, e.g. for `tl.dot`), which is added to the code
object's fixed LDS size. Loading such a kernel
fails if its sidecar is missing, and loading any kernel fails if its sidecar is
malformed.

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

`generate_gpu_kernels.py` emits a `.hsaco` per kernel name. It reads kernel
names (one per line) from a file or stdin and supports elementwise kernels over
contiguous, equally shaped tensors:

- `add_*` for every dtype combination ggml's CPU backend implements: f32, f16
  and bf16 with matching dtypes, and an f16 or bf16 source plus an f32 source
  into either of their dtypes.
- `scale_*` for f32 only (ggml's CPU backend has no f16/bf16 `SCALE` to check a
  generated kernel against).

Broadcasting and non-contiguous kernels are skipped. By default it writes into
the runtime cache directory above, so kernels are found automatically — no
`GGML_HSA_KERNEL_DIR` and no dependence on the build folder.

```
generate_gpu_kernels.py [arch] [names_file] [out_dir]
```

The exact kernel names a test needs can be captured by running with logging on;
misses are reported as `could not find code object for kernel <name>`:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib GGML_HSA_ENABLE_LOG=1 \
  ./build/bin/test-backend-ops -o ADD 2>&1 \
  | grep -oE 'kernel (add|scale)_[0-9a-zA-Z_]+' | sed 's/kernel //' | sort -u > names.txt

# Writes to the runtime cache dir (default); no GGML_HSA_KERNEL_DIR needed.
ROCM_PATH=/opt/rocm ./generate_gpu_kernels.py gfx1151 names.txt
```

To stage kernels in a fixed, checked-in location instead of the cache, pass an
explicit `out_dir` and point `GGML_HSA_KERNEL_DIR` at it.

## Running

```bash
LD_LIBRARY_PATH=/opt/rocm/lib ./build/bin/test-backend-ops -o ADD
```

Shapes without a matching `.hsaco` (e.g. broadcast / permuted adds) report
`not supported` and are skipped; they are not failures.
