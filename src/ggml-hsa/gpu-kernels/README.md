# GGML HSA GPU kernels

Precompiled HIP elementwise kernels for the GGML HSA **GPU** backend
(`GGML_HSA_GPU`). The backend loads AMDGPU code objects at runtime from:

```
${GGML_HSA_KERNEL_DIR}/<device_name>/<kernel_symbol>.hsaco
```

resolving the kernel descriptor symbol `<kernel_symbol>.kd`. `<device_name>` is
the HSA agent name (e.g. `gfx1151`) and `<kernel_symbol>` is the sanitized ggml
kernel name (e.g. `add_8f32_8f32_8f32`).

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
reads kernel names (one per line) from a file or stdin and currently supports
pure elementwise `add_<N><dtype>_<N><dtype>_<N><dtype>` (f32, f16).

The exact kernel names a test needs can be captured by running with logging on;
misses are reported as `could not find code object for kernel <name>`:

```bash
LD_LIBRARY_PATH=/opt/rocm/lib GGML_HSA_ENABLE_LOG=1 \
  ./build/bin/test-backend-ops -o ADD 2>&1 \
  | grep -oE 'kernel add_[0-9a-zA-Z_]+' | sed 's/kernel //' \
  | awk -F'_' '$2==$3 && $3==$4 {print}' | sort -u > names.txt

ROCM_PATH=/opt/rocm ./generate_gpu_kernels.sh gfx1151 build/gpu-kernels names.txt
```

## Running

```bash
LD_LIBRARY_PATH=/opt/rocm/lib \
GGML_HSA_KERNEL_DIR=$PWD/build/gpu-kernels \
  ./build/bin/test-backend-ops -o ADD
```

Shapes without a matching `.hsaco` (e.g. broadcast / permuted adds) report
`not supported` and are skipped; they are not failures.
