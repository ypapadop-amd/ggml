# GGML HSA Backend

The GGML HSA (`ggml-hsa`) backend enables GGML tensor operations to run on AMD XDNA NPUs (AI Engines).

## Supported Devices

| Architecture | NPU Generation | Example Platforms                  |
|--------------|----------------|------------------------------------|
| `aie2`       | [AMD XDNA]     | Phoenix, Hawk Point                |
| `aie2p`      | [AMD XDNA2]    | Strix Point, Strix Halo, Krackan   |

[AMD XDNA]: https://www.amd.com/en/technologies/xdna.html
[AMD XDNA2]: https://www.amd.com/en/technologies/xdna.html#xdna2

## Supported Operations

| Category  | Operations                                                     |
|-----------|----------------------------------------------------------------|
| Binary    | `ADD`, `SUB`, `MUL`, `DIV` (with multi-dimensional broadcast)  |
| Unary     | `SQR`, `SQRT`, `LOG`, `ABS`, `SGN`, `NEG`, `STEP`, `FLOOR`, `CEIL`, `ROUND`, `TRUNC`, `RELU`, `HARDSWISH`, `HARDSIGMOID` |
| Matrix    | `MUL_MAT`                                                      |
| Pooling   | `POOL_2D` (`MAX` and `AVG`, with padding)                     |
| Convolution | `IM2COL` (2D, `f32` image, `f32`/`bf16` output)             |
| Reduction | `ARGMAX`, `COUNT_EQUAL`                                        |
| Loss      | `CROSS_ENTROPY_LOSS`                                           |
| Other     | `SCALE`, `SOFT_MAX`, `CLAMP`                                   |
| Host-only | `DUP`, `CPY`, `CONT` (CPU execution)                           |

> **Note:** Operations like `SIN`, `COS`, `EXP`, `TANH`, `ELU`, `SIGMOID`, `SILU`,
> `GELU`, `GELU_QUICK`, `GELU_ERF`, `XIELU` are registered but not yet implemented.

### Broadcasting

Binary operations support GGML-style broadcasting where `src1` can be repeated to match `dst`:

- `dst->ne[i] % src1->ne[i] == 0` must hold for all dimensions
- Examples: `(10,5,4,3) + (10,5,4,3)` (element-wise), `(20,5,4,3) + (10,5,4,3)` (broadcast in dim0)
- Multi-dimensional broadcasting: `(20,10,8,6) + (10,5,4,3)` (broadcast in all dims)

## Supported Data Types

| Type             | Support                                |
|------------------|----------------------------------------|
| `GGML_TYPE_I8`   | Native `aie2` / `aie2p` datatype       |
| `GGML_TYPE_I16`  | Native `aie2` / `aie2p` datatype       |
| `GGML_TYPE_I32`  | Native `aie2` / `aie2p` datatype       |
| `GGML_TYPE_BF16` | Native `aie2` / `aie2p` datatype       |
| `GGML_TYPE_F16`  | Supported via conversion to/from `BF16` |
| `GGML_TYPE_F32`  | Emulated (slower than native types)    |

## Prerequisites

### Tested Configurations

| Component   | Version                                                              |
|-------------|----------------------------------------------------------------------|
| OS          | [Ubuntu 24.04.2], [Ubuntu 25.10]                                     |
| ROCm        | [7.2.1][ROCm 7.2.1]                                                  |
| XDNA Driver | [1.6][XDNA Driver 1.6]                                               |
| MLIR-AIE    | [1.4.3][MLIR-AIE 1.4.3]                                              |

[Ubuntu 24.04.2]: https://releases.ubuntu.com/noble/
[Ubuntu 25.10]: https://releases.ubuntu.com/questing/
[ROCm 7.2.1]: https://rocm.docs.amd.com/en/docs-7.2.1/
[XDNA Driver 1.6]: https://github.com/amd/xdna-driver/tree/1.6
[MLIR-AIE 1.4.3]: https://github.com/Xilinx/mlir-aie/tree/v1.4.3

### ROCm

`ggml-hsa` requires [ROCm](https://github.com/ROCm/rocm-systems) 7.2.1 or newer. See the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

Due to ongoing NPU support work in [ROCR](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime), `ggml-hsa` requires a ROCR compiled from source: [`develop`](https://github.com/ROCm/rocm-systems/tree/develop) at or after commit [`512eb6021e7`](https://github.com/ROCm/rocm-systems/commit/512eb6021e7) ("fix(rocr/aie): Avoid unmap during AIE memory release", #12389), which fixes an abort when freeing an HSA vmem buffer. This is not optional: TheRock's nightly wheels do not carry this fix yet (see [One ROCR for both backends](#one-rocr-for-both-backends)).

`ggml-hsa` allocates its buffers with the HSA virtual memory (vmem) API.

`ggml-hsa` loads its kernels as AIE hsacos (HSA code objects) and dispatches them by kernel object.
This needs a ROCR with AIE hsaco support, which is not on `develop` yet: build branch
[`users/ypapadop-amd/aie-hsaco`](https://github.com/ROCm/rocm-systems/tree/users/ypapadop-amd/aie-hsaco).
It also needs the hsaco packer (`aie.compiler.hsaco`) from MLIR-AIE
[`main`](https://github.com/Xilinx/mlir-aie) at or after commit
[`a1c715a`](https://github.com/Xilinx/mlir-aie/commit/a1c715a709c) (#3796), which no MLIR-AIE
release contains yet, including the 1.4.3 that `requirements-iron.txt` pins.

### Compiling ROCR from source

```bash
REPO=/path/to/rocm-systems        # a checkout of https://github.com/ROCm/rocm-systems, `develop` branch
ROCR=$REPO/projects/rocr-runtime
PREFIX=$HOME/opt/rocm             # must not be /opt/rocm: that is the system runtime and lacks the AIE header

cmake -S "$ROCR" -B "$ROCR/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$ROCR/build" -j"$(nproc)"
cmake --install "$ROCR/build"
```

Install `libdrm-dev` and `libnuma-dev` first if they are missing. Confirm the AIE header made it into
the install, since the system runtime does not have it:

```bash
ls "$PREFIX/include/hsa/hsa_ext_amd_aie.h"
```

Re-run both the build and the install after every branch switch or pull: `ggml-hsa` links against
`$PREFIX`, not the build tree, so skipping the install silently keeps testing the previous commit.

### AMD XDNA Driver

`ggml-hsa` depends on the [AMD XDNA Driver](https://github.com/amd/xdna-driver). Installation instructions:

- Via IRON: [build_drivers.sh](https://github.com/Xilinx/mlir-aie/blob/main/utils/build_drivers.sh)
- Direct: [xdna-driver README](https://github.com/amd/xdna-driver#linux-compilation-and-installation)

### Compilation Backends

`ggml-hsa` supports multiple compilation backends for generating AIE kernels. Each operation selects its backend at runtime based on tensor configuration.

#### IRON (MLIR-AIE)

The default backend using the [IRON framework](https://github.com/Xilinx/mlir-aie).

Install IRON dependencies:

```bash
python3 -m pip install -r src/ggml-hsa/requirements-iron.txt
```

Or use the setup script to create a virtual environment:

```bash
source src/ggml-hsa/env_setup.sh iron
```

> **Note:** IRON environments consume considerable storage. For pre-generated kernels, set `GGML_HSA_KERNEL_DIR` and disable JIT at compile time.

#### Triton-XDNA

An optional backend using [Triton-XDNA](https://github.com/ROCm/Triton-XDNA) for compiler-driven kernel generation via MLIR-AIR/AIE.

Install Triton dependencies (includes IRON):

```bash
python3 -m pip install -r src/ggml-hsa/requirements-triton.txt
```

Or use the setup script:

```bash
source src/ggml-hsa/env_setup.sh triton
```

> **Note:** Operations may prefer either backend.

## Building

### Basic HSA Build

```bash
cmake -S . -B build \
  -DGGML_HSA=ON \
  -DGGML_HSA_JIT_COMPILE=ON \
  -Dhsa-runtime64_DIR=/path/to/rocm/lib/cmake/hsa-runtime64 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
```

### Combined HSA + HIP Build

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HSA=ON \
  -DGGML_HSA_JIT_COMPILE=ON \
  -Dhsa-runtime64_DIR=/path/to/rocm/lib/cmake/hsa-runtime64 \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1102 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j
```

### Running HIP and HSA in the Same Process

Using both backends in one process (e.g., HIP output feeding an NPU op) has two requirements that a
combined build alone does not meet.

#### One ROCR for both backends

A process loads only one `libhsa-runtime64.so.1`, and HIP and `ggml-hsa` must share it. It must be a
ROCR with NPU support, which a system ROCm usually does not provide: for example, ROCm 7.2.4's
`libamdhip64` links its own `libhsa-runtime64` without AIE support.

As of this writing, TheRock's nightly wheels bundle a ROCR that does not yet carry the AIE vmem unmap
fix described in [ROCm](#rocm) (`512eb6021e7`, #12389). The wheel is only used to get a matching HIP;
the ROCR built from source, not the one inside the wheel, is what `ggml-hsa` and HIP actually load at
run time — building it yourself (see [Compiling ROCR from source](#compiling-rocr-from-source)) is
required, not optional, until that fix reaches a nightly build.

One way to get a matching HIP is [TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)'s
nightly pip wheels, run against a ROCR built from source:

```bash
python3 -m venv build/.venv-rocm
source build/.venv-rocm/bin/activate
pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[devel,libraries]" rocm-sdk-device-gfx1103   # device package for your GPU
rocm-sdk init                                       # expands the devel tree (~3 GB)
ROCM_SDK="$(rocm-sdk path --root)"

HIPCXX="$ROCM_SDK/lib/llvm/bin/clang++" HIP_PATH="$ROCM_SDK" \
cmake -S . -B build \
  -DGGML_HSA=ON \
  -DGGML_HSA_JIT_COMPILE=ON \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1103 \
  -Dhsa-runtime64_DIR=/path/to/rocr/lib/cmake/hsa-runtime64 \
  -DCMAKE_PREFIX_PATH="$ROCM_SDK" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

`hsa-runtime64_DIR` points at the ROCR built from source, not at the one inside the wheel. At run
time, put that ROCR first so it is the one loaded, and activate the IRON environment for the JIT:

```bash
source /path/to/iron/venv/bin/activate
LD_LIBRARY_PATH=/path/to/rocr/lib ./build/bin/<program>
```

To check which libraries were loaded, run the program with `LD_DEBUG=libs` and look for
`libhsa-runtime64.so.1` and `libamdhip64.so` in the `calling init` lines.

#### Avoiding the LLVM clash with the JIT compiler

With HIP loaded, the first JIT compilation aborts the process:

```text
: CommandLine Error: Option 'print-inst-addrs' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

HIP loads its own `libLLVM.so`. The JIT compiler runs in an embedded Python interpreter and imports
MLIR-AIE, whose native modules also contain LLVM. Both copies register their command-line options in
the same registry.

The interpreter only starts when a kernel is missing from the cache, so the clash is avoided as long
as no kernel needs to be compiled in the HIP process:

- **Warm the kernel cache first.** Run the same operations, with the same shapes and types, once from
  a process that does not load HIP (e.g., an HSA-only build) with the same kernel cache directory.
  The HIP process then finds every kernel in the cache.
- **Use precompiled kernels.** Build with `-DGGML_HSA_JIT_COMPILE=OFF` and point
  `GGML_HSA_KERNEL_DIR` at precompiled kernels. No interpreter is ever started.

Do not set `GGML_HSA_KERNEL_CACHE_CLEAR=1` in the HIP process: it empties the cache and forces a compilation.

#### Sharing tensors with HIP

A HIP tensor can be used by the NPU in place, in either direction, without copies. HIP owns the
memory; the HSA backend imports the HIP buffer and places HSA tensors on the same memory:

```c
// HIP produces x; the NPU reads it and writes its result into the HIP tensor y
ggml_backend_buffer_t imported = ggml_backend_hsa_buffer_import(0, hip_buffer);

ggml_tensor * x_npu = ggml_new_tensor(hsa_ctx, x->type, GGML_MAX_DIMS, x->ne);
ggml_backend_hsa_tensor_alloc_alias(imported, x_npu, x);   // NPU input

ggml_tensor * r = ggml_sub(hsa_ctx, x_npu, c);
ggml_backend_hsa_tensor_alloc_alias(imported, r, y);       // NPU output, before allocating
ggml_backend_alloc_ctx_tensors(hsa_ctx, hsa_backend);
```

| Function | Purpose |
|----------|---------|
| `ggml_backend_hsa_buffer_import(device, buffer)` | Maps another ROCm backend's buffer for the NPU. The result is an HSA buffer over the same memory, at the same offsets. |
| `ggml_backend_hsa_tensor_alloc_alias(imported, tensor, src)` | Places an unallocated HSA tensor on the memory of `src`. `tensor` must have the type, shape and strides of `src` (copy `src->nb` when `src` is a view). For an NPU input, pass a new tensor; for an NPU output, pass the op's result, before the graph is allocated. |

The caller is responsible for:

- **Ordering.** Nothing synchronizes the two backends. Finish the producer before the consumer runs:
  `ggml_backend_graph_compute` waits for completion; after `ggml_backend_graph_compute_async`, call
  `ggml_backend_synchronize`.
- **Lifetime.** Free the imported buffer before the buffer it maps.

Only buffers of other ROCm backends in the same process can be imported, not host memory or HSA
buffers. The import maps the whole dma-buf the HIP buffer lives in (HIP places small buffers in a
shared block), but only the HIP buffer's range is reachable through the imported buffer.

`tests/ggml-hsa/test-hip-zero-copy-hsa.cpp` runs HIP → NPU → HIP this way and shows that the two
backends see the same memory. It is built only with `GGML_HIP=ON`.

## JIT Compilation

JIT compilation generates kernels on-the-fly. Precompiled kernels in `GGML_HSA_KERNEL_DIR` take precedence.

**Cache Location** (in order of precedence):

1. `GGML_HSA_KERNEL_CACHE_DIR`
2. `${XDG_CACHE_HOME}/ggml`
3. `$HOME/.cache/ggml`
4. `/tmp/ggml/ggml-hsa`

> **Warning:** Setting `GGML_HSA_KERNEL_CACHE_CLEAR=1` deletes all files in the cache directory.

## Reference

### CMake Options

| Option                 | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `GGML_HSA`             | Enable HSA backend                                                 |
| `GGML_HSA_JIT_COMPILE` | Enable JIT compilation (requires IRON environment)                 |

### Environment Variables

| Variable                                | Description                                                                                                                                                                                             |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GGML_HSA_ENABLE_LOG`                   | Enable internal logging (`1`, `true`, or `on`)                                                                                                                                                          |
| `GGML_HSA_KERNEL_DIR`                   | Precompiled kernel directory path                                                                                                                                                                       |
| `GGML_HSA_KERNEL_CACHE_DIR`             | JIT cache directory                                                                                                                                                                                     |
| `GGML_HSA_KERNEL_CACHE_CLEAR`           | Clear JIT cache on startup (`1`, `true`, or `on`)                                                                                                                                                       |
| `GGML_HSA_JIT_VERBOSE`                  | Verbose JIT output (`1`, `true`, or `on`)                                                                                                                                                               |
| `GGML_HSA_QUEUE_ERROR_DRAIN_TIMEOUT_MS` | Milliseconds teardown waits for work still in flight when the runtime suspended the queue (default `1000`, `0` disables the wait). On timeout the dispatch signal is leaked instead of destroyed.        |
| `GGML_HSA_KERNEL_INLINE`                | Inline supported kernels' core functions into the tile loop instead of linking them as a `.o` (mlir-aie `ExternalFunction(inline=True)`) (`1`, `true`, or `on`); default off. Support varies by kernel; see `core_function_object()` in `kernels/iron_kernels/utils.py`.        |
