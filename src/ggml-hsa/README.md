# GGML HSA Backend

The GGML HSA (`ggml-hsa`) backend enables GGML tensor operations to run on AMD XDNA NPUs (AI Engines).

## Supported Devices

| Architecture | NPU Generation | Example Platforms       |
|--------------|----------------|-------------------------|
| `aie2`       | [AMD XDNA]     | Phoenix, Hawk Point     |
| `aie2p`      | [AMD XDNA2]    | Strix Halo, Krackan     |

[AMD XDNA]: https://www.amd.com/en/technologies/xdna.html
[AMD XDNA2]: https://www.amd.com/en/technologies/xdna.html#xdna2

## Supported Operations

| Category  | Operations                                                     |
|-----------|----------------------------------------------------------------|
| Binary    | `ADD`, `SUB`, `MUL`, `DIV` (with multi-dimensional broadcast)  |
| Unary     | `SQR`, `LOG`, `ABS`, `SGN`, `NEG`, `STEP`, `FLOOR`, `CEIL`, `ROUND`, `TRUNC`, `RELU`, `HARDSWISH`, `HARDSIGMOID` |
| Matrix    | `MUL_MAT`                                                      |
| Reduction | `ARGMAX`, `COUNT_EQUAL`                                        |
| Loss      | `CROSS_ENTROPY_LOSS`                                           |
| Other     | `SCALE`, `SOFT_MAX`, `CLAMP`                                   |
| Host-only | `DUP`, `CPY`, `CONT` (CPU execution)                           |

> **Note:** Operations like `SQRT`, `SIN`, `COS`, `EXP`, `TANH`, `ELU`, `SIGMOID`, `SILU`,
> `GELU`, `GELU_QUICK`, `GELU_ERF`, and `XIELU` are registered but not yet implemented.

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
| `GGML_TYPE_F16`  | Supported via conversion to `BF16`     |
| `GGML_TYPE_F32`  | Emulated (slower than native types)    |

## Prerequisites

### Tested Configurations

| Component   | Version                                                              |
|-------------|----------------------------------------------------------------------|
| OS          | [Ubuntu 24.04.2], [Ubuntu 25.10]                                     |
| ROCm        | [7.2.0][ROCm 7.2.0]                                                  |
| XDNA Driver | [1.6][XDNA Driver 1.6]                                               |
| MLIR-AIE    | [1.2.1][MLIR-AIE 1.2.1]                                              |

[Ubuntu 24.04.2]: https://releases.ubuntu.com/noble/
[Ubuntu 25.10]: https://releases.ubuntu.com/questing/
[ROCm 7.2.0]: https://rocm.docs.amd.com/en/docs-7.2.0/
[XDNA Driver 1.6]: https://github.com/amd/xdna-driver/tree/1.6
[MLIR-AIE 1.2.1]: https://github.com/Xilinx/mlir-aie/tree/v1.2.1

### ROCm

`ggml-hsa` requires [ROCm](https://github.com/ROCm/rocm-systems) 7.1.1 or newer. See the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

Due to ongoing NPU support work in [ROCR](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime), it is recommended to compile the latest ROCR from source. Commit [`863ffc1`](https://github.com/ROCm/rocm-systems/commit/863ffc1c07cf56567101fff2c39b66efb4cdb579) is confirmed working.

### AMD XDNA Driver

`ggml-hsa` depends on the [AMD XDNA Driver](https://github.com/amd/xdna-driver). Installation instructions:

- Via IRON: [build_drivers.sh](https://github.com/Xilinx/mlir-aie/blob/main/utils/build_drivers.sh)
- Direct: [xdna-driver README](https://github.com/amd/xdna-driver#linux-compilation-and-installation)

### MLIR-AIE (IRON)

`ggml-hsa` supports JIT compilation via the [IRON framework](https://github.com/Xilinx/mlir-aie).

Install dependencies:

```bash
python3 -m pip install -r src/ggml-hsa/requirements.txt
```

Or use the setup script to create a virtual environment:

```bash
source src/ggml-hsa/env_setup.sh
```

> **Note:** IRON environments consume considerable storage. For pre-generated kernels, set `GGML_HSA_KERNEL_DIR` and disable JIT at compile time.

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

| Variable                      | Description                                                     |
|-------------------------------|-----------------------------------------------------------------|
| `GGML_HSA_ENABLE_LOG`         | Enable internal logging (`1`, `true`, or `on`)                  |
| `GGML_HSA_KERNEL_DIR`         | Precompiled kernel directory path                               |
| `GGML_HSA_KERNEL_CACHE_DIR`   | JIT cache directory                                             |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | Clear JIT cache on startup (`1`, `true`, or `on`)               |
| `GGML_HSA_JIT_VERBOSE`        | Verbose JIT output (`1`, `true`, or `on`)                       |
