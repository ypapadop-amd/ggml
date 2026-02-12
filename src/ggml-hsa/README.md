# GGML HSA (ggml-hsa) Backend

## Supported Devices

The GGML HSA (`ggml-hsa`) backend supports the following AMD XDNA NPUs:

- [AMD XDNA](https://www.amd.com/en/technologies/xdna.html) (`aie2` architecture), e.g., Phoenix, Hawk Point.
- [AMD XDNA2](https://www.amd.com/en/technologies/xdna.html#xdna2) (`aie2p` architecture), e.g., Strix Halo, Krackan.

## Supported Datatypes

The following data types are supported by the HSA backend:

| Type             | Comment                                       |
|------------------|-----------------------------------------------|
| `GGML_TYPE_I8`   | Native `aie2` / `aie2p` datatype.             |
| `GGML_TYPE_I16`  | Native `aie2` / `aie2p` datatype.             |
| `GGML_TYPE_I32`  | Native `aie2` / `aie2p` datatype.             |
| `GGML_TYPE_BF16` | Native `aie2` / `aie2p` datatype.             |
| `GGML_TYPE_F16`  | Supported via conversion to `GGML_TYPE_BF16`. |
| `GGML_TYPE_F32`  | Emulated, so slower than native types.        |

## Prerequisites

### Supported Configurations

The following configurations have been tested and confirmed working:

- OS: [Ubuntu 24.04.2](https://releases.ubuntu.com/noble/), [Ubuntu 25.04](https://releases.ubuntu.com/plucky/)
- ROCm: [7.1.1](https://rocm.docs.amd.com/en/docs-7.1.1/)
- XDNA Driver: [1.6](https://github.com/amd/xdna-driver/tree/1.6)
- MLIR-AIE: [1.2.1](https://github.com/Xilinx/mlir-aie/tree/v1.2.1)

### ROCm

`ggml-hsa` requires a fairly new [ROCm](https://github.com/ROCm/rocm-systems) (development happens using ROCm 7.1.1). Installation instructions are [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

Due to the ongoing work for supporting NPUs in the ROCm Runtime, [ROCR](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime), it is recommended that the latest ROCR should be compiled source. Commit [863ffc1](https://github.com/ROCm/rocm-systems/commit/863ffc1c07cf56567101fff2c39b66efb4cdb579) is confirmed working.

### AMD XDNA™ Driver

`ggml-hsa` depends on the [AMD XDNA™ Driver](https://github.com/amd/xdna-driver). You can find the installation instructions as part of IRON [here](https://github.com/Xilinx/mlir-aie/blob/main/utils/build_drivers.sh) or directly from the driver repo [here](https://github.com/amd/xdna-driver?tab=readme-ov-file#linux-compilation-and-installation).

### MLIR-AIE

`ggml-hsa` supports JIT compilation of kernels for the generation of optimized kernels for the target architecture at runtime. This is supported via integration with the [IRON framework](https://github.com/Xilinx/mlir-aie), a Python-based solution for creating AIE kernels.

An IRON environment must be created to compile GGML in by installing the necessary dependencies:

```bash
python3 -m pip install -r src/ggml-hsa/requirements.txt
```

Alternatively, one can use the [env_setup.sh](./env_setup.sh) script to set up a Python virtual environment; the script will create a Python virtual environment, activate it, and install the necessary dependencies.

```bash
source ./env_setup.sh
```

> **Note** : An IRON environment can consume considerable storage space and JIT compilation is expensive. If kernels are already generated, they can be placed in a directory specified by the environment variable `GGML_HSA_KERNEL_DIR` and JIT can be disabled at compile time; in that case `ggml-hsa` will use the already generated kernels.

## HSA Build

To build GGML with HSA support, enable the `GGML_HSA=ON` CMake option. If JIT compilation is desired, also enable the `GGML_HSA_JIT_COMPILE=ON` CMake option.

If a different ROCR installation is preferred, you can point to it by setting the `hsa-runtime64_DIR` CMake variable; this will require the ROCR shared library directory to be in your LD_LIBRARY_PATH before any installed ROCm libraries.

```bash
cmake -S . -B build -DGGML_HSA=ON -Dhsa-runtime64_DIR=/path/to/rocm/lib/cmake/hsa-runtime64 -DGGML_HSA_JIT_COMPILE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j
```

If GPU acceleration via HIP is also desired, you can combine the above with the HIP compilation options:

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1102 \
  -DGGML_HSA=ON -Dhsa-runtime64_DIR=/path/to/rocm/lib/cmake/hsa-runtime64 -DGGML_HSA_JIT_COMPILE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j
```

## JIT Compilation

The JIT compilation process generates kernels on-the-fly from the installed kernel sources. If `GGML_HSA_KERNEL_DIR` is set, any kernels found there take precedence over JIT compiled kernels.

JIT generated kernels are cached in a directory in the following order of precedence:

1. The directory specified by the environment variable `GGML_HSA_KERNEL_CACHE_DIR`, or
2. the `${XDG_CACHE_HOME}.ggml/` directory if `XDG_CACHE_HOME` is defined, or
3. in `$HOME/.cache/ggml` if `$HOME` is defined, or
4. in `/tmp/ggml/ggml-hsa` otherwise.

One can clear the cache by setting the environment variable `GGML_HSA_KERNEL_CACHE_CLEAR` to an appropriate value.

> **WARNING:** Setting `GGML_HSA_KERNEL_CACHE_CLEAR` will delete all the files in the cache directory.

## CMake Build Options

The following CMake build options are supported:

| Option | Description |
| ----- | ----- |
| `GGML_HSA` | Enable HSA backend. |
| `GGML_HSA_JIT_COMPILE` | Enable JIT compilation of kernels. Requires an IRON environment to be present. |

## Environment Variables

The following environment variables are supported:

| Variable                      | Description                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------|
| `GGML_HSA_ENABLE_LOG`         | If set to `1`, `true`, or `on` enable internal logging. Off by default for release builds. |
| `GGML_HSA_KERNEL_DIR`         | Precompiled kernel directory path.                                                         |
| `GGML_HSA_KERNEL_CACHE_DIR`   | Cached kernel directory, populated by JIT kernel compilation.                              |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | If set to `1`, `true`, or `on` remove all files in the cached kernel directory.            |
| `GGML_HSA_JIT_VERBOSE`        | If set to `1`, `true`, or `on`, enable verbose output during JIT compilation.              |
