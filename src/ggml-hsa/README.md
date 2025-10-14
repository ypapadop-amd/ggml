# GGML HSA (ggml-hsa) Backend

## Supported Devices

The GGML HSA (`ggml-hsa`) backend supports the following devices:
- [AMD XDNA](https://www.amd.com/en/technologies/xdna.html) (`aie2` architecture), e.g., Phoenix, Hawk Point.

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

### Tested Configurations

The following configurations have been tested and confirmed working:
- OS: [Ubuntu 25.04](https://releases.ubuntu.com/plucky/)
- ROCm: [7.0.2](https://rocm.docs.amd.com/en/docs-7.0.2/)
- XDNA Driver: [1.6](https://github.com/amd/xdna-driver/tree/1.6)
- MLIR-AIE: [latest](https://github.com/Xilinx/mlir-aie/commit/083064591d1678e194f03c8b185339a2cf392b89)

### ROCm

`ggml-hsa` requires a fairly new [ROCR](https://github.com/ROCm/rocm-systems) (development happens using ROCm 7.0.2). Installation instructions are [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

### AMD XDNA™ Driver

`ggml-hsa` depends on the [AMD XDNA™ Driver](https://github.com/amd/xdna-driver). You can find the installation instructions [here](https://github.com/amd/xdna-driver?tab=readme-ov-file#linux-compilation-and-installation).

### MLIR-AIE

`ggml-hsa` supports JIT compilation of kernels for the generation of optimized kernels for the target architecture at runtime. This is supported via integration with the [IRON framework](https://github.com/Xilinx/mlir-aie), a Python-based solution for creating AIE kernels.

An IRON environment must be created to compile GGML in by installing the necessary dependencies:
```bash
MLIR_PYTHON_EXTRAS_SET_VERSION="0.0.8.3" HOST_MLIR_PYTHON_PACKAGE_PREFIX="aie" \
python3 -m pip install -r ${SCRIPT_DIR_NAME}/requirements.txt
```

Alternatively, one can use the [env_setup.sh](./env_setup.sh) script to set up a Python virtual environment; the script will create a Python virtual environment, activate it, and install the necessary dependencies.
```bash
source ./env_setup.sh
```

> **Note** : An IRON environment can consume considerable storage space and JIT compilation is expensive. If kernels are already generated, they can be placed in a directory specified by the environment variable `GGML_HSA_KERNEL_DIR` and JIT can be disabled at compile time.

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
| Option                  | Description                                                                                                         |
|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| `GGML_HSA`              | Enable HSA backend.                                                                                                 |
| `GGML_HSA_JIT_COMPILE`  | Enable JIT compilation of kernels. Requires an IRON environment to be active.                                       |

## Environment Variables

The following environment variables are supported:
| Variable                      | Description                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------|
| `GGML_HSA_ENABLE_LOG`         | If set to `1`, `true`, or `on` enable internal logging. Off by default for release builds. |
| `GGML_HSA_KERNEL_DIR`         | Precompiled kernel directory path.                                                         |
| `GGML_HSA_KERNEL_CACHE_DIR`   | Cached kernel directory, populated by JIT kernel compilation.                              |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | If set to `1`, `true`, or `on` remove all files in the cached kernel directory.            |
| `GGML_HSA_JIT_VERBOSE`        | If set to `1`, `true`, or `on`, enable verbose output during JIT compilation.              |
