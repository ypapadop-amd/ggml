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

### AMD XDNA™ Driver

`ggml-hsa` depends on the [AMD XDNA™ Driver](https://github.com/amd/xdna-driver). You can find the installation instructions [here](https://github.com/amd/xdna-driver?tab=readme-ov-file#linux-compilation-and-installation).


### ROCm

`ggml-hsa` requires a fairly new [ROCR](https://github.com/ROCm/rocm-systems) (development happens using ROCm 7.0.1). Installation instructions are [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

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

## Kernel Pregeneration

For environments where JIT compilation is not available (e.g., lacking IRON dependencies), kernels can be pregenerated using the `ggml-hsa-gen-kernels` tool. This tool reads a configuration file and generates precompiled kernels that can be used at runtime.

> **Note:** The pregeneration tool requires an IRON environment to compile the kernels. However, once kernels are pregenerated, they can be used in environments without IRON.

### Usage

```bash
# Pregenerate kernels (requires IRON environment)
ggml-hsa-gen-kernels \
    --config example-kernel-config.json \
    --output-dir /path/to/precompiled/kernels \
    --verbose
```

After building with CMake, the tool is installed as `ggml-hsa-gen-kernels` in the bin directory. Example configuration files are installed in `share/ggml-hsa/examples/`.

**Typical Workflow:**

1. Set up IRON environment on a machine with compilation tools:
   ```bash
   source /path/to/ggml/src/ggml-hsa/env_setup.sh
   ```

2. Pregenerate kernels using the configuration file:
   ```bash
   ggml-hsa-gen-kernels \
       --config /usr/local/share/ggml-hsa/examples/example-kernel-config.json \
       --output-dir ./precompiled_kernels \
       --verbose
   ```

3. Copy the precompiled kernels to the target system (without IRON):
   ```bash
   scp -r ./precompiled_kernels target-system:/opt/ggml-hsa/kernels
   ```

4. On the target system, set the environment variable:
   ```bash
   export GGML_HSA_KERNEL_DIR=/opt/ggml-hsa/kernels
   ```

5. Run your application - it will use the precompiled kernels without JIT compilation.

### Configuration File Format

The configuration file is a JSON file specifying the kernels to generate:

```json
{
    "kernels": [
        {
            "kernel_name": "ggml_op_add",
            "kernel_source": "binary_ops.py",
            "arch": "aie2",
            "input_tensors": [
                "(1024,1,1,1)/f32",
                "(1024,1,1,1)/f32"
            ],
            "output_tensor": "(1024,1,1,1)/f32",
            "exported_name": "add_f32_1024"
        }
    ]
}
```

**Tensor Description Format:** `(dim0,dim1,dim2,dim3)/dtype[/(stride0,stride1,stride2,stride3)]`

**Supported dtypes:** `f32`, `f16`, `bf16`, `i8`, `i16`, `i32`

**Example:** `(1024,768,1,1)/bf16`

### Using Pregenerated Kernels

Set the `GGML_HSA_KERNEL_DIR` environment variable to the output directory:

```bash
export GGML_HSA_KERNEL_DIR=/path/to/precompiled/kernels
```

The backend will automatically use pregenerated kernels from this directory, avoiding JIT compilation at runtime.

The output directory structure will be:
```
/path/to/precompiled/kernels/
└── aie2/                          # Architecture-specific directory
    ├── add_f32_1024.pdi           # PDI file for kernel
    ├── add_f32_1024_insts.bin     # Instructions binary for kernel
    ├── mul_f32_1024.pdi
    ├── mul_f32_1024_insts.bin
    └── ...
```

See [example-kernel-config.json](./example-kernel-config.json) and [minimal-kernel-config.json](./minimal-kernel-config.json) for complete example configuration files.

## Environment Variables

The following environment variables are supported:
| Variable                      | Description                                                                                |
|-------------------------------|--------------------------------------------------------------------------------------------|
| `GGML_HSA_ENABLE_LOG`         | If set to `1`, `true`, or `on` enable internal logging. Off by default for release builds. |
| `GGML_HSA_KERNEL_DIR`         | Precompiled kernel directory path.                                                         |
| `GGML_HSA_KERNEL_CACHE_DIR`   | Cached kernel directory, populated by JIT kernel compilation.                              |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | If set to `1`, `true`, or `on` remove all files in the cached kernel directory.            |
| `GGML_HSA_JIT_VERBOSE`        | If set to `1`, `true`, or `on`, enable verbose output during JIT compilation.              |
