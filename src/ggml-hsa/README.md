# GGML HSA Backend

## JIT Compilation

The HSA backend supports JIT compilation of kernels. This requires an IRON environment to be active which can be set up using
```bash
$ source env_setup.sh
```

The JIT compilation process generates kernels on-the-fly from the installed IRON kernel sources. If `GGML_HSA_KERNEL_DIR` is set, any kernels found there take precedence over JIT compiled kernels.

JIT generated kernels are cached in a directory in the following order of precedence:
1. The directory specified by the environment variable `GGML_HSA_KERNEL_CACHE_DIR`, or
2. the `${XDG_CACHE_HOME}.ggml/` directory if `XDG_CACHE_HOME` is defined, or
3. in `$HOME/.cache/ggml` if `$HOME` is defined, or
4. in `/tmp/ggml/ggml-hsa` otherwise.

One can clear the cache by setting the environment variable `GGML_HSA_KERNEL_CACHE_CLEAR` to an appropriate value.

***WARNING:*** **Setting `GGML_HSA_KERNEL_CACHE_CLEAR` will delete all files in the cache directory.**

## NPU Supported Datatypes

The following data types are supported by AIEs:
| Type             | Comment                                |
|------------------|----------------------------------------|
| `GGML_TYPE_I8`   | Native datatype.                       |
| `GGML_TYPE_I16`  | Native datatype.                       |
| `GGML_TYPE_I32`  | Native datatype.                       |
| `GGML_TYPE_BF16` | Native datatype.                       |
| `GGML_TYPE_F32`  | Emulated, so slower than native types. |

## CMake Build Options

The following CMake build options are supported:
| Option                  | Description                                                                                                         |
|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| `GGML_HSA`              | Enable HSA backend.                                                                                                 |
| `GGML_HSA_JIT_COMPILE`  | Enable JIT compilation of kernels. Requires an IRON environment to be active.                                       |

## Environment Variables

The following environment variables are supported:
| Variable                      | Description                                                                        |
|-------------------------------|------------------------------------------------------------------------------------|
| `GGML_HSA_KERNEL_DIR`         | Precompiled kernel directory path.                                                 |
| `GGML_HSA_KERNEL_CACHE_DIR`   | Cached kernel directory, populated by JIT kernel compilation.                      |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | If set to `1`, `true`, or `TRUE`, remove all files in the cached kernel directory. |
| `GGML_HSA_JIT_VERBOSE`        | If set to `1`, `true`, or `TRUE`, enable verbose output during JIT compilation.    |
