# GGML HSA Backend

## Compile Options

HSA backend compile options:
| Option                         | Description                                                                                                         |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `GGML_HSA`                     | Enable HSA backend.                                                                                                 |
| `GGML_HSA_JIT_COMPILE`         | Enable JIT compilation of kernels. Requires an IRON environment to be active.                                       |
| `GGML_HSA_CPU_FALLBACK`        | Enable fallback to  for HSA backend. This is an option to be used only during development, as the CPU fallback requires additional synchronization and has significant overhead.                                                                                                          |

## Environment Variables

HSA backend environment variables:
| Variable                      | Description                                                                     |
|-------------------------------|---------------------------------------------------------------------------------|
| `GGML_HSA_JIT_CLEAR_CACHE`    | If set to `1`, `true`, or `TRUE`, clear JIT compiled kernel cache.              |
| `GGML_HSA_JIT_VERBOSE`        | If set to `1`, `true`, or `TRUE`, enable verbose output during JIT compilation. |

## JIT Compilation

The HSA backend supports JIT compilation of kernels. This requires an IRON environment to be active. The JIT compilation process generates kernels on-the-fly from the installed IRON kernel sources.

JIT generated kernels are cached in `XDG_CACHE_HOME` or if not defined in `$HOME/.cache/ggml`. One can clear the cache by setting the environment variable `GGML_HSA_JIT_CLEAR_CACHE` to something other than `0`.
