# GGML HSA Backend

## Kernel cache

JIT generated kernels are cached in `XDG_CACHE_HOME` or if not defined in `$HOME/.cache/ggml`. One can clear the cache by setting the environment variable `GGML_HSA_CLEAR_KERNEL_CACHE_HSA_CLEAR_CACHE` to something else than `0`.
