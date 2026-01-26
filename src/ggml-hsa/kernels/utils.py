def suppress_import_pyxrt_msg():
    """Function to suppress pyxrt not found message."""

    # suppress stderr from aie imports until https://github.com/Xilinx/mlir-aie/issues/2833
    # is resolved
    import os
    import contextlib

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull):
            import aie.utils
import os
import contextlib

# suppress stderr from aie imports until https://github.com/Xilinx/mlir-aie/issues/2833
# is resolved
with open(os.devnull, "w", encoding="utf-8") as _ggml_hsa_devnull:
    with contextlib.redirect_stderr(_ggml_hsa_devnull):
        import aie.utils as _ggml_hsa_aie_utils


def suppress_import_pyxrt_msg():
    """Return the pre-imported aie.utils module with pyxrt message suppressed.

    The aie.utils module is imported once at module import time with stderr
    suppressed to avoid noisy pyxrt not found messages. This function is
    retained for backward compatibility and simply returns the cached module.
    """

    return _ggml_hsa_aie_utils
