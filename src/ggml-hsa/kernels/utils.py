def suppress_import_pyxrt_msg():
    """Function to suppress pyxrt not found message."""

    # suppress stderr from aie imports until https://github.com/Xilinx/mlir-aie/issues/2833
    # is resolved
    import os
    import sys
    import contextlib

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stderr(devnull):
            import aie.utils
