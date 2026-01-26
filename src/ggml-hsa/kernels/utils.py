def suppress_import_pyxrt_msg():
    """Function to suppress pyxrt not found message."""

    # suppress stderr from aie imports until https://github.com/Xilinx/mlir-aie/issues/2833
    # is resolved
    import os
    import sys

    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

    import aie.utils

    sys.stderr = stderr_backup
