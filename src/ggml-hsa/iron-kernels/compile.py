# compile.py -*- Python -*-

import argparse
import subprocess
import os
import sys


def compile_mlir_to_binary(
    mlir_path: str, inst_filename: str, xclbin_filename: str, debug: bool = False
):
    """
    Compile an MLIR file to instruction and xclbin files using aiecc.py.

    Parameters:
        mlir_path (str): Path to the MLIR input file.
        inst_filename (str): Name of the instruction binary file (e.g., 'inst.bin').
        xclbin_filename (str): Name of the xclbin file (e.g., 'final.xclbin').
        debug (bool): If True, print the commands being executed. Default is False.
    """

    mlir_dir = os.path.dirname(os.path.abspath(mlir_path))

    cmd = [
        "aiecc.py",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--xclbin-name={xclbin_filename}",
        f"--npu-insts-name={inst_filename}",
        "aie.mlir",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=mlir_dir,
            check=True,
            stdout=sys.stdout if debug else subprocess.DEVNULL,
            stderr=sys.stderr if debug else subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[aiecc] Compilation failed:\n{e}")

def file_path(string):
    if not os.path.isfile(string):
        raise NotADirectoryError(string)
    return string


def main():
    parser = argparse.ArgumentParser(
        prog="compile.py",
        description="Compiles IRON kernels",
    )
    parser.add_argument(
        "--kernel_output_name",
        type=str,
        required=True,
        help="Kernel output name",
    )
    parser.add_argument(
        "--kernel_source",
        type=file_path,
        required=True,
        help="Kernel source",
    )
    parser.add_argument(
        "--kernel_compile_args",
        type=file_path,
        required=True,
        help="Kernel source arguments",
    )
    parser.add_argument(
        "--peano_source",
        type=file_path,
        required=True,
        help="Peano kernel source",
    )
    parser.add_argument(
        "--peano_output",
        type=file_path,
        required=True,
        help="Peano output file",
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
