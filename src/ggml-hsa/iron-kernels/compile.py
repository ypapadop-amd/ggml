# compile.py -*- Python -*-

import argparse
import subprocess
import os
import sys


def compile_mlir(
    kernel_source: str, kernel_compile_args: str, mlir_filename: str, output_dir: str
):
    """
    Creates an MLIR file from kernel_source.
    """
    output_path = os.path.join(output_dir, mlir_filename)
    cmd = [sys.executable, kernel_source] + list(kernel_compile_args.split())
    with open(output_path, "w", encoding="utf-8") as output_file:
        try:
            subprocess.run(
                cmd,
                cwd=output_dir,
                check=True,
                stdout=output_file,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError("MLIR Compilation failed") from ex


def compile_pdi(
    mlir_filename: str, pdi_filename: str, insts_filename: str, output_dir: str
):
    """
    Compile an MLIR file to instruction and xclbin files using aiecc.py.

    Parameters:
        mlir_file (str): MLIR input file.
        pdi_file (str): PDI output file.
        insts_file (str): Instructions output file.
    """

    mlir_path = os.path.join(output_dir, mlir_filename)
    pdi_output_path = os.path.join(output_dir, pdi_filename)
    insts_output_path = os.path.join(output_dir, insts_filename)
    cmd = [
        "aiecc.py",
        "--alloc-scheme=basic-sequential",
        "--aie-generate-pdi",
        "--aie-generate-npu-insts",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        f"--pdi-name={pdi_output_path}",
        f"--npu-insts-name={insts_output_path}",
        f"{mlir_path}",
    ]
    try:
        subprocess.run(
            cmd,
            cwd=output_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as ex:
        raise RuntimeError("aiecc Compilation failed") from ex


def file_path(string: str):
    """
    Checks if a string is an existing file.
    """

    if not os.path.isfile(string):
        raise FileNotFoundError(string)
    return string


def main():
    parser = argparse.ArgumentParser(
        prog="compile.py",
        description="Compiles IRON kernels",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Kernel name",
    )
    parser.add_argument(
        "--kernel_source",
        type=file_path,
        required=True,
        help="Kernel source",
    )
    parser.add_argument(
        "--kernel_compile_args",
        type=str,
        required=True,
        help="Kernel source arguments",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    mlir_filename = f"{args.name}.mlir"

    compile_mlir(
        kernel_source=args.kernel_source,
        kernel_compile_args=args.kernel_compile_args,
        mlir_filename=mlir_filename,
        output_dir=args.output_directory,
    )

    compile_pdi(
        mlir_filename=mlir_filename,
        pdi_filename=f"{args.name}.pdi",
        insts_filename=f"{args.name}_insts.bin",
        output_dir=args.output_directory,
    )


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
