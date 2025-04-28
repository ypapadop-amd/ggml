# compiler.py -*- Python -*-

import argparse
import subprocess
import os
import sys


def compile_mlir(
    kernel_source: str,
    kernel_compile_args: str,
    mlir_filename: str,
    output_directory: str,
):
    """
    Compiles the IRON kernel source to MLIR.

    Parameters:
        kernel_source (str): Kernel source file.
        kernel_compile_args (str): Compile arguments.
        mlir_filename (str): MLIR output file.
        output_directory (str): Output and working directory.
    """
    output_path = os.path.join(output_directory, mlir_filename)
    cmd = [sys.executable, kernel_source] + list(kernel_compile_args.split())
    with open(output_path, "w", encoding="utf-8") as output_file:
        try:
            subprocess.run(
                cmd,
                cwd=output_directory,
                check=True,
                stdout=output_file,
                stderr=sys.stderr,
            )
        except subprocess.CalledProcessError as ex:
            raise RuntimeError("MLIR Compilation failed") from ex


def compile_pdi(
    mlir_filename: str, pdi_filename: str, insts_filename: str, output_directory: str
):
    """
    Compile an MLIR file to PDI and instruction using aiecc.

    Parameters:
        mlir_filename (str): MLIR input file.
        pdi_filename (str): PDI output file.
        insts_filename (str): Instructions output file.
        output_directory (str): Output and working directory.
    """

    mlir_path = os.path.join(output_directory, mlir_filename)
    pdi_output_path = os.path.join(output_directory, pdi_filename)
    insts_output_path = os.path.join(output_directory, insts_filename)
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
            cwd=output_directory,
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


def compile_kernel(
    name: str,
    kernel_source: str,
    kernel_compile_args: str,
    output_directory: str,
):
    """
    Compiles the kernel code to PDI and instruction files.
    """

    mlir_filename = f"{name}.mlir"

    os.makedirs(output_directory, exist_ok=True)

    compile_mlir(
        kernel_source=kernel_source,
        kernel_compile_args=kernel_compile_args,
        mlir_filename=mlir_filename,
        output_directory=output_directory,
    )

    compile_pdi(
        mlir_filename=mlir_filename,
        pdi_filename=f"{name}.pdi",
        insts_filename=f"{name}_insts.bin",
        output_directory=output_directory,
    )


def main():
    """
    Main function for use during AOT compilation.
    """

    parser = argparse.ArgumentParser(
        prog="compiler.py",
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

    compile_kernel(
        name=args.name,
        kernel_source=args.kernel_source,
        kernel_compile_args=args.kernel_compile_args,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()
