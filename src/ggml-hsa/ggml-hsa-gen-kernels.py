#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Tool to pregenerate HSA kernels from a configuration file.

This tool reads a configuration file specifying kernel parameters and generates
precompiled kernels that can be used with the GGML_HSA_KERNEL_DIR environment
variable, avoiding the need for JIT compilation at runtime.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def setup_logging(verbose: bool):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def validate_config(config: dict) -> bool:
    """
    Validate the configuration file structure.
    
    Parameters:
        config (dict): Configuration dictionary to validate.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if 'kernels' not in config:
        logging.error("Configuration must contain 'kernels' key")
        return False
    
    if not isinstance(config['kernels'], list):
        logging.error("'kernels' must be a list")
        return False
    
    for i, kernel in enumerate(config['kernels']):
        if 'kernel_name' not in kernel:
            logging.error(f"Kernel {i} missing 'kernel_name'")
            return False
        if 'kernel_source' not in kernel:
            logging.error(f"Kernel {i} missing 'kernel_source'")
            return False
        if 'arch' not in kernel:
            logging.error(f"Kernel {i} missing 'arch'")
            return False
        if 'input_tensors' not in kernel:
            logging.error(f"Kernel {i} missing 'input_tensors'")
            return False
        if 'output_tensor' not in kernel:
            logging.error(f"Kernel {i} missing 'output_tensor'")
            return False
        if 'exported_name' not in kernel:
            logging.error(f"Kernel {i} missing 'exported_name'")
            return False
    
    return True


def parse_tensor_desc(tensor_str: str):
    """
    Parse tensor description string into a TensorDesc.
    
    Format: "(shape0,shape1,shape2,shape3)/dtype[/(stride0,stride1,stride2,stride3)]"
    Example: "(1024,1,1,1)/f32" or "(1024,768,1,1)/bf16/(1,1024,786432,786432)"
    
    Parameters:
        tensor_str (str): Tensor description string.
        
    Returns:
        TensorDesc: Parsed tensor description.
    """
    from kernels.tensor_desc import tensordesc
    
    parts = tensor_str.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid tensor descriptor: {tensor_str} (expected format: (dim0,dim1,dim2,dim3)/dtype)")
    
    # Parse shape - must be enclosed in parentheses
    shape_part = parts[0].strip()
    if not shape_part.startswith('(') or not shape_part.endswith(')'):
        raise ValueError(f"Invalid shape format in {tensor_str}: shape must be enclosed in parentheses")
    
    shape_str = shape_part.strip('()')
    try:
        shape = tuple(int(x.strip()) for x in shape_str.split(','))
    except ValueError as e:
        raise ValueError(f"Invalid shape values in {tensor_str}: {e}")
    
    if len(shape) != 4:
        raise ValueError(f"Shape must have exactly 4 dimensions, got {len(shape)} in {tensor_str}")
    
    # Parse dtype
    dtype = parts[1].strip()
    if not dtype:
        raise ValueError(f"Invalid tensor descriptor: {tensor_str} (missing dtype)")
    
    # Parse optional stride
    stride = None
    contiguous = True
    if len(parts) >= 3:
        stride_part = parts[2].strip()
        if not stride_part.startswith('(') or not stride_part.endswith(')'):
            raise ValueError(f"Invalid stride format in {tensor_str}: stride must be enclosed in parentheses")
        
        stride_str = stride_part.strip('()')
        try:
            stride = tuple(int(x.strip()) for x in stride_str.split(','))
        except ValueError as e:
            raise ValueError(f"Invalid stride values in {tensor_str}: {e}")
        
        if len(stride) != 4:
            raise ValueError(f"Stride must have exactly 4 dimensions, got {len(stride)} in {tensor_str}")
        contiguous = False
    
    return tensordesc(dtype=dtype, shape=shape, stride=stride, contiguous=contiguous)


def generate_kernels(config_file: str, output_dir: str, kernels_dir: str, verbose: bool):
    """
    Generate kernels from configuration file.
    
    Parameters:
        config_file (str): Path to configuration file.
        output_dir (str): Output directory for generated kernels.
        kernels_dir (str): Directory containing kernel source files.
        verbose (bool): Enable verbose output.
    """
    # Add kernels directory to Python path
    sys.path.insert(0, kernels_dir)
    
    try:
        from kernels.build import compile_kernel
    except ImportError as e:
        logging.error(f"Failed to import kernel build module: {e}")
        logging.error(f"Make sure kernels directory is at: {kernels_dir}")
        return 1
    
    # Load configuration
    logging.info(f"Loading configuration from {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1
    
    if not validate_config(config):
        return 1
    
    # Process each kernel
    total_kernels = len(config['kernels'])
    logging.info(f"Generating {total_kernels} kernel(s)")
    
    success_count = 0
    for i, kernel_config in enumerate(config['kernels'], 1):
        kernel_name = kernel_config['kernel_name']
        logging.info(f"[{i}/{total_kernels}] Processing kernel: {kernel_name}")
        
        try:
            # Parse tensor descriptions
            input_tensors = [parse_tensor_desc(t) for t in kernel_config['input_tensors']]
            output_tensor = parse_tensor_desc(kernel_config['output_tensor'])
            
            # Resolve kernel source path
            kernel_source = Path(kernels_dir) / kernel_config['kernel_source']
            if not kernel_source.exists():
                logging.error(f"Kernel source not found: {kernel_source}")
                continue
            
            # Generate output directory
            arch = kernel_config['arch']
            exported_name = kernel_config['exported_name']
            kernel_output_dir = Path(output_dir) / arch
            
            # Compile kernel
            compile_kernel(
                kernel_name=kernel_name,
                kernel_source=str(kernel_source),
                arch=arch,
                input_tensors=input_tensors,
                output_tensor=output_tensor,
                exported_name=exported_name,
                output_directory=str(kernel_output_dir),
                verbose=verbose
            )
            
            success_count += 1
            logging.info(f"Successfully generated kernel: {exported_name}")
            
        except Exception as e:
            logging.error(f"Failed to generate kernel {kernel_name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    logging.info(f"Generated {success_count}/{total_kernels} kernel(s) successfully")
    return 0 if success_count == total_kernels else 1


def main():
    """Main entry point for the kernel pregeneration tool."""
    parser = argparse.ArgumentParser(
        prog='ggml-hsa-gen-kernels',
        description='Pregenerate HSA kernels from configuration file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Format:
    The configuration file should be a JSON file with the following structure:
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

Tensor Description Format:
    Shape and dtype: "(dim0,dim1,dim2,dim3)/dtype"
    With stride: "(dim0,dim1,dim2,dim3)/dtype/(stride0,stride1,stride2,stride3)"
    
    Supported dtypes: f32, f16, bf16, i8, i16, i32
    Example: "(1024,768,1,1)/bf16"

Environment Variables:
    The generated kernels can be used by setting:
        export GGML_HSA_KERNEL_DIR=<output-directory>
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to kernel configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for generated kernels'
    )
    
    parser.add_argument(
        '--kernels-dir',
        type=str,
        help='Directory containing kernel source files (default: ./kernels)',
        default=None
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Determine kernels directory
    if args.kernels_dir:
        kernels_dir = args.kernels_dir
    else:
        # Try to find kernels directory relative to this script
        script_dir = Path(__file__).parent
        kernels_dir = str(script_dir / 'kernels')
    
    if not os.path.isdir(kernels_dir):
        logging.error(f"Kernels directory not found: {kernels_dir}")
        logging.error("Please specify --kernels-dir or ensure kernels are in ./kernels")
        return 1
    
    logging.info(f"Using kernels directory: {kernels_dir}")
    
    return generate_kernels(args.config, args.output_dir, kernels_dir, args.verbose)


if __name__ == '__main__':
    sys.exit(main())
