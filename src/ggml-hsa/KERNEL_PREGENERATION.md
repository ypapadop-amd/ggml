# HSA Kernel Pregeneration Tool

## Overview

The `ggml-hsa-gen-kernels` tool allows pregeneration of HSA kernels from a configuration file. This is essential for deployment environments where:

1. JIT compilation dependencies (IRON framework) are not available
2. Compilation at runtime is too slow or resource-intensive
3. Known kernel configurations are being used repeatedly

## Quick Start

### Prerequisites

- IRON environment (only for pregeneration, not for using pregenerated kernels)
- ROCm installation
- Python 3 with required dependencies

### Basic Usage

```bash
# On development machine with IRON
ggml-hsa-gen-kernels \
    --config my-kernels.json \
    --output-dir ./precompiled \
    --verbose

# Copy to production machine
export GGML_HSA_KERNEL_DIR=./precompiled

# Run application - kernels are loaded from disk, no JIT compilation
./my-application
```

## Configuration File Format

### Structure

```json
{
    "kernels": [
        {
            "kernel_name": "ggml_op_add",
            "kernel_source": "binary_ops.py",
            "arch": "aie2",
            "input_tensors": ["(1024,1,1,1)/f32", "(1024,1,1,1)/f32"],
            "output_tensor": "(1024,1,1,1)/f32",
            "exported_name": "add_f32_1024"
        }
    ]
}
```

### Field Descriptions

- **kernel_name**: Name of the kernel function (must match function in source)
- **kernel_source**: Python file containing the kernel definition
- **arch**: Target architecture (e.g., "aie2", "aie2p")
- **input_tensors**: Array of input tensor descriptions
- **output_tensor**: Output tensor description
- **exported_name**: Name for the generated kernel files

### Tensor Description Format

Format: `(dim0,dim1,dim2,dim3)/dtype[/(stride0,stride1,stride2,stride3)]`

**Required fields:**
- Shape: 4 comma-separated dimensions in parentheses
- Dtype: Data type specifier

**Optional fields:**
- Stride: 4 comma-separated stride values in parentheses (for non-contiguous tensors)

**Supported dtypes:**
- `f32` - 32-bit floating point
- `f16` - 16-bit floating point
- `bf16` - BFloat16
- `i8` - 8-bit integer
- `i16` - 16-bit integer
- `i32` - 32-bit integer

**Examples:**
```
(1024,1,1,1)/f32                           # Simple contiguous tensor
(1024,768,1,1)/bf16                        # 2D tensor
(1024,1,1,1)/f32/(1,1024,1048576,1048576)  # Non-contiguous with stride
```

## Available Kernels

The following kernel sources are available:

### Binary Operations (`binary_ops.py`)
- `ggml_op_add` - Element-wise addition
- `ggml_op_sub` - Element-wise subtraction
- `ggml_op_mul` - Element-wise multiplication
- `ggml_op_div` - Element-wise division

### Unary Operations (`unary_ops.py`)
- `ggml_unary_op_abs` - Absolute value
- `ggml_unary_op_sgn` - Sign function
- `ggml_unary_op_neg` - Negation
- `ggml_unary_op_step` - Step function
- `ggml_unary_op_tanh` - Hyperbolic tangent
- `ggml_unary_op_elu` - Exponential Linear Unit
- `ggml_unary_op_relu` - Rectified Linear Unit
- `ggml_unary_op_sigmoid` - Sigmoid activation
- `ggml_unary_op_gelu` - Gaussian Error Linear Unit
- `ggml_unary_op_gelu_quick` - Fast GELU approximation
- `ggml_unary_op_silu` - Sigmoid Linear Unit
- `ggml_unary_op_hardswish` - Hard Swish activation
- `ggml_unary_op_hardsigmoid` - Hard Sigmoid activation
- `ggml_unary_op_exp` - Exponential function
- `ggml_unary_op_gelu_erf` - GELU using error function

### Matrix Operations (`mul_mat.py`)
- `ggml_op_mul_mat` - Matrix multiplication

## Output Structure

Generated kernels are organized by architecture:

```
output-dir/
└── aie2/
    ├── kernel_name.pdi           # PDI (Program Debug Information) file
    └── kernel_name_insts.bin     # Binary instructions
```

The backend automatically discovers kernels in this structure when `GGML_HSA_KERNEL_DIR` is set.

## Example Workflows

### Minimal Setup

For testing or simple applications:

```bash
ggml-hsa-gen-kernels \
    --config /usr/local/share/ggml-hsa/examples/minimal-kernel-config.json \
    --output-dir ./kernels
```

### Production Deployment

1. **Development Phase** (with IRON):
   ```bash
   # Create custom config for your model
   cat > model-kernels.json << EOF
   {
       "kernels": [
           {
               "kernel_name": "ggml_op_add",
               "kernel_source": "binary_ops.py",
               "arch": "aie2",
               "input_tensors": ["(4096,1,1,1)/bf16", "(4096,1,1,1)/bf16"],
               "output_tensor": "(4096,1,1,1)/bf16",
               "exported_name": "add_bf16_4096_model"
           }
       ]
   }
   EOF
   
   # Generate kernels
   ggml-hsa-gen-kernels \
       --config model-kernels.json \
       --output-dir ./model-kernels \
       --verbose
   ```

2. **Deployment Phase** (without IRON):
   ```bash
   # Copy kernels to production
   tar czf model-kernels.tar.gz model-kernels/
   scp model-kernels.tar.gz production:/opt/
   
   # On production machine
   cd /opt && tar xzf model-kernels.tar.gz
   export GGML_HSA_KERNEL_DIR=/opt/model-kernels
   ```

## Troubleshooting

### Common Issues

**Error: "Failed to import kernel build module: No module named 'numpy'"**
- Solution: IRON environment is not active. Run `source env_setup.sh` first.

**Error: "Kernel source not found"**
- Solution: Verify kernel_source path in config. Should be relative to kernels directory.

**Error: "Invalid tensor descriptor"**
- Solution: Check tensor format. Must be `(dim0,dim1,dim2,dim3)/dtype` with parentheses.

**Error: "Shape must have exactly 4 dimensions"**
- Solution: GGML tensors always have 4 dimensions. Use 1 for unused dimensions.

### Validation

Test your configuration before running:

```bash
# Validate JSON syntax
python3 -c "import json; json.load(open('config.json'))"

# Dry run (will fail at compilation without IRON, but validates config)
ggml-hsa-gen-kernels --config config.json --output-dir /tmp/test
```

## Performance Considerations

- Pregenerated kernels have identical performance to JIT-compiled kernels
- Loading from disk is faster than JIT compilation
- Kernel size is typically 10-100KB per kernel
- Consider generating all kernels used by your model for optimal performance

## Integration with CMake

When building with CMake and `GGML_HSA_JIT_COMPILE=ON`, the tool is automatically installed:

```cmake
# Tool installed to: ${CMAKE_INSTALL_BINDIR}/ggml-hsa-gen-kernels
# Examples installed to: share/ggml-hsa/examples/
```

Access installed examples:
```bash
ls /usr/local/share/ggml-hsa/examples/
# minimal-kernel-config.json
# example-kernel-config.json
```
