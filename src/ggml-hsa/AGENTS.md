# GGML HSA backend AGENTS.md - AI Agent Guidelines for ggml-hsa

This document provides guidance for AI agents working on the ggml-hsa codebase.

## Project Overview

The ggml-hsa backend enables GGML tensor operations to run on AMD XDNA NPUs (AI Engines). It supports:

- **aie2** architecture (Phoenix, Hawk Point)
- **aie2p** architecture (Strix Halo, Krackan)

The backend uses the IRON framework (part of MLIR-AIE) for kernel development and supports both JIT and AOT compilation.

## Codebase Structure

```
src/ggml-hsa/
├── ggml-hsa.cpp                 # Backend implementation (HSA runtime integration)
├── common.hpp                   # Common utilities and type definitions
├── host-ops.cpp/hpp             # Host-side operation implementations
├── kernel-discovery.cpp/hpp     # Runtime kernel discovery and loading
├── aie-kernel.cpp/hpp           # AIE kernel abstraction layer
├── aie-kernel-compiler.cpp/hpp  # JIT compilation interface
├── type-traits.hpp              # GGML type to C++ type mapping
├── kernels/                     # AIE kernel implementations
│   ├── build.py                 # Kernel compilation orchestrator
│   ├── tensor_desc.py           # Tensor descriptor utilities
│   ├── utils.py                 # Shared Python utilities
│   ├── unary_ops.py/cc          # Unary operations (sqr, sqrt, relu, etc.)
│   ├── binary_ops.py/cc         # Binary operations (add, sub, mul, div)
│   ├── scale.py/cc              # Scale operation
│   ├── softmax.py/cc            # Softmax operation
│   ├── clamp.py/cc              # Clamp operation
│   ├── gemm.py                  # Matrix multiplication
│   ├── aie2/                    # aie2-specific core functions
│   └── aie2p/                   # aie2p-specific kernels
└── cmake/                       # CMake utilities
```

## Kernel Development Pattern

Each kernel consists of two files:

### 1. Python File (e.g., `unary_ops.py`)

Defines the IRON program structure:

- Data movement (ObjectFifos)
- Worker placement
- Runtime sequences
- External function declarations

### 2. C++ File (e.g., `unary_ops.cc`)

Implements the core computation or core function using the AIE API:

- Uses `#ifdef GGML_OP_<OP>` guards for selective compilation
- Uses `INPUT_DTYPE` and `OUTPUT_DTYPE` macros for type flexibility
- Includes `<aie_api/aie.hpp>` for AIE intrinsics
- Functions follow naming convention: `ggml_op_<operation>`

## Adding a New Kernel

1. **Register the operation** in `build.py`:

   ```python
   op_to_kernel_map = {
       "NEW_OP": Kernel("ggml_op_new_op", "new_op.py"),
   }
   ```

2. **Create the Python file** (`kernels/new_op.py`):
   - Import from `aie.iron` (ObjectFifo, Program, Runtime, Worker, etc.)
   - Define the data flow and compute structure
   - Export via `ggml_op_new_op(arch, input_tensors, output_tensor, op_params)`

3. **Create the C++ file** (`kernels/new_op.cc`):
   - Use compile guards: `#ifdef GGML_OP_NEW_OP`
   - Implement: `void ggml_op_new_op(const INPUT_DTYPE*, OUTPUT_DTYPE*, int32_t N)`
   - Use `extern "C"` linkage

4. (optional) **Add backend support** in `ggml-hsa.cpp`:
   - Add to `ggml_hsa_op_supports()` for operation support check
   - Add case in `ggml_hsa_compute_forward()` for dispatch

## Code Conventions

### C++ (Host Code)

- Use `std::` prefix for standard library types
- Use `GGML_ASSERT()` / `GGML_ABORT()` for error handling
- Check HSA calls with `GGML_HSA_CHECK()` macro
- Follow existing formatting (see `.clang-format`)

### C++ (Kernel Code)

- Include `ggml-aie.hpp` for common type definitions
- Use `event0()` / `event1()` for profiling regions
- Prefer vectorized operations from `aie_api/aie.hpp`
- Keep kernels simple and focused on compute
- Follow existing formatting (see `.clang-format`)

### Python

- Follow existing patterns in `unary_ops.py` / `binary_ops.py`
- Use `CoreFunctionSpec` or similar for external function specs
- Handle tensor alignment with `arch_aligned_num_elements()`
- Use `max_tile_size()` for optimal tiling
- Follow existing formatting using `black`

## Data Types

Supported GGML types and their mappings:

| GGML Type | Native Support | Notes |
| ----------- | --------------- | ------- |
| `GGML_TYPE_I8` | Yes | Native AIE type |
| `GGML_TYPE_I16` | Yes | Native AIE type |
| `GGML_TYPE_I32` | Yes | Native AIE type |
| `GGML_TYPE_BF16` | Yes | Native AIE type |
| `GGML_TYPE_F16` | Via BF16 | Converted internally |
| `GGML_TYPE_F32` | Emulated | Slower than native |

## Environment Setup

```bash
# Set up Python environment with IRON dependencies
source ./env_setup.sh
# Or manually:
python3 -m pip install -r requirements.txt
```

## Testing

- Ensure that an IRON environment is present and active
- Build with `GGML_HSA=ON` and optionally `GGML_HSA_JIT_COMPILE=ON`
- Test files are in `tests` and `tests/ggml-hsa/`
- Ensure kernels work for both `aie2` and `aie2p` architectures

## Common Pitfalls

1. **Tensor alignment**: AIE requires specific alignment (4-byte boundaries)
2. **Tile sizes**: Must evenly divide the total element count
3. **Type casting**: Be explicit with casts in kernel code
4. **Contiguous tensors**: Many operations require contiguous memory layout
5. **op_params encoding**: Non-zero op_params are encoded in kernel names

## Useful Environment Variables

| Variable | Purpose |
| ---------- | --------- |
| `GGML_HSA_ENABLE_LOG` | Enable debug logging |
| `GGML_HSA_KERNEL_DIR` | Precompiled kernel directory |
| `GGML_HSA_KERNEL_CACHE_DIR` | JIT cache directory |
| `GGML_HSA_JIT_VERBOSE` | Verbose JIT output |
