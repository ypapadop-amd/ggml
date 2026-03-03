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
├── kernels/                     # AIE kernel implementations (two-layer architecture)
│   ├── __init__.py              # Package exports (ggml_compile_op, TensorDesc)
│   ├── build.py                 # Kernel compilation orchestrator
│   ├── tensor_desc.py           # Tensor descriptor dataclass
│   ├── binary_ops.py            # Top-level GGML binary op wrappers
│   ├── unary_ops.py             # Top-level GGML unary op wrappers
│   ├── scale.py                 # Top-level scale op wrapper
│   ├── soft_max.py              # Top-level softmax op wrapper
│   ├── clamp.py                 # Top-level clamp op wrapper
│   ├── mul_mat.py               # Top-level matrix multiply wrapper
│   └── iron/                    # IRON kernel implementations
│       ├── __init__.py          # Subpackage init
│       ├── utils.py             # Shared utilities (alignment, device mapping)
│       ├── binary_ops.py/cc     # Binary ops IRON design + AIE core function
│       ├── unary_ops.py/cc      # Unary ops IRON design + AIE core function
│       ├── scale.py/cc          # Scale IRON design + AIE core function
│       ├── softmax.py/cc        # Softmax IRON design + AIE core function
│       ├── clamp.py/cc          # Clamp IRON design + AIE core function
│       ├── gemm.py              # Matrix multiplication IRON design
│       ├── ggml-aie.hpp         # Common AIE type definitions
│       ├── aie_kernel_utils.h   # AIE kernel utility macros
│       ├── aie2/                # aie2-specific core functions (mm.cc, zero.cc)
│       └── aie2p/               # aie2p-specific core functions (mm.cc, zero.cc)
└── cmake/                       # CMake utilities
```

### Two-Layer Architecture

The kernel code follows a two-layer architecture:

1. **Top-level wrappers** (`kernels/*.py`): Thin wrappers that validate inputs and
   delegate to the corresponding IRON implementation. These follow the standard
   GGML operation signature: `ggml_op_<name>(arch, input_tensors, output_tensor, op_params)`.

2. **IRON implementations** (`kernels/iron/*.py`): Low-level kernel designs that
   define data movement (ObjectFifos), worker placement, and runtime sequences.
   These are paired with C++ core functions (`kernels/iron/*.cc`) that implement
   the actual vectorized computations using the AIE API.

## Kernel Development Pattern

Each kernel consists of three files across two layers:

### 1. Top-level Wrapper (e.g., `kernels/unary_ops.py`)

Thin wrapper that:

- Imports the IRON implementation from `iron/` subpackage
- Provides the standard GGML operation signature
- Validates inputs before delegation

### 2. IRON Design (e.g., `kernels/iron/unary_ops.py`)

Defines the IRON program structure:

- Data movement via ObjectFifos (input/output streaming)
- Worker placement on AIE tiles
- Runtime sequences for DMA transfers
- External function declarations for C++ core functions
- Tiling and alignment calculations

### 3. C++ Core Function (e.g., `kernels/iron/unary_ops.cc`)

Implements the core computation using the AIE API:

- Uses `#ifdef GGML_OP_<OP>` guards for selective compilation
- Uses `INPUT_DTYPE` and `OUTPUT_DTYPE` macros for type flexibility
- Includes `<aie_api/aie.hpp>` for AIE vector intrinsics
- Functions follow naming convention: `ggml_op_<operation>`
- Uses `extern "C"` linkage for IRON integration

## Adding a New Kernel

1. **Register the operation** in `kernels/build.py`:

   ```python
   op_to_kernel_map = {
       "NEW_OP": Kernel("ggml_op_new_op", "new_op.py"),
   }
   ```

2. **Create the top-level wrapper** (`kernels/new_op.py`):

   ```python
   """Top-level entry point for GGML_OP_NEW_OP."""
   from .iron.new_op import new_op

   def ggml_op_new_op(arch: str, input_tensors: list, output_tensor, op_params: bytearray):
       """GGML_OP_NEW_OP implementation."""
       return new_op(arch=arch, input_tensors=input_tensors,
                     output_tensor=output_tensor, op_params=op_params)
   ```

3. **Create the IRON design** (`kernels/iron/new_op.py`):
   - Import from `aie.iron` (ObjectFifo, Program, Runtime, Worker, etc.)
   - Import utilities from `.utils` (arch_to_device, align_to_arch, etc.)
   - Define the data flow and compute structure
   - Create external function specs for the C++ core function

4. **Create the C++ core function** (`kernels/iron/new_op.cc`):
   - Use compile guards: `#ifdef GGML_OP_NEW_OP`
   - Implement: `void ggml_op_new_op(const INPUT_DTYPE*, OUTPUT_DTYPE*, int32_t N)`
   - Use `extern "C"` linkage
   - Include `ggml-aie.hpp` for common type definitions

5. **Register the file with CMake**
   - Add the files in the `kernels/CMakeLists.txt`

6. (optional) **Add backend support** in `ggml-hsa.cpp`:
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

- Follow existing patterns in `iron/unary_ops.py` / `iron/binary_ops.py`
- Use `CoreFunctionSpec` dataclass for external function specifications
- Import utilities from `iron/utils.py`:
  - `arch_to_device()` - Convert arch string to IRON device object
  - `arch_aligned_num_elements()` - Align tensor sizes to architecture requirements
  - `align_to_arch()` - Align arbitrary sizes to byte boundaries
  - `max_tile_size()` - Calculate optimal tile size for vectorization
  - `suppress_import_pyxrt_msg()` - Suppress noisy pyxrt import messages
- Top-level wrappers import from `.iron.<module>` subpackage
- Follow existing formatting using `black`
- Add module docstrings to all Python files

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
- A specific operation can be tested using `test-backend-ops -o OP`
- **Success:** Look for `<N>/<N> tests passed`.
- **Failure:** Look for `0/0 tests passed` or `Could not create kernel for tensor`.

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
