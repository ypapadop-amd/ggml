# GGML HSA backend AGENTS.md - AI Agent Guidelines for ggml-hsa

This document provides guidance for AI agents working on the ggml-hsa codebase.

## Project Overview

The ggml-hsa backend enables GGML tensor operations to run on AMD XDNA NPUs (AI Engines). It supports:

- **aie2** architecture (Phoenix, Hawk Point)
- **aie2p** architecture (Strix Halo, Krackan)

The backend uses a multi-backend kernel compilation system with per-operation dispatch. Currently supported backends:

- **IRON** (MLIR-AIE framework) - Optimized AIE kernels

The system supports both JIT and AOT compilation.

### Host Operations vs AIE Kernels

Some operations run on the host CPU rather than the AIE:

- **Host operations** (`DUP`, `CPY`, `CONT`): Implemented in `host-ops.cpp`, execute on the CPU
- **AIE kernels**: All other supported operations, compiled and dispatched to AIE tiles

Host operations are handled separately in `ggml_backend_hsa_device_supports_op()` and bypass
the kernel compilation pipeline.

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
│   ├── build_iron.py            # IRON backend compiler
│   ├── kernel.py                # Core types: Backend enum, Kernel, KernelSpec
│   ├── tensor_desc.py           # Tensor descriptor dataclass
│   ├── binary_ops.py            # Top-level GGML binary op dispatch
│   ├── unary_ops.py             # Top-level GGML unary op dispatch
│   ├── scale.py                 # Top-level scale op dispatch
│   ├── soft_max.py              # Top-level softmax op dispatch
│   ├── clamp.py                 # Top-level clamp op dispatch
│   ├── mul_mat.py               # Top-level matrix multiply dispatch
│   ├── argmax.py                # Top-level argmax op dispatch
│   ├── count_equal.py           # Top-level count_equal op dispatch
│   ├── cross_entropy_loss.py    # Top-level cross entropy loss op dispatch
│   └── iron/                    # IRON kernel implementations
│       ├── __init__.py          # Subpackage init
│       ├── utils.py             # Shared utilities (alignment, device mapping)
│       ├── binary_ops.py/cc     # Binary ops (ADD, SUB, MUL, DIV) - multiple ops per file
│       ├── unary_ops.py/cc      # Unary ops (ABS, NEG, RELU, SILU, etc.) - multiple ops per file
│       ├── scale.py/cc          # Scale IRON design + AIE core function
│       ├── softmax.py/cc        # Softmax IRON design + AIE core function
│       ├── clamp.py/cc          # Clamp IRON design + AIE core function
│       ├── argmax.py/cc         # Argmax IRON design + AIE core function
│       ├── count_equal.py/cc    # Count equal IRON design + AIE core function
│       ├── cross_entropy_loss.py/cc  # Cross entropy loss IRON design + AIE core function
│       ├── gemm.py              # Matrix multiplication IRON design
│       ├── ggml-aie.hpp         # Common AIE type definitions
│       ├── aie_kernel_utils.h   # AIE kernel utility macros
│       ├── aie_kernel_math.h    # AIE math utility functions (vec_exp)
│       ├── aie2/                # aie2-specific core functions (use only when shared won't work)
│       └── aie2p/               # aie2p-specific core functions (use only when shared won't work)
└── cmake/                       # CMake utilities
```

**Note:** Related operations are grouped in the same file (e.g., all unary ops in `unary_ops.py/cc`,
all binary ops in `binary_ops.py/cc`). Architecture-specific directories (`aie2/`, `aie2p/`) should
only be used when a shared implementation cannot work across architectures; prefer shared
implementations in the parent `iron/` directory.

### Two-Layer Dispatch Architecture

The kernel build system uses a two-layer dispatch architecture that separates
static operation mapping from runtime backend selection:

#### Layer 1: Static Mapping (Kernel)

The `_op_to_kernel_map` in `build.py` maps GGML operation names to `Kernel` objects:

```python
from kernel import Kernel

_op_to_kernel_map = {
    "ADD": Kernel("ggml_op_add", "binary_ops.py"),
    "SCALE": Kernel("ggml_op_scale", "scale.py"),
}
```

The `Kernel` dataclass identifies:

- `name`: The dispatch function name (e.g., `"ggml_op_add"`)
- `source_file`: The Python module containing the dispatch function

#### Layer 2: Runtime Dispatch (KernelSpec)

Dispatch functions examine tensor parameters and return a `KernelSpec`:

```python
from kernel import Backend, KernelSpec
from .iron.scale import scale

def ggml_op_scale(arch, input_tensors, output_tensor, op_params) -> KernelSpec:
    return KernelSpec(
        backend=Backend.IRON,
        op_name="GGML_OP_SCALE",
        arch=arch,
        input_tensors=input_tensors,
        output_tensor=output_tensor,
        op_params=op_params,
        function=scale,
    )
```

The `KernelSpec` specifies:

- `backend`: Which compilation backend to use (`Backend.IRON`)
- `op_name`: Name of the operation (e.g., `"GGML_OP_SCALE"`)
- `arch`: Target architecture string (`"aie2"` or `"aie2p"`)
- `input_tensors`: List of input tensor descriptors
- `output_tensor`: Output tensor descriptor
- `op_params`: Operation-specific parameters as a bytearray
- `function`: The callable that generates backend-specific IR

This enables per-invocation backend selection based on tensor shapes, dtypes,
or other runtime parameters.

### Compilation Pipeline

The compilation flow in `ggml_compile_op`:

1. Look up `Kernel` from `_op_to_kernel_map`
2. Dynamically import the dispatch module
3. Call dispatch function to get `KernelSpec` (includes all tensor/op context)
4. Look up compiler function via `get_compiler(backend)`
5. Invoke the backend-specific compiler

```
ggml_compile_op("SCALE", ...)
    └─> get_kernel("SCALE") -> Kernel("ggml_op_scale", "scale.py")
    └─> import_from_path("ggml_op_scale", "scale.py")
    └─> ggml_op_scale(...) -> KernelSpec(backend=IRON, function=scale)
    └─> get_compiler(Backend.IRON) -> compile_iron_kernel
    └─> compile_iron_kernel(kernel_spec, ...)
```

### Backend Compilers

Each backend has a dedicated compiler module:

- **IRON** (`build_iron.py`): Compiles IRON Python designs to PDI/instructions
  - Calls the `KernelSpec.function` to generate an MLIR module
  - Compiles any C++ core functions to object files
  - Produces final `.pdi` and `_insts.bin` files

Compilers are registered in `build.py`:

```python
from kernel import Backend
from build_iron import compile_iron_kernel

_compilers = {
    Backend.IRON: compile_iron_kernel,
}
```

### IRON Kernel Implementations

IRON kernels (`kernels/iron/*.py`) define:

- Data movement via ObjectFifos (input/output streaming)
- Worker placement on AIE tiles
- Runtime sequences for DMA transfers
- External function declarations for C++ core functions

These are paired with C++ core functions (`kernels/iron/*.cc`) that implement
the actual vectorized computations using the AIE API.

### Broadcasting Support

Binary operations (`ADD`, `SUB`, `MUL`, `DIV`) support multi-dimensional broadcasting
following GGML semantics where `src1` can be repeated to fill `dst`:

- **Validation**: `dst->ne[i] % src1->ne[i] == 0` for all dimensions (per `ggml_can_repeat`)
- **Implementation**: The broadcast kernel receives full `src1` buffer and shape tuples,
  then computes per-element `src1` indices via 4D coordinate decomposition and modulo

Key data structures in `binary_ops.py`:

```python
@dataclass(frozen=True)
class BroadcastFunctionSpec:
    external_function: ExternalFunction
    num_elements_out: int
    num_elements_src1: int
    src1_ne: tuple  # (ne0, ne1, ne2, ne3) - src1 shape
    dst_ne: tuple   # (ne0, ne1, ne2, ne3) - dst shape
```

The C++ kernel computes broadcast indices using 32-bit arithmetic only (AIE cores lack
64-bit division runtime):

```cpp
// Decompose global index g into 4D dst coordinates
int32_t i0 = g % dst_ne0;
int32_t i1 = (g / d1) % dst_ne1;
int32_t i2 = (g / d2) % dst_ne2;
int32_t i3 = g / (d2 * dst_ne2);

// Apply broadcast modulo to get src1 coordinates
int32_t j0 = i0 % src1_ne0;
int32_t j1 = i1 % src1_ne1;
int32_t j2 = i2 % src1_ne2;
int32_t j3 = i3 % src1_ne3;

// Compute linear src1 index
int32_t idx_src1 = j0 + j1 * s1 + j2 * s2 + j3 * s3;
```

## Kernel Development Pattern

Each kernel consists of three files across two layers:

### 1. Dispatch Function (e.g., `kernels/unary_ops.py`)

Returns a `KernelSpec` specifying backend, function, and tensor context:

- Imports the kernel function from the appropriate backend subpackage
- Provides the standard GGML dispatch signature
- Returns `KernelSpec` with all fields: `backend`, `op_name`, `arch`, `input_tensors`, `output_tensor`, `op_params`, `function`
- May use `functools.partial` to bind operation-specific parameters

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
   _op_to_kernel_map = {
       "NEW_OP": Kernel("ggml_op_new_op", "new_op.py"),
   }
   ```

2. **Create the dispatch function** (`kernels/new_op.py`):

   ```python
   """Top-level entry point for GGML_OP_NEW_OP."""
   from .iron.new_op import new_op
   from .kernel import Backend, KernelSpec

   def ggml_op_new_op(
       arch: str, input_tensors: list, output_tensor, op_params: bytearray
   ) -> KernelSpec:
       """GGML_OP_NEW_OP implementation."""
       return KernelSpec(
           backend=Backend.IRON,
           op_name="GGML_OP_NEW_OP",
           arch=arch,
           input_tensors=input_tensors,
           output_tensor=output_tensor,
           op_params=op_params,
           function=new_op,
       )
   ```

3. **Create the IRON design** (`kernels/iron/new_op.py`):
   - Import from `aie.iron` (ObjectFifo, Program, Runtime, Worker, etc.)
   - Import utilities from `.utils` (arch_to_device, align_to_arch, etc.)
   - Define the data flow and compute structure
   - Create external function specs for the C++ core function
   - Function signature: `def new_op(arch, input_tensors, output_tensor, op_params)`

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

## Adding a New Compilation Backend

To add a new backend (e.g., Triton):

1. **Add to the Backend enum** in `kernels/kernel.py`:

   ```python
   class Backend(Enum):
       IRON = auto()
       TRITON = auto()  # New backend
   ```

2. **Create the backend compiler** (`kernels/build_triton.py`):

   ```python
   def compile_triton_kernel(
       kernel_spec: KernelSpec,
       arch: str,
       input_tensors: list[TensorDesc],
       output_tensor: TensorDesc,
       op_params: bytearray,
       work_dir: Path,
       exported_name: str,
       output_directory: Path,
       logger: logging.Logger,
       verbose: bool,
   ) -> None:
       # Call kernel_spec.function to generate Triton IR
       # Compile to PDI and instructions
       pass
   ```

3. **Register the compiler** in `kernels/build.py`:

   ```python
   from build_triton import compile_triton_kernel

   _compilers = {
       Backend.IRON: compile_iron_kernel,
       Backend.TRITON: compile_triton_kernel,
   }
   ```

4. **Update dispatch functions** to return the new backend when appropriate:

   ```python
   def ggml_op_new_op(...) -> KernelSpec:
       if some_condition:
           return KernelSpec(backend=Backend.TRITON, function=triton_new_op)
       return KernelSpec(backend=Backend.IRON, function=iron_new_op)
   ```

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
- **Success:** Look for `<N>/<N> tests passed`.
- **Failure:** Look for `0/0 tests passed` or `Could not create kernel for tensor`.

### Testing Commands

```bash
# Test a specific operation (clear cache when testing kernel changes)
GGML_HSA_KERNEL_CACHE_CLEAR=1 ./bin/test-backend-ops -o ADD -b HSA

# Test with verbose JIT output for debugging compilation issues
GGML_HSA_KERNEL_CACHE_CLEAR=1 GGML_HSA_JIT_VERBOSE=1 ./bin/test-backend-ops -o SOFT_MAX -b HSA

# Test with debug logging enabled
GGML_HSA_KERNEL_CACHE_CLEAR=1 GGML_HSA_ENABLE_LOG=1 ./bin/test-backend-ops -o MUL_MAT -b HSA

# Run all HSA backend tests
./bin/test-backend-ops -b HSA
```

## Debugging

### Common Error Messages

| Error | Cause | Solution |
| ----- | ----- | -------- |
| `Could not create kernel for tensor` | Kernel compilation failed or op not supported | Enable `GGML_HSA_JIT_VERBOSE=1` to see compilation errors |
| `0/0 tests passed` | Operation not supported for tensor configuration | Check tensor shapes, types, and contiguity requirements |
| `exception caught` in logs | Runtime error during kernel execution | Enable `GGML_HSA_ENABLE_LOG=1` for detailed error context |
| `unsupported device` | Architecture not recognized | Verify device reports as `aie2` or `aie2p` |

### Debugging Workflow

1. **Enable verbose logging**:

   ```bash
   GGML_HSA_ENABLE_LOG=1 GGML_HSA_JIT_VERBOSE=1 ./bin/test-backend-ops -o OP -b HSA
   ```

2. **Clear the kernel cache** when testing kernel changes:

   ```bash
   GGML_HSA_KERNEL_CACHE_CLEAR=1 ./bin/test-backend-ops -o OP -b HSA
   ```

3. **Check compilation output**: JIT artifacts are stored in the cache directory
   (default: `~/.cache/ggml-hsa-kernels/` or `GGML_HSA_KERNEL_CACHE_DIR`)

4. **Inspect generated MLIR**: With `GGML_HSA_JIT_VERBOSE=1`, the compilation log shows
   the generated MLIR and any compilation errors

### Kernel Naming Conventions

Kernel names are generated deterministically based on tensor configuration:

```text
<op_name>_<arch>_<input_types>_<output_type>_<shapes>[_<op_params_hash>]
```

Components:

- `op_name`: GGML operation (e.g., `ADD`, `SOFT_MAX`)
- `arch`: Target architecture (`aie2` or `aie2p`)
- `input_types`: Input tensor data types
- `output_type`: Output tensor data type
- `shapes`: Tensor dimensions
- `op_params_hash`: (Optional) Hash of non-zero `op_params` bytes

Example: `ADD_aie2_bf16_bf16_bf16_1024` for a 1024-element bf16 add on aie2.

When `op_params` contains non-zero values (e.g., scale factors, epsilon), they are
encoded into the kernel name to ensure different parameter combinations produce
distinct cached kernels.

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
| `GGML_HSA_KERNEL_CACHE_CLEAR` | Set to `1` to clear the kernel cache (required when testing kernel changes) |
| `GGML_HSA_JIT_VERBOSE` | Verbose JIT output |
