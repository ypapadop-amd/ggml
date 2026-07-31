# GGML HSA backend AGENTS.md - AI Agent Guidelines for ggml-hsa

This document provides guidance for AI agents working on the ggml-hsa codebase.

## Project Overview

The ggml-hsa backend enables GGML tensor operations to run on AMD XDNA NPUs (AI Engines). It supports:

| Architecture | IRON Device | AIE Array | Example Platforms |
| ------------ | ----------- | --------- | ----------------- |
| **aie2** | NPU1 | 4×4 cores | Phoenix, Hawk Point |
| **aie2p** | NPU2 | 4×8 cores | Strix Point, Strix Halo, Krackan |

The backend uses a multi-backend kernel compilation system with per-operation dispatch. Currently supported backends:

- **IRON** (MLIR-AIE framework) - Optimized AIE kernels (default)
- **TRITON** (Triton-XDNA) - Compiler-driven kernel generation via MLIR-AIR/AIE (optional)

The system supports both JIT and AOT compilation.

### Host Operations vs AIE Kernels

Some operations run on the host CPU rather than the AIE:

- **Host operations** (`DUP`, `CPY`, `CONT`): Implemented in `host-ops.cpp`, execute on the CPU
- **AIE kernels**: All other supported operations, compiled and dispatched to AIE tiles

Host operations are handled separately in `ggml_backend_hsa_device_supports_op()` and bypass
the kernel compilation pipeline.

## Codebase Structure

```text
src/ggml-hsa/
├── ggml-hsa.cpp                 # Backend implementation (HSA runtime integration)
├── common.hpp                   # Common utilities and type definitions
├── host-ops.cpp/hpp             # Host-side operation implementations
├── kernel-discovery.cpp/hpp     # Runtime kernel discovery and loading
├── aie-kernel.cpp/hpp           # AIE kernel abstraction layer
├── kernel-compiler.cpp/hpp      # JIT compilation interface
├── type-traits.hpp              # GGML type to C++ type mapping
├── kernels/                     # AIE kernel implementations (two-layer architecture)
│   ├── __init__.py              # Package exports (ggml_compile_op, CompilerConfig, Kernel, TensorDesc, ggml_tensor_to_tensordesc)
│   ├── build.py                 # Kernel compilation orchestrator
│   ├── build_iron.py            # IRON backend compiler
│   ├── build_triton.py          # Triton backend compiler
│   ├── kernel.py                # Core types: Backend enum, Kernel, KernelSpec
│   ├── tensor_desc.py           # Tensor descriptor dataclass
│   ├── binary_ops.py            # Top-level GGML binary op dispatch
│   ├── unary_ops.py             # Top-level GGML unary op dispatch
│   ├── scale.py                 # Top-level scale op dispatch
│   ├── soft_max.py              # Top-level softmax op dispatch
│   ├── clamp.py                 # Top-level clamp op dispatch
│   ├── mul_mat.py               # Top-level matrix multiply dispatch
│   ├── pool_2d.py               # Top-level 2D pooling op dispatch
│   ├── argmax.py                # Top-level argmax op dispatch
│   ├── count_equal.py           # Top-level count_equal op dispatch
│   ├── cross_entropy_loss.py    # Top-level cross entropy loss op dispatch
│   ├── triton/                  # Triton kernel implementations
│   │   ├── __init__.py          # Subpackage init
│   │   ├── utils.py             # Shared utilities (dtype conversion)
│   │   ├── vecadd.py            # Vector addition Triton kernel
│   │   ├── vecadd_aie2.mlir     # MLIR transform/tiling script for AIE2
│   │   └── vecadd_aie2p.mlir    # MLIR transform/tiling script for AIE2P
│   └── iron/                    # IRON kernel implementations
│       ├── __init__.py          # Subpackage init
│       ├── utils.py             # Shared utilities (alignment, device mapping)
│       ├── binary_ops.py/cc     # Binary ops (ADD, SUB, MUL, DIV) with broadcast support
│       ├── unary_ops.py/cc      # Unary ops - see "Supported Operations" section for details
│       ├── scale.py/cc          # Scale IRON design + AIE core function
│       ├── softmax.py/cc        # Softmax IRON design + AIE core function (unary/masked/ternary variants)
│       ├── clamp.py/cc          # Clamp IRON design + AIE core function
│       ├── pool_2d.py/cc        # 2D pooling IRON design + AIE core function (MAX/AVG, one channel-plane per tile)
│       ├── argmax.py/cc         # Argmax IRON design + AIE core function
│       ├── count_equal.py/cc    # Count equal IRON design + AIE core function
│       ├── cross_entropy_loss.py/cc  # Cross entropy loss IRON design + AIE core function
│       ├── gemm.py              # Matrix multiplication IRON design
│       ├── ggml-aie.hpp         # Common AIE type definitions
│       ├── aie_kernel_utils.h   # Loop optimization macros (AIE_LOOP_UNROLL, AIE_PREPARE_FOR_PIPELINING, etc.)
│       ├── aie_kernel_math.h    # AIE math utility functions (scalar_exp, scalar_log, pow2, vec_exp)
│       ├── aie2/                # aie2-specific core functions
│       │   ├── mm.cc            # Matrix multiply kernels for aie2
│       │   └── zero.cc          # Zero-initialization helpers for aie2
│       └── aie2p/               # aie2p-specific core functions
│           ├── mm.cc            # Matrix multiply kernels for aie2p
│           └── zero.cc          # Zero-initialization helpers for aie2p
└── cmake/                       # CMake utilities
```

**Note:** Related operations are grouped in the same file (e.g., all unary ops in `unary_ops.py/cc`,
all binary ops in `binary_ops.py/cc`). Architecture-specific directories (`aie2/`, `aie2p/`) contain
kernels that require different implementations per architecture (e.g., matrix multiply uses
architecture-specific intrinsics). Prefer shared implementations in the parent `iron/` directory
when possible.

### Two-Layer Dispatch Architecture

The kernel build system uses a two-layer dispatch architecture that separates
static operation mapping from runtime backend selection:

#### Layer 1: Static Mapping (Kernel)

The `_OP_KERNEL_MAP` in `build.py` maps GGML operation names to `Kernel` objects:

```python
from kernel import Kernel

_OP_KERNEL_MAP = {
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
from functools import partial
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
        function=partial(
            scale,
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
            op_params=op_params,
        ),
    )
```

The `function` field uses `functools.partial` to bind all arguments at dispatch time,
so the backend compiler can call it with no arguments to generate the IR.

The `KernelSpec` specifies:

- `backend`: Which compilation backend to use (`Backend.IRON` or `Backend.TRITON`)
- `op_name`: Name of the operation (e.g., `"GGML_OP_SCALE"`)
- `arch`: Target architecture string (`"aie2"` or `"aie2p"`)
- `input_tensors`: List of input tensors
- `output_tensor`: Output tensor
- `function`: The callable that generates backend-specific IR
- `op_params`: Operation-specific parameters as a bytearray (optional, defaults to `None`)
- `config`: Dictionary for additional backend-specific configuration (optional, defaults to `{}`)

This enables per-invocation backend selection based on tensor shapes, dtypes,
or other runtime parameters. Dispatch functions can return a single `KernelSpec`
or a `list[KernelSpec]` for multi-backend fallback (see "Multi-Backend Fallback").

### Compilation Pipeline

`ggml_compile_op(op_name, arch, input_tensors, output_tensor, op_params, exported_name, config)`
takes a `CompilerConfig` (output directory, backend order, verbosity). The flow:

1. Look up `Kernel` from `_OP_KERNEL_MAP`
2. Dynamically import the dispatch module
3. Call dispatch function to get `KernelSpec` or `list[KernelSpec]`
4. Normalize/order the specs via `_make_kernel_specs` using `config.compilers` (see "Multi-Backend Fallback")
5. Iterate through specs, look up compiler via `_get_compiler(backend)`
6. Invoke the backend-specific compiler; on success, stop; on failure, try next spec

```text
ggml_compile_op("SCALE", ...)
    └─> _get_kernel("SCALE") -> Kernel("ggml_op_scale", "scale.py")
    └─> _import_from_path("ggml_op_scale", "scale.py")
    └─> ggml_op_scale(...) -> KernelSpec(backend=IRON, function=scale)
    └─> _get_compiler(Backend.IRON) -> compile_iron_kernel
    └─> compile_iron_kernel(kernel_spec, ...)
```

### Multi-Backend Fallback

Dispatch functions can return a `list[KernelSpec]` to enable fallback across backends.
The compilation pipeline tries each spec in order, stopping at the first success:

```text
ggml_compile_op("ADD", ...)
    └─> ggml_op_add(...) -> [KernelSpec(IRON, ...), KernelSpec(TRITON, ...)]
    └─> try compile_iron_kernel(spec[0]) -> success? done
    └─> try compile_triton_kernel(spec[1]) -> success? done
    └─> all failed? log error
```

This allows operations to prefer one backend but gracefully fall back to another.
For example, `ggml_op_add` returns IRON as first priority and Triton as fallback.

The `GGML_HSA_JIT_COMPILER_ORDER` environment variable (comma-separated backend names,
e.g. `iron,triton`, case-insensitive) reorders the candidate specs and drops any whose
backend is not listed. When unset or empty, the dispatch function's order is used
unchanged. This replaces the former `GGML_HSA_PREFER_TRITON` boolean.

### Backend Compilers

Each backend has a dedicated compiler module:

- **IRON** (`build_iron.py`): Compiles IRON Python designs to PDI/instructions
  - Calls the `KernelSpec.function` to generate an MLIR module
  - Compiles any C++ core functions to object files
  - Produces PDI `.pdi` and instructions `_insts.bin` files for AIE execution

- **TRITON** (`build_triton.py`): Compiles Triton kernels via MLIR-AIR/AIE
  - Uses `config_context` from `triton.backends.amd_triton_npu.config` to set compilation parameters (`compile_only`, `transform_tiling_script` from `kernel_spec.config["transform_script"]`, `output_format`, `debug`, `target`)
  - Sets `TRITON_CACHE_DIR` environment variable for artifact caching
  - Calls `kernel_spec.function()` to trigger Triton compilation
  - Extracts PDI and instructions from the generated `aie.xclbin` via `xclbinutil`
  - Produces PDI `.pdi` and instructions `_insts.bin` files for AIE execution

Compilers are resolved in `build.py` by `_get_compiler()`:

```python
from kernel import Backend


def _get_compiler(backend: Backend) -> Callable:
    if backend.name == Backend.IRON.name:
        from build_iron import compile_iron_kernel

        return compile_iron_kernel
    if backend.name == Backend.TRITON.name:
        from build_triton import compile_triton_kernel

        return compile_triton_kernel
    raise NotImplementedError(...)
```

Backend compilers are imported lazily so an IRON-only environment (without the
Triton/torch dependencies) can still compile IRON kernels. Lookup is by
`backend.name` (string comparison) rather than identity, to handle the case
where `Backend` enums from dynamically imported modules have different class
identity.

### IRON Kernel Implementations

IRON kernels (`kernels/iron/*.py`) define:

- Data movement via ObjectFifos (input/output streaming)
- Worker placement on AIE tiles
- Runtime sequences for DMA transfers
- External function declarations for C++ core functions

These are paired with C++ core functions (`kernels/iron/*.cc`) that implement
the actual vectorized computations using the AIE API.

### Matrix Multiplication (`MUL_MAT`)

The matrix multiplication kernel uses a sophisticated tiled design that distributes
computation across the AIE array:

- **Tiling**: Matrices are divided into tiles of size (m, k) for A, (k, n) for B, and (m, n) for C
- **Array utilization**: Uses 4 rows × N columns (N=4 for NPU1/aie2, N=8 for NPU2/aie2p)
- **Data flow**:
  - Matrix A tiles are broadcast across columns, distributed across rows
  - Matrix B tiles are broadcast across rows, distributed across columns
  - Output C tiles are joined from compute cores back through memory hierarchy
- **MAC dimensions**: Architecture-specific microkernel dimensions (r, s, t) determine tile constraints
  - NPU1 bf16: (4, 8, 4), i8: (4, 8, 8), i16: (4, 4, 4)
  - NPU2 bf16: (4, 8, 8) or (8, 8, 8) with bfp16 emulation, i8: (8, 8, 8), i16: (4, 4, 8)
- **Memory hierarchy**: L3 (host) → L2 (mem tiles) → L1 (compute cores) with ObjectFifos
- **Core functions**: Architecture-specific kernels in `aie2/mm.cc` and `aie2p/mm.cc`

The implementation in `gemm.py` includes both a standalone CLI tool and a `gemm()` function
callable from the dispatch layer. Key parameters include tile sizes (m, k, n), number of
columns, data types, and layout (row-major vs column-major).

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
    dst_ne: tuple  # (ne0, ne1, ne2, ne3) - dst shape
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

### Nullable Source Tensors

Some operations (e.g., `SOFT_MAX`) have optional input tensors. The compilation system
handles these as follows:

- **Host side**: The `input_tensors` list may contain `None` for optional tensors that
  are not provided. The list length matches GGML's source array size, preserving indices.
- **Dispatch functions**: Check for `None` before accessing tensor properties:

  ```python
  def ggml_op_soft_max(arch, input_tensors, output_tensor, op_params) -> KernelSpec:
      input_tensor = input_tensors[0]  # Required
      mask_tensor = input_tensors[1] if len(input_tensors) >= 2 else None  # Optional
      sink_tensor = input_tensors[2] if len(input_tensors) >= 3 else None  # Optional
  ```

- **IRON kernels**: Branch on tensor presence to generate different program structures
  (e.g., different numbers of ObjectFifos and DMA transfers)

Example in `softmax.py`:

```python
if input_tensor_count == 1:
    return create_unary_program(...)  # Just input → output
elif input_tensor_count == 2:
    return create_binary_program(...)  # input + mask → output
else:
    return create_ternary_program(...)  # input + mask + sink → output
```

## Kernel Development Pattern

Each kernel consists of three files across two layers:

### 1. Dispatch Function (e.g., `kernels/unary_ops.py`)

Returns a `KernelSpec` (or `list[KernelSpec]` for multi-backend fallback) specifying backend, function, and tensor context:

- Imports the kernel function from the appropriate backend subpackage
- Provides the standard GGML dispatch signature
- Returns `KernelSpec` with all fields: `backend`, `op_name`, `arch`, `input_tensors`, `output_tensor`, `function`
- Returns `list[KernelSpec]` when the operation supports fallback across backends (tried in order)
- Uses `functools.partial` to bind all arguments to the kernel function at dispatch time
- `op_params` and `config` are optional (included only when the operation requires them)

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
   _OP_KERNEL_MAP = {
       "NEW_OP": Kernel("ggml_op_new_op", "new_op.py"),
   }
   ```

2. **Create the dispatch function** (`kernels/new_op.py`):

   The dispatch function can return a single `KernelSpec` or a `list[KernelSpec]`
   for multi-backend fallback:

   ```python
   """Top-level entry point for GGML_OP_NEW_OP."""

   from .kernel import Backend, KernelSpec


   def ggml_op_new_op(
       arch: str, input_tensors: list, output_tensor, op_params: bytearray
   ) -> KernelSpec:
       """GGML_OP_NEW_OP implementation."""
       from functools import partial
       from .iron.new_op import new_op

       return KernelSpec(
           backend=Backend.IRON,
           op_name="GGML_OP_NEW_OP",
           arch=arch,
           input_tensors=input_tensors,
           output_tensor=output_tensor,
           function=partial(
               new_op,
               arch=arch,
               input_tensors=input_tensors,
               output_tensor=output_tensor,
           ),
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

To add a new backend, follow the pattern used for the Triton backend. This example shows how Triton was added:

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
       exported_name: str,
       output_directory: Path,
       logger: logging.Logger,
       verbose: bool,
   ) -> None:
       # Access kernel_spec.arch, kernel_spec.input_tensors, etc. as needed
       # Call kernel_spec.function to generate Triton IR
       # Compile to PDI and instructions
       pass
   ```

3. **Register the compiler** in `kernels/build.py` by adding a lazy-import
   branch to `_get_compiler()`:

   ```python
   def _get_compiler(backend: Backend) -> Callable:
       ...
       if backend.name == Backend.NEW.name:
           from build_new import compile_new_kernel

           return compile_new_kernel
   ```

4. **Update dispatch functions** to return the new backend when appropriate:

   Return a single `KernelSpec` for exclusive backend use, or a `list[KernelSpec]`
   for multi-backend fallback (tried in order):

   ```python
   def ggml_op_new_op(...) -> KernelSpec | list[KernelSpec]:
       # Single backend
       if exclusive:
           return KernelSpec(
               backend=Backend.TRITON,
               function=partial(triton_new_op, ...),
               config={"transform_script": "/path/to/transform.mlir"},
           )
       # Multi-backend fallback (IRON first, Triton fallback)
       return [
           KernelSpec(backend=Backend.IRON, function=partial(iron_new_op, ...)),
           KernelSpec(backend=Backend.TRITON, function=partial(triton_new_op, ...),
                      config={"transform_script": "/path/to/transform.mlir"}),
       ]
   ```

   The `config` dict passes backend-specific parameters (e.g., MLIR transform scripts)
   that the compiler needs but are not part of the standard kernel specification.

## Code Conventions

### C++ (Host Code)

- Use `std::` prefix for standard library types
- Use `GGML_ASSERT()` / `GGML_ABORT()` for error handling
- Check HSA calls with `GGML_HSA_CHECK()` macro
- Follow existing formatting (see `.clang-format`)

### C++ (Kernel Code)

- Include `ggml-aie.hpp` for type aliases (`i8`, `i16`, `i32`, `bf16`, `f32`) and `is_floating_point_v<T>`
- Include `aie_api/aie.hpp` for AIE intrinsics and vector types
- Use `event0()` / `event1()` (from aie_api) to mark profiling regions
- Use loop macros from `aie_kernel_utils.h` for optimization hints
- Use `INPUT_DTYPE` / `OUTPUT_DTYPE` macros (set by compiler) for type flexibility; binary ops use `INPUT0_DTYPE` / `INPUT1_DTYPE` / `OUTPUT_DTYPE`
- Prefer vectorized operations using `aie::vector<T, N>` and `aie::accum<T, N>`
- Keep kernels simple and focused on compute
- Follow existing formatting (see `.clang-format`)

### Python

- Follow existing patterns in `iron/unary_ops.py` / `iron/binary_ops.py`
- Use `CoreFunctionSpec` dataclass for external function specifications
- Import utilities from `iron/utils.py`:
  - `arch_to_device()` - Convert arch string to IRON device object (`"aie2"` → `NPU1()`, `"aie2p"` → `NPU2()`)
  - `arch_aligned_num_elements()` - Align tensor sizes to architecture requirements
  - `align_to_arch()` - Align arbitrary sizes to byte boundaries (default 4-byte alignment)
  - `max_tile_size()` - Calculate optimal tile size based on 512-bit vector register width
- Top-level wrappers import from `.iron.<module>` subpackage
- Follow existing formatting using `ruff` (see `kernels/ruff.toml`)
- Use Google-style docstrings (`Parameters:`, `Returns:`, `Raises:`) — not numpy-style
- Do not duplicate type annotations in docstrings; types belong in function signatures
- Add module docstrings to all Python files

## Supported Operations

### Fully Implemented

These operations have complete AIE kernel implementations:

| Category | Operations |
| -------- | ---------- |
| Binary | `ADD`, `SUB`, `MUL`, `DIV` (with broadcast support) |
| Unary (GGML_UNARY_OP) | `ABS`, `SGN`, `NEG`, `STEP`, `RELU`, `HARDSWISH`, `HARDSIGMOID`, `FLOOR`, `CEIL`, `ROUND`, `TRUNC` |
| Unary (GGML_OP) | `SQR`, `LOG` |
| Pooling | `POOL_2D` (`MAX` and `AVG`, with padding) |
| Other | `SCALE`, `SOFT_MAX`, `CLAMP`, `ARGMAX`, `COUNT_EQUAL`, `CROSS_ENTROPY_LOSS`, `MUL_MAT` |
| Host-only | `DUP`, `CPY`, `CONT` (run on CPU, not AIE) |

### Registered but Not Implemented

These operations are registered in `build.py` but raise `NotImplementedError`:

- `SQRT`, `SIN`, `COS` (require math library functions)
- `TANH`, `ELU`, `SIGMOID`, `SILU`, `EXP` (require exp/transcendental functions)
- `GELU`, `GELU_QUICK`, `GELU_ERF`, `XIELU` (require erf or approximations)

These are placeholders for future implementation. The `aie_kernel_math.h` header provides
utility functions like `scalar_exp`, `scalar_log`, `pow2`, and `vec_exp` that can be used as building
blocks for implementing these operations.

## Data Types

Supported GGML types and their mappings:

| GGML Type | Native Support | Notes |
| ----------- | --------------- | ------- |
| `GGML_TYPE_I8` | Yes | Native AIE type |
| `GGML_TYPE_I16` | Yes | Native AIE type |
| `GGML_TYPE_I32` | Yes | Native AIE type |
| `GGML_TYPE_BF16` | Yes | Native AIE type (preferred for float ops) |
| `GGML_TYPE_F16` | Via BF16 | Reinterpreted as BF16 by host; no conversion |
| `GGML_TYPE_F32` | Emulated | Software emulation on AIE; slower than native |

## Environment Setup

**Important:** A Python virtual environment with backend dependencies must be active.
If Python cannot find the `aie` package (IRON) or `triton` package (Triton), the virtual environment is not set up or not activated.

```bash
# Set up Python environment with IRON dependencies (default)
source ./env_setup.sh iron
# Or manually:
python3 -m pip install -r requirements-iron.txt

# Set up Python environment with Triton dependencies (includes IRON)
source ./env_setup.sh triton
# Or manually:
python3 -m pip install -r requirements-triton.txt
```

### MLIR-AIE Version

The project currently uses **mlir-aie v1.4.0**. Ensure your environment
matches this version to avoid compatibility issues with:

- IRON API changes
- ObjectFifo semantics
- Compiler flags and build system updates

## Testing

- Ensure that an IRON environment is present and active
- Build with `GGML_HSA=ON` and optionally `GGML_HSA_JIT_COMPILE=ON`
- Test files are in `tests/test-backend-ops.cpp` (shared) and `tests/ggml-hsa/` (HSA-specific: `test-mul-mat-hsa.cpp`, `test-vector-hsa.cpp`, `test-backend-ops-mnist.cpp`)
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
   (default: `~/.cache/ggml` or `GGML_HSA_KERNEL_CACHE_DIR`)

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

## Agent Rules

### Python Linting and Formatting

`ruff` is not part of the backend requirements. Install the dev tools first:

```bash
python3 -m pip install -r src/ggml-hsa/requirements-dev.txt
```

After modifying any Python file under `src/ggml-hsa/`, run:

```bash
ruff check src/ggml-hsa/
ruff format src/ggml-hsa/
```

Fix any issues reported by `ruff check` before considering the task complete.
The ruff configuration is in `src/ggml-hsa/ruff.toml`.

### Documentation Maintenance

After any change to the ggml-hsa codebase, review and update:

- `src/ggml-hsa/AGENTS.md` — Keep codebase structure, supported operations, conventions, and examples in sync with the code
- `src/ggml-hsa/README.md` — Keep user-facing documentation (supported operations, data types, build instructions, environment variables) in sync with the code

This includes but is not limited to: adding/removing operations, changing file structure, adding environment variables, modifying build options, or updating dependencies.

### Feature Branches

Create a feature branch for any new work. Do not commit directly to `master`, `main`, or `hsa-backend`.
