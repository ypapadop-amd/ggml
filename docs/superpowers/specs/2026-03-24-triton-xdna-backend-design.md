# Triton-XDNA Backend Support Design

**Date:** 2026-03-24
**Status:** Approved
**Author:** AI-assisted design

## Overview

Add Triton-XDNA as an optional compilation backend to the ggml-hsa system. The implementation supports per-operation backend selection, allowing IRON and Triton backends to coexist within a single execution. Triton-XDNA is optional at runtime with graceful degradation when not available.

## Background

The ggml-hsa backend currently supports only the IRON (MLIR-AIE) compilation backend for generating NPU kernels. Triton-XDNA is an alternative compilation framework that:

- Compiles Triton kernels to AMD XDNA NPUs using MLIR-AIR/AIE
- Achieves performance parity with handwritten implementations for dense matrix multiplication
- Supports matmul, elementwise operations, softmax, and layer normalization
- Uses the same underlying MLIR-AIE infrastructure as IRON

Adding Triton as a backend option enables:

1. Per-operation backend selection based on performance characteristics
2. Future experimentation with Triton's high-level kernel language
3. Gradual migration of operations from IRON to Triton where beneficial
4. Flexibility to choose the best compilation approach per operation

## Requirements

### Functional Requirements

1. **Optional Dependency**: Triton-XDNA must be installable independently of IRON
2. **Graceful Degradation**: System continues to work with IRON-only installation
3. **Per-Operation Selection**: Different operations can use different backends in the same execution
4. **Backend Coexistence**: IRON and Triton backends available simultaneously when both installed
5. **Clear Error Messages**: Users receive actionable guidance when Triton is needed but not installed

### Non-Functional Requirements

1. **Backward Compatibility**: Existing code and workflows continue unchanged
2. **No Build-Time Configuration**: Backend availability determined at runtime via imports
3. **Minimal Code Changes**: Leverage existing two-layer dispatch architecture
4. **Storage Efficiency**: Users only install what they need

## Dependency Relationship

The dependency hierarchy is:

- **IRON only**: `mlir-aie` + `llvm-aie`
- **Triton**: `triton-xdna` (includes `mlir-aie`, `llvm-aie`, `mlir-air` as dependencies)

**Key insight**: Installing Triton automatically provides IRON dependencies. Users never need separate IRON+Triton installations.

## Design

### Requirements Files

Split the single `requirements.txt` into two backend-specific files:

**`requirements-iron.txt`** (IRON-only, default):
```
--extra-index-url https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.1
--extra-index-url https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
--extra-index-url https://pypi.org/simple

mlir_aie==1.3.1
llvm-aie
black
```

**`requirements-triton.txt`** (Triton + IRON dependencies):
```
--extra-index-url https://pypi.org/simple

triton-xdna
--find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels
--find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti
--find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
--find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

black
```

**Migration**: Remove existing `requirements.txt` and update documentation to reference backend-specific files.

### env_setup.sh Interface

Update `env_setup.sh` to accept an optional comma-separated backend list:

```bash
# Default (IRON only)
source env_setup.sh

# Explicit IRON
source env_setup.sh iron

# Triton (includes IRON)
source env_setup.sh triton

# Both (redundant but supported)
source env_setup.sh iron,triton
```

**Implementation details:**

1. Parse first argument as comma-separated backend list (default: `iron`)
2. For each backend, install from `requirements-<backend>.txt`
3. Create/activate venv at `.venv`, upgrade pip, install dependencies
4. Validate that requested requirements files exist

**Backward compatibility**: No argument defaults to `iron`, preserving existing behavior.

### Kernel Build System Changes

**Add Backend.TRITON to `kernels/kernel.py`:**

```python
from enum import Enum, auto

class Backend(Enum):
    IRON = auto()
    TRITON = auto()
```

**Create `kernels/build_triton.py`:**

Skeleton compiler that validates imports and structure but defers actual compilation:

```python
"""Triton-XDNA backend compiler for ggml-hsa kernels."""

import logging
from pathlib import Path
from .kernel import KernelSpec

def compile_triton_kernel(
    kernel_spec: KernelSpec,
    work_dir: Path,
    exported_name: str,
    output_directory: Path,
    logger: logging.Logger,
    verbose: bool,
) -> None:
    """Compile a kernel using the Triton-XDNA backend.

    Args:
        kernel_spec: Kernel specification with backend, tensors, and function
        work_dir: Temporary working directory for intermediate files
        exported_name: Name for the exported kernel
        output_directory: Directory for final PDI and instruction files
        logger: Logger instance for compilation messages
        verbose: Enable verbose compilation output

    Raises:
        ImportError: If triton-xdna package not installed
        NotImplementedError: Compilation not yet implemented (skeleton only)
    """
    # Validate Triton-XDNA availability
    try:
        import triton
        # TODO: Import actual Triton-XDNA compilation modules
    except ImportError as e:
        error_msg = (
            "Triton backend requested but triton-xdna package not found.\n"
            "Install Triton support with:\n"
            "    source src/ggml-hsa/env_setup.sh triton"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e

    # Log kernel information
    logger.info(f"Triton compilation requested for {kernel_spec.op_name}")
    logger.info(f"  Architecture: {kernel_spec.arch}")
    logger.info(f"  Input tensors: {len(kernel_spec.input_tensors)}")
    logger.info(f"  Output tensor: {kernel_spec.output_tensor.shape}")

    # Validate kernel_spec structure
    assert kernel_spec.backend == Backend.TRITON, \
        f"Expected TRITON backend, got {kernel_spec.backend}"
    assert kernel_spec.function is not None, \
        "KernelSpec.function must be provided"
    assert kernel_spec.arch in ["aie2", "aie2p"], \
        f"Unsupported architecture: {kernel_spec.arch}"

    # Placeholder for actual compilation
    raise NotImplementedError(
        f"Triton kernel compilation not yet implemented for {kernel_spec.op_name}.\n"
        f"Backend infrastructure is in place. Kernel implementation needed in kernels/triton/."
    )
```

**Register compiler in `kernels/build.py`:**

```python
from .build_iron import compile_iron_kernel
from .build_triton import compile_triton_kernel
from .kernel import Backend

_compilers = {
    Backend.IRON: compile_iron_kernel,
    Backend.TRITON: compile_triton_kernel,
}
```

### Graceful Degradation Strategy

The system handles missing Triton installation at multiple levels:

1. **Import time**: `build_triton.py` imports succeed (no Triton imports at module level)
2. **Registration time**: `Backend.TRITON` always exists in enum, compiler always registered
3. **Compilation time**: `compile_triton_kernel()` raises `ImportError` with helpful message
4. **Error propagation**: Compilation failure surfaces to user with installation instructions

**Advantages:**

- No conditional logic in `build.py` or dispatch functions
- Dispatch functions can freely return `Backend.TRITON`
- Clear error messages guide users to solution
- No runtime overhead for backend detection

### Initial Backend Coverage

**All operations remain on IRON initially**. The Triton backend exists but is not used by any dispatch function. This provides:

- Safe infrastructure rollout with zero risk to existing functionality
- Gradual migration path: operations can opt-in one at a time
- Testing infrastructure without changing behavior
- Clear separation of infrastructure (this work) from kernel implementation (future work)

**Future work**: Modify dispatch functions to return `Backend.TRITON` and implement corresponding kernel generators in `kernels/triton/` directory.

### Documentation Updates

**README.md changes:**

1. Update "MLIR-AIE (IRON)" section title to "Compilation Backends"
2. Add Triton-XDNA as supported backend alongside IRON
3. Update environment setup examples:
   ```bash
   # IRON only (default)
   source src/ggml-hsa/env_setup.sh

   # Triton + IRON
   source src/ggml-hsa/env_setup.sh triton
   ```
4. Add note that Triton is optional and includes IRON dependencies

**AGENTS.md changes:**

1. Update "Project Overview" backend list to include TRITON
2. Update "Backend Compilers" section with Triton entry
3. Refresh "Adding a New Compilation Backend" example (currently uses Triton hypothetically)
4. Document `requirements-iron.txt` vs `requirements-triton.txt` split
5. Update "Environment Setup" section with new `env_setup.sh` usage
6. Add section on backend selection strategy

### Error Messages

**When Triton requested but not installed:**

```
ERROR: Triton backend requested but triton-xdna package not found.
Install Triton support with:
    source src/ggml-hsa/env_setup.sh triton
```

**When Triton backend infrastructure incomplete:**

```
NotImplementedError: Triton kernel compilation not yet implemented for GGML_OP_<operation>.
Backend infrastructure is in place. Kernel implementation needed in kernels/triton/.
```

## Implementation Plan

The implementation consists of these independent tasks:

1. **Requirements split**:
   - Create `requirements-iron.txt` (copy of current `requirements.txt`)
   - Create `requirements-triton.txt` with Triton dependencies
   - Remove `requirements.txt`

2. **env_setup.sh enhancement**:
   - Add argument parsing for comma-separated backend list
   - Map backends to requirements files
   - Install from appropriate requirements files
   - Validate requested files exist

3. **Backend enum expansion**:
   - Add `Backend.TRITON` to `kernels/kernel.py`

4. **Triton compiler skeleton**:
   - Create `kernels/build_triton.py` with skeleton implementation
   - Implement import validation and error messages
   - Validate `kernel_spec` structure
   - Raise `NotImplementedError` at compilation step

5. **Compiler registration**:
   - Import and register `compile_triton_kernel` in `kernels/build.py`

6. **Documentation updates**:
   - Update README.md with backend information
   - Update AGENTS.md with new backend details

7. **Testing**:
   - Verify IRON-only installation works
   - Verify Triton installation works
   - Verify helpful error when Triton missing but requested
   - Verify existing tests pass unchanged

## Migration Path

**Immediate impact**: None. All existing code continues working unchanged.

**Future migration**:

1. Add Triton kernel implementation directory: `kernels/triton/`
2. Implement kernel generators (e.g., `kernels/triton/matmul.py`)
3. Modify dispatch functions to return `Backend.TRITON` for selected operations
4. Implement actual compilation in `compile_triton_kernel()`
5. Benchmark IRON vs Triton for each operation
6. Gradually migrate operations based on performance data

**Rollback strategy**: Since all operations stay on IRON, rollback is simple - remove Triton-specific code and revert to single `requirements.txt`.

## Testing Strategy

**Unit tests:**

- `env_setup.sh` parses backend arguments correctly
- `build_triton.py` import validation works
- Error messages are clear and actionable

**Integration tests:**

- IRON-only installation: `source env_setup.sh iron`
- Triton installation: `source env_setup.sh triton`
- Backend enum includes both IRON and TRITON
- Existing kernel compilation tests pass unchanged
- Attempting to use TRITON backend raises clear `NotImplementedError`

**Manual validation:**

- README instructions work for both backends
- Error messages guide users correctly
- No regressions in existing IRON functionality

## Open Questions

None. Design approved for implementation.

## Future Work

1. **Triton kernel implementations**: Create actual kernel generators in `kernels/triton/`
2. **Compilation pipeline**: Implement PDI/instruction generation in `compile_triton_kernel()`
3. **Performance benchmarking**: Compare IRON vs Triton for each operation
4. **Backend selection heuristics**: Develop rules for automatic backend selection
5. **Mixed backend optimization**: Optimize data movement when operations use different backends

## References

- [Triton-XDNA README](https://github.com/amd/Triton-XDNA/blob/main/README.md)
- [MLIR-AIE v1.3.1](https://github.com/Xilinx/mlir-aie/tree/v1.3.1)
- [ggml-hsa AGENTS.md](../../src/ggml-hsa/AGENTS.md)
