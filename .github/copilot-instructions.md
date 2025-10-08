# GitHub Copilot Instructions for GGML

## Project Overview

GGML is a tensor library for machine learning with a focus on:
- Low-level cross-platform implementation
- Integer quantization support for efficient model inference
- Broad hardware support (CPU, CUDA, Metal, HIP/HSA, SYCL, Vulkan, WebGPU, OpenCL)
- Automatic differentiation
- Zero memory allocations during runtime
- No third-party dependencies for core functionality

**Note:** This project is under active development. Core library development primarily happens in the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) repositories.

## Build System

### CMake Configuration

- **Minimum CMake version:** 3.14
- **Languages:** C (C11), C++ (C++17), Assembly
- **Default build type:** Release (if not specified)
- **Shared libraries:** Default ON (except MINGW/Emscripten/WASM)

### Building the Project

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

### Key CMake Options

- `BUILD_SHARED_LIBS` - Build shared libraries (default: ON except Windows/MINGW)
- `GGML_BUILD_TESTS` - Build test suite (default: ON when standalone)
- `GGML_BUILD_EXAMPLES` - Build example programs (default: ON when standalone)
- `GGML_CUDA` - Enable CUDA backend
- `GGML_METAL` - Enable Metal backend (default: ON for Apple platforms)
- `GGML_HIP` - Enable HIP backend
- `GGML_HSA` - Enable HSA backend
- `GGML_SYCL` - Enable SYCL backend
- `GGML_VULKAN` - Enable Vulkan backend
- `GGML_BLAS` - Enable BLAS support

## Coding Standards

### Code Style

- **Indentation:** 4 spaces (see `.editorconfig`)
- **Line endings:** LF (Unix-style)
- **Charset:** UTF-8
- **Final newline:** Required
- **Trailing whitespace:** Remove

### Formatting Tools

- A `.clang-format` file exists in `src/ggml-hsa/` based on LLVM style
- **Column limit:** 100 characters
- **Pointer alignment:** Middle (e.g., `int * ptr`)
- **Brace style:** Attach

### Naming Conventions

- Public API functions: `ggml_*` prefix
- Backend-specific functions: `ggml_<backend>_*` (e.g., `ggml_cuda_*`, `ggml_metal_*`)
- Types: `struct ggml_*`
- Enums: `GGML_*` (uppercase with underscores)

## Architecture

### Directory Structure

```
├── include/          # Public headers (ggml.h, ggml-*.h, gguf.h)
├── src/              # Core implementation and backend implementations
│   ├── ggml.c       # Core tensor library
│   ├── ggml-cpu/    # CPU-specific optimizations
│   ├── ggml-cuda/   # CUDA backend
│   ├── ggml-metal/  # Metal backend
│   ├── ggml-hip/    # HIP backend
│   ├── ggml-hsa/    # HSA backend
│   └── ...          # Other backends
├── examples/         # Example applications (GPT-2, GPT-J, MNIST, SAM, etc.)
├── tests/            # Test suite
├── cmake/            # CMake modules
├── scripts/          # Utility scripts
└── docs/             # Documentation (GGUF format spec)
```

### Key Components

- **ggml.h/ggml.c** - Core tensor operations and compute graph
- **ggml-backend.h** - Backend abstraction layer
- **ggml-alloc.h** - Memory allocation utilities
- **gguf.h** - GGUF file format for model serialization
- **Backend implementations** - Hardware-specific optimizations

## Testing

### Running Tests

```bash
cd build
ctest --output-on-failure
```

### Test Organization

- Unit tests in `tests/` directory
- Backend-specific tests in `tests/ggml-<backend>/`
- Test naming: `test-*.c` or `test-*.cpp`
- Use CTest for test execution

### Writing Tests

- Follow existing test patterns in `tests/` directory
- Test both correctness and performance where applicable
- Include edge cases and boundary conditions
- Backend tests should verify backend-specific functionality

## Contributing Guidelines

⚠️ **Important:** For changes to the core `ggml` library (including CMake build system):
- Open a PR in https://github.com/ggml-org/llama.cpp first
- This ensures better visibility, testing, and review
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for details

### Pull Request Process

1. Ensure code follows the established style
2. Add or update tests as needed
3. Verify all tests pass locally
4. Update documentation if changing public APIs
5. Keep changes focused and minimal

## Common Tasks

### Adding a New Backend

1. Create `src/ggml-<backend>/` directory
2. Implement backend interface defined in `ggml-backend.h`
3. Add CMakeLists.txt with appropriate options
4. Create public header `include/ggml-<backend>.h`
5. Add tests in `tests/ggml-<backend>/`
6. Update main CMakeLists.txt with new options

### Adding New Tensor Operations

1. Add operation to `enum ggml_op` in `include/ggml.h`
2. Implement forward pass in `src/ggml.c`
3. Implement backward pass (gradient) if needed
4. Add operation to backend implementations
5. Add comprehensive tests
6. Update documentation

### Optimizing Existing Operations

1. Profile to identify bottlenecks
2. Consider SIMD/vectorization opportunities (see `src/ggml-cpu/`)
3. Implement backend-specific optimizations
4. Add performance tests
5. Verify correctness with existing tests

## Backend-Specific Notes

### CUDA Backend
- Use `ggml_cuda.h` for CUDA-specific APIs
- CUDA kernels in `src/ggml-cuda/`

### Metal Backend
- macOS/iOS GPU acceleration
- Shaders in Metal Shading Language
- Default ON for Apple platforms

### HIP/HSA Backends
- AMD GPU support
- Use appropriate compiler flags for ROCm

### CPU Backend
- SIMD optimizations in `src/ggml-cpu/`
- Multiple implementations for different architectures
- llamafile integration for optimized matrix multiplication

## Python Bindings

Python bindings are available in `examples/python/`:
- Auto-generated using CFFI
- Support for quantized tensors with automatic conversion
- See `examples/python/README.md` for usage

## Resources

- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGUF file format specification](../docs/gguf.md)
- [llama.cpp project](https://github.com/ggerganov/llama.cpp) - Primary development hub
- [whisper.cpp project](https://github.com/ggerganov/whisper.cpp) - Speech recognition with ggml

## Important Reminders

1. **Minimal changes**: Make surgical, focused changes
2. **Test early and often**: Run tests after each significant change
3. **Follow existing patterns**: Match the style and structure of existing code
4. **Consider performance**: GGML is performance-critical; profile changes
5. **Cross-platform**: Ensure changes work on Linux, macOS, and Windows
6. **Documentation**: Update comments and docs for public API changes
7. **Upstream first**: Core changes should go to llama.cpp repository first
