# JIT Compilation Process — ggml-hsa

## Overview

The ggml-hsa backend JIT-compiles GGML operations into AIE (AI Engine) kernels
at **tensor initialization time** — before graph execution begins. Each kernel
compiles to two binary artifacts:

| Artifact | Contents |
|---|---|
| `<name>.pdi` | Programmable Device Image (bitstream/configuration) |
| `<name>_insts.bin` | DMA instruction sequence (dword array) |

Two compilation backends exist: **IRON** (MLIR-AIE) and **Triton-XDNA**.

---

## End-to-End Flow

```
 GGML allocates tensor buffer
           │
           ▼
 ggml_backend_hsa_buffer_init_tensor()        ─── ggml-hsa.cpp
           │
           ▼
 ggml_backend_hsa_tensor_extra()              ─── ggml-hsa.cpp
   • flatten/normalize element-wise ops to 1D
   • substitute fp16 → bf16 on aie2/aie2p
           │
           ▼
 ggml_hsa_create_kernel_name()                ─── ggml-hsa.cpp
   → e.g. "add-1024bf16-1024bf16-1024bf16"
           │
           ▼
 ┌─ ggml_hsa_get_cached_kernel()               ─── ggml-hsa.cpp
 │  (unordered_map<string, shared_ptr<kernel>>)
 │    YES → done
 │    NO  ↓
 │
 │  ggml_hsa_create_kernel()                  ─── kernel-discovery.cpp
 │    dispatches by device type (HSA_DEVICE_TYPE_AIE)
 │         │
 │         ▼
 │  ggml_hsa_create_aie_kernel()              ─── kernel-discovery.cpp
 │    │
 │    ├─ precompiled dir ($GGML_HSA_KERNEL_DIR)?
 │    │    found → load from disk
 │    │
 │    ├─ disk cache?
 │    │    found → load from disk
 │    │
 │    └─ JIT compile (GGML_HSA_JIT_COMPILE)
 │         │
 │         ▼
 │    ggml_hsa_compile_aie_kernel()           ─── aie-kernel-compiler.cpp
 │      C++ → Python bridge (pybind11)
 │         │
 │         ▼
 │    build.ggml_compile_op()                 ─── kernels/build.py
 │      • _get_kernel(op_name) → look up in _OP_KERNEL_MAP
 │      • import dispatch module (e.g. mul_mat.py)
 │      • dispatch_fn() → KernelSpec or list[KernelSpec]
 │      • iterate specs: _get_compiler(backend) → try each backend
 │      • first successful compilation wins; all fail → log error
 │         │
 │         ├── IRON ──────────────────────────── kernels/build_iron.py
 │         │    1. function() → MLIR module (aie.iron DSL)
 │         │    2. compile C++ core functions via Peano/llvm-aie → .o
 │         │    3. compile_mlir_module() → .pdi + _insts.bin
 │         │
 │         └── Triton ────────────────────────── kernels/build_triton.py
 │              1. set_active(NPUDriver()) for npu1/npu2 target
 │              2. TempEnvSet: AMD_TRITON_NPU_DEBUG, AMD_TRITON_NPU_TARGET,
 │                 TRITON_CACHE_DIR
 │              3. config_context(compile_only=True,
 │                 transform_tiling_script=..., output_format="xclbin")
 │              4. function() → compiled_kernel (triggers Triton JIT)
 │              5. extract .pdi from xclbin via xclbinutil; copy _insts.bin
 │         │
 │         ▼
 │    artifacts written to cache_dir/<device>/
 │
 │  load from disk:
 │    ggml_hsa_load_pdi()                     ─── kernel-discovery.cpp
 │      hsa_amd_memory_pool_allocate(dev_memory) → read .pdi bytes
 │    ggml_hsa_load_insts()                   ─── kernel-discovery.cpp
 │      hsa_amd_memory_pool_allocate(dev_memory) → read _insts.bin
 │
 └─ ggml_hsa_cache_kernel()                   ─── ggml-hsa.cpp
      insert into in-memory map
           │
           ▼
 ═══════════════════════════════════════════
  At graph execution time
 ═══════════════════════════════════════════
           │
           ▼
 ggml_backend_hsa_graph_compute()             ─── ggml-hsa.cpp
   for each node:
     • optional CPU-side data transforms (requires_sync)
     • tensor_extra.kernel->dispatch(ctx, srcs, dst)
           │
           ▼
 ggml_hsa_aie_kernel::dispatch()              ─── aie-kernel.cpp
   • allocate payload from kernarg_memory pool
   • fill hsa_amd_aie_ert_start_kernel_data_t:
       pdi_addr, insts ptr+count, tensor ptrs+sizes
   • claim queue slot (hsa_queue_add_write_index_relaxed)
   • write vendor packet (HSA_AMD_PACKET_TYPE_AIE_ERT)
   • ring doorbell → AIE array executes
           │
           ▼
 ggml_hsa_wait_dispatches()                   ─── ggml-hsa.cpp
   hsa_signal_wait_scacquire(signal, EQ, 0)
   free kernargs
```

---

## Kernel Name Generation

`ggml_hsa_create_kernel_name()` in `ggml-hsa.cpp` builds a deterministic
cache key encoding:

- Operation name (lowercased): `add`, `mul_mat`, `soft_max`, ...
- Output tensor: shape + dtype + non-contiguous flag (e.g. `1024f32`, `3x3x4f32n`)
- Each source tensor in the same format, or `null`
- For non-unary ops with non-zero `op_params`: hex hash of the param bytes

**Flattening optimization:** Contiguous element-wise ops (ADD, SUB, MUL, DIV,
SCALE, all unary ops) are collapsed to 1D before naming, maximizing cache hits
across different shapes with identical element counts.

Example: `add-1024bf16-1024bf16-1024bf16`

---

## Compilation Backends

### IRON (MLIR-AIE)

The primary backend. Kernel authors write Python functions that use the
`aie.iron` DSL to construct MLIR modules describing the AIE array
configuration — tiles, object FIFOs, compute cores, and DMA sequences.

**C++ core functions** (vectorized compute kernels in `.cc` files like
`unary_ops.cc`, `binary_ops.cc`, `mm.cc`) are compiled to `.o` via
**Peano/llvm-aie** with operation-specific defines:

| Op type | Defines |
|---|---|
| Unary | `-D<OP_NAME>=1 -DINPUT_DTYPE=... -DOUTPUT_DTYPE=...` |
| Binary | `-D<OP_NAME>=1` or `-D<OP_NAME>_BROADCAST=1 -DINPUT0_DTYPE=... -DINPUT1_DTYPE=... -DOUTPUT_DTYPE=...` |
| GEMM | `-DDIM_M=N -DDIM_N=N -DDIM_K=N -D<input_dtype>_<output_dtype>_ONLY -DB_COL_MAJ -DC_COL_MAJ` |

The MLIR module is then lowered through MLIR-AIE passes
(`--alloc-scheme=basic-sequential`) to produce the final `.pdi` and
`_insts.bin` files.

### Triton-XDNA

Alternative backend using the Triton-XDNA stack with `NPUDriver`. The
architecture string is mapped to a Triton target (`aie2` → `npu1`,
`aie2p` → `npu2`) via `_get_triton_target()` (`build_triton.py`).

Compilation runs inside a combined context:

- `TempEnvSet` (`build_triton.py`) temporarily sets `AMD_TRITON_NPU_DEBUG`,
  `AMD_TRITON_NPU_TARGET`, and `TRITON_CACHE_DIR`
- `config_context()` (from `triton.backends.amd_triton_npu.config`) sets
  `compile_only=True`, `transform_tiling_script` (from
  `KernelSpec.config["transform_script"]`), and `output_format="xclbin"`

Calling `kernel_spec.function()` inside this context triggers the Triton JIT
compiler. The output xclbin is located via `get_npu_cache_dir()`, the `.pdi`
is extracted with `xclbinutil`, and `insts.bin` is copied to the output
directory.

---

## Caching

### Lookup Order

1. **In-memory cache** — `unordered_map<string, shared_ptr<ggml_hsa_kernel>>`
   in `ggml_hsa_device_info::device_info::kernels`. Zero-cost on repeated use.

2. **Precompiled directory** — `$GGML_HSA_KERNEL_DIR/<device>/<name>.pdi`.
   For shipping pre-built kernels. Checked before the disk cache.

3. **Disk cache** — resolved by priority:
   - `$GGML_HSA_KERNEL_CACHE_DIR`
   - `$XDG_CACHE_HOME/ggml`
   - `$HOME/.cache/ggml`
   - `/tmp/ggml/ggml-hsa`

4. **JIT compile** — only if `GGML_HSA_JIT_COMPILE` is enabled (default ON).

### Eviction

`ggml_hsa_purge_unused_cached_kernels()` is called from the backend context
destructor. It removes in-memory entries where `use_count() == 1` (no tensor
still holds a reference).

---

## HSA Dispatch

Dispatch uses the AMD vendor-specific packet type `HSA_AMD_PACKET_TYPE_AIE_ERT`
with opcode `HSA_AMD_AIE_ERT_START_CU`.

The payload (`hsa_amd_aie_ert_start_kernel_data_t`) contains:

| Field | Content |
|---|---|
| `pdi_addr` | Pointer to PDI in dev_memory |
| `data[0..1]` | Transaction opcode (`0x3, 0x0`) |
| `data[2..4]` | Instructions pointer (lo/hi 32-bit) + dword count |
| `data[5..N]` | Per-tensor: address (lo/hi) + byte size |

Payload memory is allocated from the `kernarg_memory` pool and tracked in
`ctx.kernargs` for deferred free after signal wait.

---

## Key Data Structures

| Structure | Location | Purpose |
|---|---|---|
| `ggml_hsa_kernel` | `common.hpp` | Abstract base with virtual `dispatch()` |
| `ggml_hsa_aie_kernel` | `aie-kernel.hpp` | Holds PDI + instruction buffers |
| `ggml_hsa_pdi_buffer` | `aie-kernel.hpp` | HSA-allocated PDI bytes |
| `ggml_hsa_insts_buffer` | `aie-kernel.hpp` | HSA-allocated instruction dwords + count |
| `ggml_backend_hsa_tensor_extra` | `common.hpp` | Per-tensor: node_t (tensor+convert info), kernel, staging buffer, sync flag |
| `ggml_backend_hsa_context` | `common.hpp` | Queue, signal, pending payloads |
| `KernelSpec` | `kernels/kernel.py` | Python: backend, op_name, arch, tensors, function, config |
| `TensorDesc` | `kernels/tensor_desc.py` | Python: dtype, shape, stride, contiguity |

---

## Environment Variables

| Variable | Effect |
|---|---|
| `GGML_HSA_KERNEL_DIR` | Precompiled kernel directory (priority over cache) |
| `GGML_HSA_KERNEL_CACHE_DIR` | Override disk cache location |
| `GGML_HSA_KERNEL_CACHE_CLEAR` | Set `1` to clear cache on startup |
| `GGML_HSA_JIT_VERBOSE` | Verbose Python compiler output |
| `GGML_HSA_ENABLE_LOG` | Verbose C++ logging (defaults ON in debug builds) |
| `GGML_HSA_JIT_COMPILE` | CMake option (default ON): enable JIT compilation |

---

## Error Handling

- **Python compilation failure:** `py::error_already_set` is caught in
  `ggml_hsa_compile_aie_kernel()`, which returns `GGML_STATUS_FAILED`.
- **File load failure:** `ggml_hsa_load_pdi()` / `ggml_hsa_load_insts()`
  return `GGML_STATUS_ALLOC_FAILED` or `GGML_STATUS_FAILED`.
- **Kernel not found and JIT disabled:** returns `GGML_STATUS_FAILED`,
  causing `ggml_backend_hsa_tensor_extra` constructor to throw.
- **Duplicate cache insert:** `GGML_ABORT` — indicates a logic error.
- **supports_op probe:** constructs a temporary `tensor_extra` in a try/catch
  to test compilability without side effects.
