# JIT Compilation Process — ggml-hsa

## Overview

The ggml-hsa backend JIT-compiles GGML operations into AIE (AI Engine) kernels
at **tensor initialization time** — before graph execution begins. Each kernel
compiles to a PDI (Programmable Device Image) and a DMA instruction sequence,
which are packed into one artifact:

| Artifact | Contents |
|---|---|
| `<name>.hsaco` | HSA code object with an `aie2`/`aie2p` section holding one `PdiInsts` kernel named `<name>` (see ROCr `docs/aie-hsaco-format.md`) |

Packing uses `aie.compiler.hsaco.pack` from mlir-aie (`kernels/aie_hsaco.py`).

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
 │    ggml_hsa_compile_kernel()               ─── kernel-compiler.cpp
 │      C++ → Python bridge (pybind11); builds a CompilerConfig
 │         │
 │         ▼
 │    build.ggml_compile_op(config)           ─── kernels/build.py
 │      • _get_kernel(op_name) → look up in _OP_KERNEL_MAP
 │      • import dispatch module (e.g. mul_mat.py)
 │      • dispatch_fn() → KernelSpec or list[KernelSpec]
 │      • _make_kernel_specs() → order/filter by config.compilers
 │      • iterate specs: _get_compiler(backend) → try each backend
 │      • first successful compilation wins; all fail → log error
 │         │
 │         ├── IRON ──────────────────────────── kernels/build_iron.py
 │         │    1. function() → MLIR module (aie.iron DSL)
 │         │    2. compile C++ core functions via Peano/llvm-aie → .o
 │         │    3. compile_mlir_module() → .pdi + _insts.bin (in the work dir)
 │         │    4. pack into .hsaco; num_cols from the aie.device target model
 │         │
 │         └── Triton ────────────────────────── kernels/build_triton.py
 │              1. set_active(NPUDriver()) for npu1/npu2 target
 │              2. TempEnvSet: AMD_TRITON_NPU_DEBUG, AMD_TRITON_NPU_TARGET,
 │                 TRITON_CACHE_DIR
 │              3. config_context(compile_only=True,
 │                 transform_tiling_script=..., output_format="xclbin")
 │              4. function() → compiled_kernel (triggers Triton JIT)
 │              5. extract .pdi + partition from xclbin via xclbinutil
 │              6. pack .pdi + insts.bin into .hsaco; num_cols from column_width
 │         │
 │         ▼
 │    artifacts written to cache_dir/<device>/
 │
 │  load from disk:
 │    ggml_hsa_aie_kernel::load()             ─── aie-kernel.cpp
 │      hsa_code_object_reader_create_from_file(.hsaco)
 │      hsa_executable_load_agent_code_object() + freeze
 │      symbol <name> → HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
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
     • optional source pre-processing (per source: on-queue or CPU-side)
     • tensor_extra.kernel->dispatch(ctx, srcs, dst)
     • optional output post-processing (node.convert_dtype: on-queue or CPU-side)
           │
           ▼
 ggml_hsa_aie_kernel::dispatch()              ─── aie-kernel.cpp
   • claim queue slot (hsa_queue_add_write_index_relaxed)
   • fill the slot's kernargs: tensor ptrs, then tensor sizes
   • write hsa_amd_aie_kernel_dispatch_packet_t with the kernel object
   • ring doorbell once a batch is full → AIE array executes
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
- For non-unary ops with non-zero `op_params`: the 32-bit words in hex,
  dot-separated, leading zeros stripped and trailing zero words dropped
  (e.g. `1.1.1.1.1.1` for a stride-1 pad-1 dilation-1 convolution)

> **Note:** this field is injective, and deliberately so. A kernel may compile
> its `op_params` in as constants (`conv_2d` does, which is most of its
> speed-up). It was previously a hash of the param bytes; under a hash, two
> tensors with identical shapes but different `op_params` can collide and share
> a cached kernel. That is harmless while `op_params` are read at dispatch time,
> but for a kernel that compiled them in it silently computes the wrong result.
> Encoding the words means a mismatch can only ever be a cache miss.
>
> The encoding is shorter than the 16-character hash it replaced for every op in
> the tree, but it is not bounded in principle: an op using all 16 words with
> large values would reach ~143 characters.

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

The MLIR module is then lowered through MLIR-AIE passes to produce the final
`.pdi` and `_insts.bin` files, using MLIR-AIE's default buffer allocator. These
are packed into `<name>.hsaco`; the kernel's `num_cols` is the column count of
the device the module's `aie.device` targets (e.g. 1 for `npu1_1col`).

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
and the `AIE_PARTITION` JSON are extracted with `xclbinutil`, and the `.pdi` and
`insts.bin` are packed into `<name>.hsaco` in the output directory, with the
partition's `column_width` as `num_cols`.

---

## Caching

### Lookup Order

1. **In-memory cache** — `unordered_map<string, shared_ptr<ggml_hsa_kernel>>`
   in `ggml_hsa_device_info::device_info::kernels`. Zero-cost on repeated use.

2. **Precompiled directory** — `$GGML_HSA_KERNEL_DIR/<device>/<name>.hsaco`.
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

If a context is destroyed while work is still in flight on a suspended queue,
the destructor leaks the unretired packet's kernarg slot and sets the
per-device `kernels_pinned` flag instead of draining. Purging is skipped
entirely while `kernels_pinned` is set (checked at the top of
`ggml_hsa_purge_unused_cached_kernels()`), so that device's kernel cache is
never evicted from again for the life of the process — a leaked reference
into the cache from the unretired packet is what makes eviction unsafe there.

---

## HSA Dispatch

Dispatch writes an `hsa_amd_aie_kernel_dispatch_packet_t`
(`hsa/hsa_ext_amd_aie.h`, opcode `HSA_AMD_AIE_PACKET_OPCODE_KMQ`):

| Field | Content |
|---|---|
| `kernel_object_low/high` | Kernel object from the loaded hsaco symbol |
| `num_kernargs` | Number of tensors (sources, then destination) |
| `kernarg_address` | `2 * num_kernargs` `uint64_t`: tensor addresses, then tensor byte sizes |

Kernargs live in a fixed per-ring-slot pool (`ctx.kernargs`) allocated from the
`kernarg_memory` pool. The hsaco's `kernarg_size` is `16 * num_kernargs`.

---

## Key Data Structures

| Structure | Location | Purpose |
|---|---|---|
| `ggml_hsa_kernel` | `common.hpp` | Abstract base with virtual `dispatch()` |
| `ggml_hsa_aie_kernel` | `aie-kernel.hpp` | Owns the HSA executable loaded from the hsaco and its kernel object |
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
| `GGML_HSA_JIT_COMPILER_ORDER` | Comma-separated backend order (e.g. `iron,triton`, case-insensitive); reorders candidate specs and drops backends not listed. Unset/empty keeps the dispatch order. |
| `GGML_HSA_ENABLE_LOG` | Verbose C++ logging (defaults ON in debug builds) |
| `GGML_HSA_KERNEL_INLINE` | Have IRON kernels hand their core function to `aiecc` as textual LLVM IR (`ExternalFunction(inline=True)`) instead of a `.o`, so it inlines into the tile loop instead of being linked with `ld.lld` (`1`, `true`, or `on`). Not usable on every kernel; each kernel opts in only once verified to compile both ways. Default off. |
| `GGML_HSA_JIT_COMPILE` | CMake option (default ON): enable JIT compilation |

---

## Error Handling

- **Python compilation failure:** `py::error_already_set` is caught in
  `ggml_hsa_compile_kernel()`, which logs at error level and returns
  `GGML_STATUS_FAILED`. An invalid backend name in `GGML_HSA_JIT_COMPILER_ORDER`
  raises `ValueError` from `_make_kernel_specs()` and surfaces the same way; an
  order that drops every candidate for an op logs a warning before failing.
- **Code object load failure:** `ggml_hsa_aie_kernel::load()` logs the failing
  HSA call and returns `GGML_STATUS_FAILED`.
- **Kernel not found and JIT disabled:** returns `GGML_STATUS_FAILED`,
  causing `ggml_backend_hsa_tensor_extra` constructor to throw.
- **Duplicate cache insert:** `GGML_ABORT` — indicates a logic error.
- **supports_op probe:** constructs a temporary `tensor_extra` in a try/catch
  to test compilability without side effects.
