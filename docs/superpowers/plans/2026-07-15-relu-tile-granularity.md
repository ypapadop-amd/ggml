# RELU tile-granularity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the RELU IRON design's per-dispatch call count by streaming fewer, larger objectfifo tiles, amortizing the ~98% per-call overhead that dominates RELU device time.

**Architecture:** Add an arch-parameter dict + a non-pow2 tile-size selector to `utils.py`, then route RELU through a new `_unary_op_tiled` dataflow in `unary_ops.py` that streams `floor(N/tile)` full tiles plus one remainder tile. The RELU kernel is unchanged (its existing runtime-N path handles both full and remainder calls). Tile size stays a runtime kernel argument so no new compiled kernels are minted.

**Tech Stack:** Python (IRON / mlir-aie design), C++ AIE kernel (unchanged), on-device verification via `test-backend-ops-mnist` and `mnist-eval`.

## Global Constraints

- **32 unique functions per queue.** Tile size MUST remain a runtime `N` argument to the kernel — never a compile-time define or part of the kernel name. RELU must contribute exactly one compiled kernel per shape, unchanged from today.
- **Bit-exact correctness.** RELU op test 3/3 at 5e-4; MNIST accuracy 98.00%; MNIST test_loss bit-identical to `0.066372`.
- **DMA stays linear.** `rt.fill`/`rt.drain` move the whole tensor contiguously; no strided 2D shim descriptors.
- **L1 fit.** in+out fifos are double-buffered; their footprint must fit inside the per-arch core data memory budget (half of `core_data_mem_bytes`).
- **Arch constants live in a dict** (`_ARCH_PARAMS` in `utils.py`) so a new NPU generation is one entry. Known values: aie2 → 64 KB core data mem, 512-bit vectors; aie2p → 64 KB core data mem, 512-bit vectors.
- **Scope: RELU only.** Do not change the dataflow of other unary ops; `_unary_op` stays the path for everything except RELU. The `max_tile_size` refactor is value-preserving (still returns the same numbers).
- **Environment:** build dir `build/`; activate `build/.venv` before running on-device tests; move `~/.cache/ggml/aie2/*relu*` aside after editing the design to force a JIT recompile (`rm` is blocked — use `mv`). Build type must be Release (already configured).

---

## File Structure

- `src/ggml-hsa/kernels/iron_kernels/utils.py` — add `_ARCH_PARAMS` dict, `_arch_params()` accessor, `tiled_tile_size()`; refactor `max_tile_size()` to read the vector width from the dict (value-preserving).
- `tests/ggml-hsa/test_utils_tiling.py` — NEW pure-Python unit tests for `_arch_params`, `tiled_tile_size`, and `max_tile_size` invariants (no device needed).
- `src/ggml-hsa/kernels/iron_kernels/unary_ops.py` — add `_unary_op_tiled()` and a RELU route in `unary_op()`; add a tiled variant of the external-function factory that does NOT pass `-DGGML_TILE_SIZE`.
- `src/ggml-hsa/kernels/iron_kernels/unary_ops.cc` — UNCHANGED (verify the runtime-N path is used; no edit expected).

---

## Task 1: Arch-parameter dict + refactor max_tile_size

**Files:**
- Modify: `src/ggml-hsa/kernels/iron_kernels/utils.py`
- Test: `tests/ggml-hsa/test_utils_tiling.py` (create)

**Interfaces:**
- Produces:
  - `_ARCH_PARAMS: dict[str, dict[str, int]]` with keys `"aie2"`, `"aie2p"`, each `{"core_data_mem_bytes": 65536, "vector_reg_bits": 512}`.
  - `_arch_params(arch: str) -> dict[str, int]` — returns the entry or raises `ValueError` for unknown arch.
  - `max_tile_size(arch, dtype, num_elements) -> int` — unchanged signature and return values; internally reads `vector_reg_bits` from the dict.

- [ ] **Step 1: Write the failing test**

Create `tests/ggml-hsa/test_utils_tiling.py`:

```python
import numpy as np
import pytest

from ggml_hsa_kernels.iron_kernels.utils import (
    _ARCH_PARAMS,
    _arch_params,
    max_tile_size,
)


def test_arch_params_has_known_archs():
    for arch in ("aie2", "aie2p"):
        p = _arch_params(arch)
        assert p["core_data_mem_bytes"] == 64 * 1024
        assert p["vector_reg_bits"] == 512


def test_arch_params_unknown_raises():
    with pytest.raises(ValueError):
        _arch_params("nope")


def test_max_tile_size_unchanged_f32_250000():
    # 250000 = 2^4 * 5^6 -> largest pow2 divisor within 512 bits is 16
    assert max_tile_size("aie2", np.dtype(np.float32), 250000) == 16


def test_max_tile_size_unchanged_pow2():
    # 2048 f32 -> full 512-bit tile (16) divides evenly, so 16
    assert max_tile_size("aie2", np.dtype(np.float32), 2048) == 16
```

Note on import path: the kernels package is imported as it is elsewhere in the repo. Before writing, confirm the exact import root by running:
`python -c "import importlib; importlib.import_module('ggml_hsa_kernels.iron_kernels.utils')"` from the build venv; if that module path is wrong, adjust the import in the test to match how `unary_ops.py` imports `from .utils import ...` (e.g. add the kernels dir to `sys.path` and `import iron_kernels.utils`). Use whichever import the repo's existing python tests use; if there are none, `sys.path.insert(0, "src/ggml-hsa/kernels")` then `from iron_kernels.utils import ...`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `source build/.venv/bin/activate && python -m pytest tests/ggml-hsa/test_utils_tiling.py -v`
Expected: FAIL — `ImportError: cannot import name '_ARCH_PARAMS'` / `_arch_params`.

- [ ] **Step 3: Add the dict + accessor and refactor max_tile_size**

In `utils.py`, add near the top (after imports):

```python
# Per-architecture on-tile resources. Add a new NPU generation by adding one entry.
_ARCH_PARAMS = {
    "aie2": {"core_data_mem_bytes": 64 * 1024, "vector_reg_bits": 512},  # NPU1/Phoenix (AIE-ML)
    "aie2p": {"core_data_mem_bytes": 64 * 1024, "vector_reg_bits": 512},  # NPU2/Strix (XDNA2)
}


def _arch_params(arch: str) -> dict:
    """Return the on-tile resource parameters for an architecture.

    Parameters:
        arch: Target architecture.

    Returns:
        The parameter dict for the architecture.

    Raises:
        ValueError: If the architecture is unknown.

    """
    params = _ARCH_PARAMS.get(arch)
    if params is None:
        msg = f"Unsupported architecture: {arch}"
        raise ValueError(msg)
    return params
```

Then change `max_tile_size` to read the width from the dict. Replace its body's width lookup:

```python
def max_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """Largest power-of-two tile within a 512-bit vector dividing num_elements.

    Parameters:
        arch: Target architecture.
        dtype: Element data type.
        num_elements: Total number of elements to tile.

    Returns:
        The chosen tile size.

    """
    vector_register_width = _arch_params(arch)["vector_reg_bits"]
    tile_size = int(vector_register_width / dtype.itemsize)

    while num_elements % tile_size != 0 and tile_size > 1:
        tile_size //= 2

    assert num_elements % tile_size == 0, (
        f"Number of elements ({num_elements}) must be a multiple of "
        f"tile size ({tile_size})."
    )

    return tile_size
```

(This removes the local `if arch in {"aie2", "aie2p"}` width block; `_arch_params` now raises for unknown archs, preserving the old error behavior.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `source build/.venv/bin/activate && python -m pytest tests/ggml-hsa/test_utils_tiling.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/ggml-hsa/kernels/iron_kernels/utils.py tests/ggml-hsa/test_utils_tiling.py
git commit -m "Add arch-parameter dict and route max_tile_size vector width through it"
```

---

## Task 2: tiled_tile_size selector

**Files:**
- Modify: `src/ggml-hsa/kernels/iron_kernels/utils.py`
- Test: `tests/ggml-hsa/test_utils_tiling.py` (extend)

**Interfaces:**
- Consumes: `_arch_params` (Task 1).
- Produces: `tiled_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int` — the largest multiple of the vector width V whose in+out double-buffered fifos fit half the core data memory, capped at the largest multiple of V ≤ num_elements, floored at V.

- [ ] **Step 1: Write the failing test**

Append to `tests/ggml-hsa/test_utils_tiling.py`:

```python
from ggml_hsa_kernels.iron_kernels.utils import tiled_tile_size


def test_tiled_tile_size_f32_mnist():
    # aie2 f32: V=16, budget=32768 bytes, 4 buffers (in+out, depth 2) of tile*4 bytes.
    # max_by_mem = (32768 // (4*4) // 16) * 16 = (2048 // 16) * 16 = 2048
    assert tiled_tile_size("aie2", np.dtype(np.float32), 250000) == 2048


def test_tiled_tile_size_multiple_of_vector_width():
    t = tiled_tile_size("aie2", np.dtype(np.float32), 250000)
    assert t % 16 == 0


def test_tiled_tile_size_capped_by_num_elements():
    # tiny tensor: capped at largest multiple of V <= N
    assert tiled_tile_size("aie2", np.dtype(np.float32), 48) == 48
    assert tiled_tile_size("aie2", np.dtype(np.float32), 50) == 48  # floor to mult of 16


def test_tiled_tile_size_floor_is_vector_width():
    # N smaller than V still returns V (kernel handles sub-V via scalar tail)
    assert tiled_tile_size("aie2", np.dtype(np.float32), 8) == 16


def test_tiled_tile_size_unknown_arch_raises():
    with pytest.raises(ValueError):
        tiled_tile_size("nope", np.dtype(np.float32), 250000)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source build/.venv/bin/activate && python -m pytest tests/ggml-hsa/test_utils_tiling.py -v -k tiled`
Expected: FAIL — `ImportError: cannot import name 'tiled_tile_size'`.

- [ ] **Step 3: Implement the selector**

Add to `utils.py`:

```python
def tiled_tile_size(arch: str, dtype: np.dtype, num_elements: int) -> int:
    """Largest multiple-of-V tile whose in+out double-buffered fifos fit half the
    core data memory, capped at num_elements and floored at the vector width V.

    Parameters:
        arch: Target architecture.
        dtype: Element data type.
        num_elements: Total number of elements to tile.

    Returns:
        The chosen tile size (a multiple of the vector width in elements).

    """
    params = _arch_params(arch)
    v = params["vector_reg_bits"] // (8 * dtype.itemsize)
    budget = params["core_data_mem_bytes"] // 2  # half DM: leave room for stack + locals
    # in + out fifos, each double-buffered (depth 2) => 4 buffers of tile*itemsize bytes.
    max_by_mem = (budget // (4 * dtype.itemsize) // v) * v
    max_by_n = (num_elements // v) * v
    return max(v, min(max_by_mem, max_by_n))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source build/.venv/bin/activate && python -m pytest tests/ggml-hsa/test_utils_tiling.py -v`
Expected: all passed (9 total).

- [ ] **Step 5: Commit**

```bash
git add src/ggml-hsa/kernels/iron_kernels/utils.py tests/ggml-hsa/test_utils_tiling.py
git commit -m "Add tiled_tile_size selector for non-pow2 L1-budgeted tiles"
```

---

## Task 3: Tiled RELU dataflow (_unary_op_tiled) + RELU route

**Files:**
- Modify: `src/ggml-hsa/kernels/iron_kernels/unary_ops.py`

**Interfaces:**
- Consumes: `tiled_tile_size` (Task 2); existing `arch_aligned_num_elements`, `arch_to_device` (utils); `ExternalFunction`, `ObjectFifo`, `Program`, `Runtime`, `Worker`, `dtype_to_str`, `range_` (already imported in unary_ops.py).
- Produces: `_unary_op_tiled(arch, input_tensors, function_spec, output_tensor)` and `_create_tiled_external_function(arch, op_name, input_tensor, output_tensor) -> CoreFunctionSpec`. `unary_op()` routes `op_name == "GGML_UNARY_OP_RELU"` to these.

**Context:** `op_name` reaching `unary_op()` is the full string `"GGML_UNARY_OP_RELU"` (confirmed: `src/ggml-hsa/kernels/unary_ops.py:289`). The existing `_unary_op` (unary_ops.py:46) tiles with `max_tile_size` and passes `-DGGML_TILE_SIZE`; the tiled path must NOT pass that define so the kernel uses its runtime-N branch and the same kernel serves both full and remainder tile sizes. `CoreFunctionSpec` (unary_ops.py:27) is `@dataclass(frozen=True)` with fields `external_function` and `num_elements` and a `tile_size` property returning `external_function.tile_size(0)`.

- [ ] **Step 1: Add the tiled external-function factory**

In `unary_ops.py`, after `_create_external_function` (ends ~line 143), add:

```python
def _create_tiled_external_function(
    arch: str,
    op_name: str,
    input_tensor,
    output_tensor,
) -> CoreFunctionSpec:
    """Create a CoreFunctionSpec for a unary op with an L1-budgeted tile size.

    Unlike _create_external_function, this does NOT pass -DGGML_TILE_SIZE: the tile
    count varies per call (full tiles vs a remainder tail), so N stays a runtime
    argument and one compiled kernel serves every tile size (32-function budget).

    Parameters:
        arch: Target architecture.
        op_name: Name of the unary operation.
        input_tensor: Input tensor.
        output_tensor: Output tensor.

    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=input_tensor)
    tile_size = tiled_tile_size(arch, input_tensor.dtype, num_elements)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name=op_name.lower(),
        object_file_name=f"{op_name.lower()}_core_function.o",
        source_file=str(current_dir / "unary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            f"-D{op_name}=1",
            f"-DINPUT_DTYPE={dtype_to_str(input_tensor.dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)
```

Add the import for the selector at the top of the file (the existing line is `from .utils import arch_aligned_num_elements, arch_to_device, max_tile_size`):

```python
from .utils import (
    arch_aligned_num_elements,
    arch_to_device,
    max_tile_size,
    tiled_tile_size,
)
```

- [ ] **Step 2: Add the tiled dataflow**

After `_create_tiled_external_function`, add:

```python
def _unary_op_tiled(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """Element-wise output_tensor = op(input_tensors[0]) with large tiles.

    Streams floor(N/tile) full tiles plus one remainder tile (when N is not a
    multiple of tile), amortizing the per-call acquire/release overhead. The tile
    size comes from tiled_tile_size (L1-budgeted, non-power-of-two). The kernel is
    told the real element count per call via its runtime N argument, so the final
    short tile is handled by the kernel's own vector-interior + scalar-tail path.

    Parameters:
        arch: Target architecture.
        input_tensors: List of one input tensor.
        function_spec: Core function specification.
        output_tensor: Output tensor.

    """
    input_tensor = input_tensors[0]

    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size
    n_full = num_elements // tile_size
    rem = num_elements % tile_size

    input_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensor.dtype]]
    output_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]
    of_in = ObjectFifo(input_tile_ty, name="in")
    of_out = ObjectFifo(output_tile_ty, name="out")

    function = function_spec.external_function

    def ext_core_fn(of_in, of_out, function):
        for _ in range_(n_full):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)
        # Remainder tail: 0 or 1 extra call. Python-level `if` -> resolved at build
        # time (rem is a compile-time constant here), so no runtime branch.
        if rem:
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            function(elem_in, elem_out, rem)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(ext_core_fn, fn_args=[of_in.cons(), of_out.prod(), function])

    input_tensor_ty = np.ndarray[(num_elements,), np.dtype[input_tensor.dtype]]
    output_tensor_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]
    rt = Runtime()
    with rt.sequence(input_tensor_ty, output_tensor_ty) as t:
        rt.start(worker)
        rt.fill(of_in.prod(), t[0])
        rt.drain(of_out.cons(), t[-1], wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()
```

- [ ] **Step 3: Route RELU through the tiled path**

In `unary_op()` (unary_ops.py ~178), replace the tail (the `function_spec = _create_external_function(...)` + `return _unary_op(...)`) with a RELU branch:

```python
    if op_name == "GGML_UNARY_OP_RELU":
        function_spec = _create_tiled_external_function(
            arch=arch,
            op_name=op_name,
            input_tensor=input_tensors[0],
            output_tensor=output_tensor,
        )
        return _unary_op_tiled(
            arch=arch,
            input_tensors=input_tensors,
            function_spec=function_spec,
            output_tensor=output_tensor,
        )

    function_spec = _create_external_function(
        arch=arch,
        op_name=op_name,
        input_tensor=input_tensors[0],
        output_tensor=output_tensor,
    )

    return _unary_op(
        arch=arch,
        input_tensors=input_tensors,
        function_spec=function_spec,
        output_tensor=output_tensor,
    )
```

- [ ] **Step 4: Force a recompile and run the RELU op test**

```bash
mkdir -p ~/.cache/ggml/_stale && mv ~/.cache/ggml/aie2/*relu* ~/.cache/ggml/_stale/ 2>/dev/null; true
cmake --build build --target ggml-hsa -j"$(nproc)"
source build/.venv/bin/activate
GGML_HSA_ENABLE_LOG=1 "$PWD/build/bin/test-backend-ops-mnist" test -o RELU 2>&1 | grep -iE "RELU\(|tests passed|Backend HSA0"
```
Expected: `3/3 tests passed`, `Backend HSA0: OK`. If a kernel fails to compile, re-run with `GGML_HSA_JIT_VERBOSE=1` to see the traceback.

- [ ] **Step 5: Confirm no new kernels / runtime-N path**

```bash
# Exactly one compiled RELU kernel per shape (no per-tile-size proliferation):
GGML_HSA_ENABLE_LOG=1 "$PWD/build/bin/test-backend-ops-mnist" test -o RELU 2>&1 | grep -c "generated kernel relu"
# The .o must still contain the runtime-N division/tail (NOT the folded constexpr form):
PEANO=$(find build/.venv -name llvm-nm -path '*llvm-aie*' | head -1)
OBJ=$(find ~/.cache/ggml/aie2 -name ggml_unary_op_relu_core_function.o | head -1)
"$PEANO" --print-size "$OBJ" | grep relu
```
Expected: kernel-count line prints `3` (one per RELU shape, unchanged from before); the function exists. (Function will be larger than the 112-byte folded version — that is expected; the fold is intentionally not used on the tiled path.)

- [ ] **Step 6: Commit**

```bash
git add src/ggml-hsa/kernels/iron_kernels/unary_ops.py
git commit -m "Route RELU through a large-tile dataflow with a remainder tail"
```

---

## Task 4: End-to-end MNIST verification + measurement

**Files:** none (verification only).

**Interfaces:** Consumes the full pipeline from Task 3.

- [ ] **Step 1: MNIST correctness (accuracy + bit-identical loss)**

```bash
source build/.venv/bin/activate
cd examples/mnist
BIN=$PWD/../../build/bin/mnist-eval
IMG=data/MNIST/raw/t10k-images-idx3-ubyte
LBL=data/MNIST/raw/t10k-labels-idx1-ubyte
"$BIN" mnist-fc-f32.gguf "$IMG" "$LBL" HSA0 2>&1 | grep -iE "us/image|test_loss|test_acc"
```
Expected: `test_acc=98.00+-0.14%`, `test_loss=0.066372+-0.008765` (bit-identical). Record us/image.

- [ ] **Step 2: Measure e2e delta (median of 5)**

```bash
for i in 1 2 3 4 5; do "$BIN" mnist-fc-f32.gguf "$IMG" "$LBL" HSA0 2>&1 | grep -iE "us/image"; done
```
Expected: median below the ~51.5 us/image pre-tiling baseline. Record the number; if it did NOT improve, STOP and investigate (do not mark complete).

- [ ] **Step 3: Confirm the RELU per-op share dropped (optional profiler)**

If the per-node profiler is still available (env `GGML_MNIST_PROFILE_LAYERS`), run it and confirm RELU (UNARY) share fell from ~11.5%. If the profiler is not currently in the tree, skip — the e2e delta in Step 2 is the gate.

- [ ] **Step 4: Regression-check the max_tile_size consumers (independent)**

Run each separately so a regression is attributable to one op:
```bash
"$PWD/../../build/bin/test-backend-ops-mnist" test -o ADD 2>&1 | tail -3
"$PWD/../../build/bin/test-backend-ops-mnist" test -o MUL_MAT 2>&1 | tail -3
```
Expected: ADD passes; MUL_MAT `3/3 tests passed`. For clamp/scale/count_equal (no MNIST-harness case), run the standalone binary if present:
```bash
"$PWD/../../build/bin/test-backend-ops" test -o SCALE 2>&1 | tail -3 || echo "no SCALE gate"
"$PWD/../../build/bin/test-backend-ops" test -o CLAMP 2>&1 | tail -3 || echo "no CLAMP gate"
```
If a consumer has no runnable HSA test, note it in the commit/PR rather than claiming a gate. Because the max_tile_size change is value-preserving, all present gates must still pass.

- [ ] **Step 5: Commit the measurement note**

Update `docs/ggml-hsa-optimization-tracking.md` §7 (or append a short dated note) with the tiled-RELU e2e number and the call-count reduction (15625 → n_full+1). Then:
```bash
git add docs/ggml-hsa-optimization-tracking.md
git commit -m "Record RELU tile-granularity result"
```

---

## Notes for the implementer

- **Import root:** confirm how to import `iron_kernels.utils` from a pytest before writing Task 1's test (see the note in Task 1 Step 1). The repo may have no existing python tests for this package; if so, use `sys.path.insert(0, "src/ggml-hsa/kernels")` + `from iron_kernels.utils import ...` and drop the `ggml_hsa_kernels.` prefix.
- **`mv` not `rm`:** `rm` is permission-blocked in this environment; move stale cache dirs aside.
- **Recompile trigger:** after any `.py` design edit, move `~/.cache/ggml/aie2/*relu*` aside or the stale `.pdi` is reused and your change won't take effect.
- **If aiecc reports L1 pressure at tile=2048:** halve the budget fraction in `tiled_tile_size` (`core_data_mem_bytes // 4`), which yields tile=1024 (245 calls) — update the Task 2 test expectation accordingly and note the change.
```
