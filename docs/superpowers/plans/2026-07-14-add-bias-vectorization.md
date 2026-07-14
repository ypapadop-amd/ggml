# Vectorized ADD Bias Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scalar, `__divsi3`-ridden broadcast path for the MNIST FC bias add with a vectorized row-tiled kernel, cutting the ADD dispatch cost that dominates MNIST at ~55% of serialized dispatch time.

**Architecture:** Co-design of the IRON Python design (`binary_ops.py`) and the C++ compute kernel (`binary_ops.cc`). A new predicate in `binary_op()` routes the single-row bias-add (`out[row,i] = src0[row,i] + src1[i]`) to a new design that tiles on one dst row: `src1` (one `ne0`-element row) is loaded once and reused for every row; the kernel adds one row per call with a `V`-wide vector interior and a scalar tail (the in-tree `scale.cc` idiom). All other ops/patterns keep the existing scalar templates.

**Tech Stack:** IRON / mlir-aie (Python design), Peano/AIECC C++ AIE kernel (`aie_api/aie.hpp`), ggml-hsa backend, `test-backend-ops-mnist` for on-device correctness.

## Global Constraints

- **Scope: ADD single-row-bias only.** SUB/MUL/DIV, general 4D broadcast, and plain element-wise keep calling `transform_binary_broadcast_n` / `transform_binary_n` unchanged. No behavior change for them.
- **Predicate for the new path (all must hold):** `op_name == "GGML_OP_ADD"` AND `src1_shape != dst_shape` (broadcast) AND `src1_shape == (dst_ne0, 1, 1, 1)` AND `src0_shape == dst_shape`.
- **No `__divsi3` on any kernel path.** The only division is `N / V` by the `constexpr` `V`, once, outside the loop.
- **DMA stays linear.** All three fifos stream contiguous `ne0`-element rows (one object == one row). No strided/replicating tap (respects the known shim-DMA strided-transfer limit).
- **Numerics preserved.** f32 add is exact vs. the scalar path; correctness gate is the existing `test-backend-ops-mnist` tolerance (1e-7 f32 / 1e-4 bf16). MNIST end-to-end accuracy must stay 98.00%.
- **Entry-point ABI:** `extern "C"` symbol `ggml_op_add_bias`, gated by `#ifdef GGML_OP_ADD_BIAS`, signature `(const INPUT0_DTYPE* src0, const INPUT1_DTYPE* src1, OUTPUT_DTYPE* out, int32_t N)` where `N == ne0`. The Python `ExternalFunction` name/signature/`-D` flags must match.
- **Env / build:** activate `build/.venv`; use absolute path to binaries. `-DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON`. Kernels cache at `~/.cache/ggml/aie2/`; `mv` stale artifacts aside (never `rm` — blocked in this env) to force recompile. `GGML_HSA_ENABLE_LOG=1` surfaces the compiled kernel name and any swallowed compile exception.
- Both `binary_ops.cc` and `binary_ops.py` are already enumerated in `src/ggml-hsa/kernels/iron_kernels/CMakeLists.txt` and `src/ggml-hsa/kernels/CMakeLists.txt`. **No CMake edits needed.** They copy to `build/src/ggml-hsa/kernels/...` as a `ggml-hsa` build dependency.

---

## Task 1: Add the vectorized `ggml_op_add_bias` C++ compute kernel

**Files:**
- Modify: `src/ggml-hsa/kernels/iron_kernels/binary_ops.cc` (add include at top; add one `#ifdef` block inside `extern "C"`)

**Interfaces:**
- Consumes: nothing (leaf compute kernel). Compile-time macros `INPUT0_DTYPE`, `INPUT1_DTYPE`, `OUTPUT_DTYPE`, `GGML_OP_ADD_BIAS` supplied by the Python `ExternalFunction` in Task 2.
- Produces: `extern "C" void ggml_op_add_bias(const INPUT0_DTYPE* __restrict src0, const INPUT1_DTYPE* __restrict src1, OUTPUT_DTYPE* __restrict out, int32_t N)` — adds one `N`-element row: `out[i] = src0[i] + src1[i]`.

- [ ] **Step 1: Add the `aie_kernel_utils.h` include**

At the top of `binary_ops.cc`, the current includes are just `#include "ggml-aie.hpp"` (line 11). Change to add the loop-macro header (needed for `AIE_PREPARE_FOR_PIPELINING` / `AIE_LOOP_MIN_ITERATION_COUNT`):

```cpp
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"
```

- [ ] **Step 2: Add the `ggml_op_add_bias` block**

Inside the `extern "C" {` block, after the `#ifdef GGML_OP_DIV_BROADCAST` … `#endif` section (i.e. after the existing broadcast entry points, before the closing `} // extern "C"`), add:

```cpp
#ifdef GGML_OP_ADD_BIAS

/**
 * @brief Row bias add: out[i] = src0[i] + src1[i] for one dst row.
 *
 * src1 is a single bias row (N == ne0 elements) reused across all dst rows;
 * the IRON design streams one src0/out row per call. Vectorized over a
 * V-wide interior with a scalar tail (N need not be a multiple of V).
 *
 * @param[in]  src0 First input row of N elements.
 * @param[in]  src1 Bias row of N elements.
 * @param[out] out  Output row of N elements.
 * @param[in]  N    Elements per row (== ne0).
 */
void ggml_op_add_bias(const INPUT0_DTYPE * __restrict src0,
                      const INPUT1_DTYPE * __restrict src1,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N) {
    event0();

    constexpr int32_t V = 512 / (sizeof(OUTPUT_DTYPE) * 8);
    const int32_t vend = (N / V) * V; // division by constexpr V → inline shift, once

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<INPUT0_DTYPE, V> a = aie::load_v<V>(src0 + i);
        aie::vector<INPUT1_DTYPE, V> b = aie::load_v<V>(src1 + i);
        aie::store_v(out + i, aie::add(a, b));
    }

    for (int32_t i = vend; i < N; ++i) {
        out[i] = src0[i] + src1[i];
    }

    event1();
}

#endif // GGML_OP_ADD_BIAS
```

Note: this plan targets the MNIST f32 case where `INPUT0_DTYPE == INPUT1_DTYPE == OUTPUT_DTYPE == float`, so `aie::add(a, b)` and the store type match. (The Python side only dispatches ADD f32 bias to this kernel.)

- [ ] **Step 3: Standalone Peano compile — confirm vectorized, no `__divsi3`**

This is the fast static gate from the optimizer skill (no board needed). Compile just the kernel for aie2 and inspect the object. Run:

```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
MLIR_AIE=build/.venv/lib/python3.13/site-packages/mlir_aie
clang++ -O2 -c -std=c++20 --target=aie2-none-unknown-elf \
  -D__AIECC__ -D__AIE_API_AIE_ADF_HPP__ -DNDEBUG \
  -DGGML_OP_ADD_BIAS=1 -DINPUT0_DTYPE=float -DINPUT1_DTYPE=float -DOUTPUT_DTYPE=float \
  -I src/ggml-hsa/kernels/iron_kernels \
  -I "$MLIR_AIE/include" \
  -I "$MLIR_AIE/aie_runtime_lib/AIE2" \
  src/ggml-hsa/kernels/iron_kernels/binary_ops.cc -o /tmp/add_bias.o 2>&1 | tail -20
echo "=== vector ops present (want > 0) ==="
"$MLIR_AIE"/bin/llvm-objdump -d /tmp/add_bias.o | grep -coE '\bv(lda|ldb|st|add)\b' || true
echo "=== __divsi3 present (want NOTHING) ==="
"$MLIR_AIE"/bin/llvm-nm /tmp/add_bias.o | grep __div || echo "no __divsi3 — good"
```

Expected: object compiles; vector-op count > 0; `no __divsi3 — good`.

If the exact `-I` paths differ in this install, find them with `find build/.venv -name aie.hpp -path '*aie_api*'` and `find build/.venv -name llvm-objdump`. If the standalone compile is impractical (missing headers), skip to Task 2's on-device test as the correctness gate and note it — the objdump check is a bonus static confirmation, not a blocker.

- [ ] **Step 4: Commit**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/kernels/iron_kernels/binary_ops.cc
git commit -m "$(cat <<'EOF'
Add vectorized ggml_op_add_bias AIE kernel for f32 bias add

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add the row-tiled IRON design + dispatch predicate

**Files:**
- Modify: `src/ggml-hsa/kernels/iron_kernels/binary_ops.py` (add `_create_bias_external_function`, `_binary_op_bias`, and a predicate branch in `binary_op`)

**Interfaces:**
- Consumes: `ggml_op_add_bias` from Task 1 (via `ExternalFunction` `source_file=binary_ops.cc`, `-DGGML_OP_ADD_BIAS=1`).
- Produces: routing so that an ADD whose `src1` is a single bias row compiles the new kernel instead of `transform_binary_broadcast_n`.

- [ ] **Step 1: Add the bias `ExternalFunction` factory**

In `binary_ops.py`, after `_create_broadcast_external_function` (ends ~line 262), add:

```python
def _create_bias_external_function(
    arch: str,
    input_tensors: list,
    output_tensor,
) -> CoreFunctionSpec:
    """Create the CoreFunctionSpec for the row-tiled ADD bias kernel.

    src1 is a single row (ne0 elements) reused across all dst rows. The tile
    is exactly one dst row, so tile_size == ne0.

    Parameters:
        arch: Target architecture.
        input_tensors: Two input tensors [src0, src1].
        output_tensor: Output tensor.

    """
    num_elements = arch_aligned_num_elements(arch=arch, tensor=output_tensor)
    ne0 = output_tensor.shape[0]
    tile_size = ne0

    if num_elements % tile_size != 0:
        msg = f"Output elements ({num_elements}) not divisible by row length ({tile_size})."
        raise ValueError(msg)

    current_dir = Path(__file__).resolve().parent
    func = ExternalFunction(
        name="ggml_op_add_bias",
        object_file_name="ggml_op_add_bias_core_function.o",
        source_file=str(current_dir / "binary_ops.cc"),
        arg_types=[
            np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]],
            np.ndarray[(tile_size,), np.dtype[input_tensors[1].dtype]],
            np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]],
            np.int32,
        ],
        compile_flags=[
            "-DGGML_OP_ADD_BIAS=1",
            f"-DINPUT0_DTYPE={dtype_to_str(input_tensors[0].dtype)}",
            f"-DINPUT1_DTYPE={dtype_to_str(input_tensors[1].dtype)}",
            f"-DOUTPUT_DTYPE={dtype_to_str(output_tensor.dtype)}",
        ],
    )
    return CoreFunctionSpec(external_function=func, num_elements=num_elements)
```

- [ ] **Step 2: Add the row-tiled design**

After `_create_bias_external_function`, add the design. It mirrors `_binary_op` but uses a third `depth=1` fifo for the reused bias row (like `_binary_op_broadcast`'s `of_src1`), and the tile is one row:

```python
def _binary_op_bias(
    arch: str,
    input_tensors: list,
    function_spec: CoreFunctionSpec,
    output_tensor,
):
    """Row-tiled ADD bias: out[row] = src0[row] + src1 (one bias row reused).

    Parameters:
        arch: Target architecture.
        input_tensors: Input tensors [src0, src1]; src1 is one ne0-element row.
        function_spec: Core function specification (tile_size == ne0).
        output_tensor: Output tensor.

    """
    num_elements = function_spec.num_elements
    tile_size = function_spec.tile_size  # == ne0
    num_tiles = num_elements // tile_size
    num_elements_src1 = arch_aligned_num_elements(arch=arch, tensor=input_tensors[1])

    src0_tile_ty = np.ndarray[(tile_size,), np.dtype[input_tensors[0].dtype]]
    src1_row_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_tile_ty = np.ndarray[(tile_size,), np.dtype[output_tensor.dtype]]

    of_src0 = ObjectFifo(src0_tile_ty, name="src0")
    of_src1 = ObjectFifo(src1_row_ty, depth=1, name="src1")  # loaded once, reused
    of_out = ObjectFifo(out_tile_ty, name="out")

    function = function_spec.external_function

    def ext_core_fn(of_src0, of_src1, of_out, function):
        src1_buf = of_src1.acquire(1)  # one bias row, reused across all tiles
        for _ in range_(num_tiles):
            src0_tile = of_src0.acquire(1)
            out_tile = of_out.acquire(1)
            function(src0_tile, src1_buf, out_tile, tile_size)
            of_src0.release(1)
            of_out.release(1)
        of_src1.release(1)

    worker = Worker(
        ext_core_fn,
        fn_args=[of_src0.cons(), of_src1.cons(), of_out.prod(), function],
    )

    # Buffers in src order then dst (kernarg layout contract).
    src0_ty = np.ndarray[(num_elements,), np.dtype[input_tensors[0].dtype]]
    src1_ty = np.ndarray[(num_elements_src1,), np.dtype[input_tensors[1].dtype]]
    out_ty = np.ndarray[(num_elements,), np.dtype[output_tensor.dtype]]

    rt = Runtime()
    with rt.sequence(src0_ty, src1_ty, out_ty) as (a, b, c):
        rt.start(worker)
        rt.fill(of_src0.prod(), a)
        rt.fill(of_src1.prod(), b)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(arch_to_device(arch), rt).resolve_program()
```

- [ ] **Step 3: Wire the predicate into `binary_op`**

In `binary_op` (bottom of the file), the current logic computes `needs_broadcast = src1_shape != dst_shape` then branches broadcast vs. element-wise. Insert the bias fast-path **after** `needs_broadcast` is computed and **before** the `if needs_broadcast:` block. Locate:

```python
    # Check if broadcasting is needed
    needs_broadcast = src1_shape != dst_shape

    if needs_broadcast:
```

Change to:

```python
    # Check if broadcasting is needed
    needs_broadcast = src1_shape != dst_shape

    # ADD-only fast path: src1 is a single bias row broadcast over dst rows.
    src1_is_bias_row = (
        src1_shape[0] == dst_shape[0]
        and src1_shape[1] == 1
        and src1_shape[2] == 1
        and src1_shape[3] == 1
    )
    if op_name == "GGML_OP_ADD" and needs_broadcast and src1_is_bias_row:
        function_spec = _create_bias_external_function(
            arch=arch,
            input_tensors=input_tensors,
            output_tensor=output_tensor,
        )
        return _binary_op_bias(
            arch=arch,
            input_tensors=input_tensors,
            function_spec=function_spec,
            output_tensor=output_tensor,
        )

    if needs_broadcast:
```

- [ ] **Step 4: Force recompile of stale ADD artifacts**

The kernel cache keys on the compiled design; move aside any stale ADD broadcast artifacts so the new kernel compiles fresh:

```bash
cd /home/ypapadop/workspace-raiders/ggml
mkdir -p /tmp/ggml-aie-cache-stash
find ~/.cache/ggml/aie2 -maxdepth 1 -name 'ggml_op_add*' -exec mv {} /tmp/ggml-aie-cache-stash/ \; 2>/dev/null || true
find ~/.cache/ggml/aie2 -maxdepth 1 -name 'add*' -exec mv {} /tmp/ggml-aie-cache-stash/ \; 2>/dev/null || true
echo "stashed: $(ls /tmp/ggml-aie-cache-stash | wc -l) entries"
```

- [ ] **Step 5: Build the backend (re-copies kernels to build tree)**

```bash
cd /home/ypapadop/workspace-raiders/ggml
cmake --build build --target ggml-hsa -j 2>&1 | tail -15
```

Expected: builds without error; the copy target restages `binary_ops.cc`/`.py` into `build/src/ggml-hsa/kernels/iron_kernels/`.

- [ ] **Step 6: On-device correctness gate — ADD stays 3/3 and takes the new path**

```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
GGML_HSA_ENABLE_LOG=1 "$PWD/build/bin/test-backend-ops-mnist" test -o ADD 2>&1 | tee /tmp/add_test.log | grep -iE "ADD\(|passed|OK|FAIL|not supported"
echo "=== new kernel actually compiled (want a hit) ==="
grep -c "ggml_op_add_bias\|add_bias" /tmp/add_test.log || true
```

Expected: `3/3 tests passed`, `Backend HSA0: OK`; and the `add_bias` grep count > 0 (proves the two MNIST bias adds took the new path, not the scalar fallback). If count is 0, the predicate didn't fire — recheck Step 3 shape indexing against the actual `output_tensor.shape` / `input_tensors[1].shape` ordering.

- [ ] **Step 7: Commit**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/kernels/iron_kernels/binary_ops.py
git commit -m "$(cat <<'EOF'
Route single-row ADD bias to vectorized row-tiled IRON design

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Verify the win — isolated ADD profile + end-to-end

**Files:**
- Temporary, reverted: whatever file holds the `GGML_HSA_PROFILE_DISPATCH` per-dispatch probe (per tracking-doc §4 recipe). Not committed.
- Modify (append only): `docs/ggml-hsa-optimization-tracking.md` (record the before/after)

**Interfaces:**
- Consumes: the built backend from Task 2.
- Produces: dated measurements appended to the tracking doc.

- [ ] **Step 1: Baseline end-to-end (already known, re-confirm)**

```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
BIN=$PWD/build/bin/mnist-eval
IMG=$PWD/examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte
LBL=$PWD/examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte
MODEL=$PWD/examples/mnist/mnist-fc-f32.gguf
for i in 1 2 3; do "$BIN" "$MODEL" "$IMG" "$LBL" HSA0 2>&1 | grep -E "took|test_acc"; done
```

Expected: ~333 µs/image, 98.00% accuracy (matches tracking §1). Discard the first cold run if it's an outlier (known cold-cache JIT hitch).

- [ ] **Step 2: Re-add the §4 per-dispatch profiler**

Follow the recipe in `docs/ggml-hsa-optimization-tracking.md` §4 / §5 ("Re-adding the host-phase timers" is the sibling recipe): add the env-gated `GGML_HSA_PROFILE_DISPATCH` wrapper around each kernel dispatch in `graph_compute` that drains the queue before stopping the timer, accumulating per-kernel µs and dispatch counts, printing the table at exit. This isolates ADD µs/dispatch. Rebuild:

```bash
cmake --build build --target ggml-hsa -j 2>&1 | tail -5
```

- [ ] **Step 3: Measure ADD µs/dispatch with the new kernel**

```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
BIN=$PWD/build/bin/mnist-eval
IMG=$PWD/examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte
LBL=$PWD/examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte
MODEL=$PWD/examples/mnist/mnist-fc-f32.gguf
GGML_HSA_PROFILE_DISPATCH=1 "$BIN" "$MODEL" "$IMG" "$LBL" HSA0 2>&1 | grep -iE "ADD|MUL_MAT|TOTAL"
```

Expected: ADD µs/dispatch **substantially below** the §4 baseline of ~59,430 µs (the scalar `__divsi3` path). Record the exact number. (This is the direct evidence the optimization landed; absolute totals are inflated because the profiler serializes — compare ADD relative to the same-run GEMM and to the §4 ADD figure.)

- [ ] **Step 4: Revert the profiler**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git checkout -- <the file(s) you modified for the probe>
git status   # confirm clean except the tracking doc (Step 6)
cmake --build build --target ggml-hsa -j 2>&1 | tail -3
```

- [ ] **Step 5: Confirm end-to-end + accuracy after revert**

```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
BIN=$PWD/build/bin/mnist-eval
IMG=$PWD/examples/mnist/data/MNIST/raw/t10k-images-idx3-ubyte
LBL=$PWD/examples/mnist/data/MNIST/raw/t10k-labels-idx1-ubyte
MODEL=$PWD/examples/mnist/mnist-fc-f32.gguf
for i in 1 2 3; do "$BIN" "$MODEL" "$IMG" "$LBL" HSA0 2>&1 | grep -E "took|test_acc"; done
```

Expected: accuracy stays 98.00%; µs/image ≤ the §1 baseline (ADD is ~55% of *serialized* dispatch, but batched end-to-end may move less — record whatever it is honestly, cross-checking against §4's caveat that serialized totals overstate the batched effect).

- [ ] **Step 6: Append results to the tracking doc and commit**

Add a new dated section to `docs/ggml-hsa-optimization-tracking.md` (append, don't overwrite) recording: the ADD µs/dispatch before (§4: 59,430) vs. after, the end-to-end before/after, accuracy, and a note that the vectorization applies to fc1 (99.2% vectorized) while fc2 stays scalar but `__divsi3`-free. Then:

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add docs/ggml-hsa-optimization-tracking.md
git commit -m "$(cat <<'EOF'
Record vectorized ADD bias benchmark results

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Predicate / dispatch (spec "Architecture & dispatch") → Task 2 Step 3. ✓
- Row-tiling scheme (spec "The row-tiling scheme") → Task 2 Steps 1–2. ✓
- Kernel loop (spec "Kernel loop") → Task 1 Step 2. ✓
- No `__divsi3` / vectorized (spec Global Constraints) → Task 1 Step 3 static check. ✓
- Verification: correctness gate → Task 2 Step 6; isolated profile → Task 3 Steps 2–3; end-to-end + accuracy → Task 3 Steps 1/5. ✓ (matches spec "Verification (full profile)")
- Cache hygiene (spec) → Task 2 Step 4. ✓
- fc2 fallback-but-still-improved (spec) → asserted in Task 2 Step 6 (both take new path) and recorded in Task 3 Step 6. ✓

**Placeholder scan:** All code steps show full code. The only intentionally non-literal spots are the §4 profiler file path (Task 3 Steps 2/4) — the probe is deliberately uncommitted and its exact location is per the tracking-doc recipe, not fixed by this change — and the tracking-doc section text (Task 3 Step 6), which depends on measured numbers. Both are unavoidable; instructions point to the concrete recipe.

**Type consistency:** `_create_bias_external_function` returns `CoreFunctionSpec` (existing dataclass, has `.tile_size` via `external_function.tile_size(0)`); `_binary_op_bias` consumes `function_spec.num_elements` and `.tile_size` — matches. Kernel symbol `ggml_op_add_bias` and macro `GGML_OP_ADD_BIAS` are consistent across Task 1 (definition) and Task 2 (`name=` and `-DGGML_OP_ADD_BIAS=1`). Kernel signature `(src0, src1, out, N)` matches the `arg_types` list order (in0, in1, out, int32) and the call `function(src0_tile, src1_buf, out_tile, tile_size)`.
