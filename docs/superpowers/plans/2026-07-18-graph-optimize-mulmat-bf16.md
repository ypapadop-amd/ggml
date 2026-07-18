# graph_optimize-based MUL_MAT bf16 cast fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the MUL_MAT f32→bf16 cast fusion out of graph_compute (whole-graph scan + `fusion_t` bookkeeping) into a `graph_optimize` backend hook that retypes the MUL_MAT to bf16 and blanks the orphaned cast, deleting the custom fusion machinery.

**Architecture:** `graph_optimize` runs per-split before allocation and before `tensor_extra` construction. A new hook retypes each qualifying padded-GEMM MUL_MAT to bf16, rewires the consumers of its f32→bf16 cast(s) onto the MUL_MAT, and sets each orphaned cast's op to `GGML_OP_NONE`. The framework's existing `ggml_op_is_empty` skip drops the blanked casts; the existing de-pad already converts f32→bf16 when the destination is bf16, so no kernel or dispatch logic changes.

**Tech Stack:** C++ (ggml-hsa backend), ggml graph API, AMD XDNA/AIE2 NPU, CMake + Ninja, embedded-Python JIT (Peano) for kernels.

## Global Constraints

- Build: `-DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON -DCMAKE_BUILD_TYPE=Release`, `-DCMAKE_PREFIX_PATH="/home/ypapadop/workspace-raiders/opt/rocm/lib/cmake/hsa-runtime64;/opt/rocm/lib/cmake"`.
- Run tests with the venv active: `source build/.venv/bin/activate` (JIT runs in embedded Python).
- `rm` is permission-blocked in this environment — use `mv` to move files/caches aside.
- No kernel source changes: `depad.cc`/`depad.py` already handle f32→bf16. Do NOT touch `~/.cache/ggml/aie2/` unless a kernel changes (none do here).
- Bit-exactness is the gate: bf16 loss `0.066342` / acc `98.00%`; f32 loss `0.066372`. Numbers must not move.
- Behavior-preserving refactor: perf-neutral (~62 µs/image bf16, ~48 µs/image f32). A regression is a bug.
- Files touched: only `src/ggml-hsa/ggml-hsa.cpp` and `src/ggml-hsa/common.hpp`.
- Device arch is `aie2` here (tile=32, n_aie_cols=4, n_aie_rows=4).

---

## File Structure

- `src/ggml-hsa/ggml-hsa.cpp` — add the shared predicate `ggml_hsa_mul_mat_is_padded_gemm`; add the hook `ggml_backend_hsa_graph_optimize`; relax the `prepare_mul_mat` dst-type gate; delete `ggml_hsa_fuse_mul_mat_narrow` + its call site + the `skip_dispatch` continue + the `post_dst`/`narrow_dst` branch; wire the hook into the iface.
- `src/ggml-hsa/common.hpp` — delete the `fusion_t` struct and the `fusion` member from `ggml_backend_hsa_tensor_extra`.

**Ordering rationale:** Tasks 1–2 add the new code paths (predicate, prepare relax) without removing anything — the graph still works via the old fusion because the bf16 example still emits a separate cast that the old scan handles. Task 3 adds the hook (now the graph is rewritten at optimize-time; the old fusion becomes a no-op because the cast is already blanked before graph_compute sees it). Task 4 deletes the now-dead fusion machinery. This keeps the build green and the tests passing after every task.

---

### Task 1: Extract the shared `ggml_hsa_mul_mat_is_padded_gemm` predicate

Factor the padded-GEMM eligibility check (minus the dst-type rule) out of `ggml_hsa_prepare_mul_mat_f32` so both `prepare_mul_mat` and the future hook share one source of truth. The predicate operates on a raw `ggml_tensor` (the hook has no `node_t`).

**Files:**
- Modify: `src/ggml-hsa/ggml-hsa.cpp` (add predicate above `ggml_hsa_prepare_mul_mat_f32` at line 658; rewrite the entry gate at lines 663-682)

**Interfaces:**
- Consumes: `ggml_hsa_has_trivial_layout(const ggml_tensor &)` (line 588).
- Produces: `static bool ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor & mm)` — true iff `mm` is a MUL_MAT eligible for the padded bf16 GEMM path, ignoring `mm.type` (the destination dtype, which callers check themselves).

- [ ] **Step 1: Add the predicate function**

Insert immediately before `ggml_hsa_prepare_mul_mat_f32` (before current line 658):

```cpp
/**
 * @brief Eligibility for the padded bf16 GEMM path, over a raw tensor.
 *
 * Shared by @c ggml_hsa_prepare_mul_mat_f32 (which additionally constrains the destination dtype)
 * and @c ggml_backend_hsa_graph_optimize (which runs before tensor_extra exists, so it only has the
 * raw graph tensor). Deliberately does NOT check @c mm.type: the two callers disagree on the
 * destination dtype, so each applies its own rule.
 *
 * @param[in] mm candidate MUL_MAT node.
 * @return @c true if @p mm is a 2-source MUL_MAT with f32/bf16 operands, trivial layout on both
 *         operands and the destination, and no batch/broadcast.
 */
static bool ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor & mm) {
    if (mm.op != GGML_OP_MUL_MAT || mm.src[0] == nullptr || mm.src[1] == nullptr ||
        mm.src[2] != nullptr) {
        return false;
    }
    const ggml_tensor & a = *mm.src[0]; // [K, M]
    const ggml_tensor & b = *mm.src[1]; // [K, N]

    const bool a_ok = a.type == GGML_TYPE_F32 || a.type == GGML_TYPE_BF16;
    const bool b_ok = b.type == GGML_TYPE_F32 || b.type == GGML_TYPE_BF16;
    if (!a_ok || !b_ok) {
        return false;
    }
    if (!ggml_hsa_has_trivial_layout(a) || !ggml_hsa_has_trivial_layout(b) ||
        !ggml_hsa_has_trivial_layout(mm)) {
        return false;
    }
    // no batching / broadcasting
    if (a.ne[2] != 1 || a.ne[3] != 1 || b.ne[2] != 1 || b.ne[3] != 1) {
        return false;
    }
    return true;
}
```

- [ ] **Step 2: Rewrite the prepare_mul_mat entry gate to call the predicate**

Replace the current entry gate in `ggml_hsa_prepare_mul_mat_f32` (lines 663-682, from `ggml_tensor & dst = node.tensor;` through the closing `}` of the trivial-layout `if`) with:

```cpp
    ggml_tensor & dst = node.tensor;

    // The GEMM microkernel runs in bf16, so both operands must be f32 (converted to bf16 below) or
    // already bf16 (converted in the graph, e.g. the bf16 MNIST variant). Shared eligibility is
    // checked over the internal source tensors via the raw-tensor predicate; here we additionally
    // require an f32 destination (ggml's native MUL_MAT output) or a bf16 destination (retyped by
    // graph_optimize so the de-pad narrows f32->bf16 in one pass).
    if (nsrcs != 2) {
        return false;
    }
    ggml_tensor & a = src_nodes[0].tensor; // [K, M]
    ggml_tensor & b = src_nodes[1].tensor; // [K, N]

    // Build a raw view of the node/sources for the shared predicate: node_t.tensor already mirrors
    // the parent's op/type/shape at this point in construction.
    ggml_tensor probe = dst;
    probe.src[0] = &a;
    probe.src[1] = &b;
    probe.src[2] = nullptr;
    if (!ggml_hsa_mul_mat_is_padded_gemm(probe)) {
        return false;
    }
    if (dst.type != GGML_TYPE_F32 && dst.type != GGML_TYPE_BF16) {
        return false;
    }
```

Note: this REPLACES the old `dst.op != GGML_OP_MUL_MAT`, the `a_ok/b_ok/dst.type` block, and the trivial-layout block. The old `dst.type != GGML_TYPE_F32` restriction is intentionally relaxed to `!= F32 && != BF16` here (this is the Task 2 relaxation, folded in so the predicate refactor and the gate relax land together and stay consistent). The rest of `prepare_mul_mat` (tiling, source/dst rewrite, `node.depad = true`) is unchanged.

- [ ] **Step 3: Configure and build**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
cmake -S . -B build -G Ninja -DGGML_HSA=ON -DGGML_HSA_JIT_COMPILE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/home/ypapadop/workspace-raiders/opt/rocm/lib/cmake/hsa-runtime64;/opt/rocm/lib/cmake" >/dev/null
cmake --build build -j 2>&1 | tail -5
```
Expected: builds to completion, links `bin/mnist-eval`, `bin/mnist-eval-bf16`, `bin/test-backend-ops-mnist`, `bin/test-depad-hsa` with no errors.

- [ ] **Step 4: Verify no behavior change (old fusion still active)**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
D=examples/mnist/data/MNIST/raw; M=examples/mnist/mnist-fc-f32.gguf
"$PWD/build/bin/mnist-eval-bf16" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
"$PWD/build/bin/mnist-eval" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
```
Expected: bf16 `test_loss=0.066342`, `test_acc=98.00`, ~62 µs/image; f32 `test_loss=0.066372`, `test_acc=98.00`, ~48 µs/image. (Task 1 changes only how eligibility is expressed; the padded path is byte-for-byte the same. The bf16 dst is not yet produced by anything, so the relaxed gate is dormant.)

- [ ] **Step 5: Commit**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/ggml-hsa.cpp
git commit -m "$(cat <<'EOF'
Extract shared padded-GEMM predicate and accept bf16 MUL_MAT dst

Factor the MUL_MAT eligibility check into ggml_hsa_mul_mat_is_padded_gemm
(raw-tensor form, dst-type agnostic) shared by prepare_mul_mat and the
upcoming graph_optimize hook, and relax prepare's dst gate to accept a
bf16 destination (de-pad already narrows f32->bf16).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add the `ggml_backend_hsa_graph_optimize` hook and wire it in

Add the hook that retypes each qualifying MUL_MAT to bf16, rewires its cast consumers' readers onto the MUL_MAT, and blanks each orphaned cast to `GGML_OP_NONE`. Wire it into the iface. After this task the graph is rewritten at optimize-time; the old graph_compute fusion still runs but finds nothing to do (the cast is already `GGML_OP_NONE` before graph_compute, so it is skipped by `ggml_op_is_empty` and the old scan's `is_convert_copy` check fails on it).

**Files:**
- Modify: `src/ggml-hsa/ggml-hsa.cpp` (add hook before the iface struct ~line 1900; change iface slot at line 1920)

**Interfaces:**
- Consumes: `ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor &)` (Task 1); `ggml_hsa_is_convert_copy(const ggml_tensor &)` (line 613); `ggml_hsa_set_contiguous_strides(ggml_tensor &)` (line 597); `ggml_graph_n_nodes` / `ggml_graph_node` (ggml.h:2740-2742).
- Produces: `static void ggml_backend_hsa_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph)` wired into the backend iface.

- [ ] **Step 1: Add the hook function**

Insert immediately before the `ggml_backend_i` iface struct definition (before the `/* .get_name ... */` block around line 1905). The hook needs neither `ctx` nor `dev_info` — it mutates only tensor fields:

```cpp
/**
 * @brief Retypes qualifying padded-GEMM MUL_MAT nodes to bf16 and blanks the following cast(s).
 *
 * ggml_mul_mat always produces f32, so a bf16 graph wraps each GEMM in an f32->bf16 cast. This hook
 * runs per split before allocation and before tensor_extra construction, the sanctioned point to
 * rewrite the graph. For a MUL_MAT eligible for the padded bf16 GEMM path whose every consumer is a
 * non-output f32->bf16 convert-cast, it retypes the MUL_MAT to bf16 (the de-pad then narrows
 * f32->bf16 in one pass), rewires each cast's downstream readers onto the MUL_MAT, and sets each
 * orphaned cast's op to GGML_OP_NONE. The framework's ggml_op_is_empty skip then drops the blanked
 * casts and gallocr gives them no buffer (0 children). All-or-nothing: if any consumer reads the
 * MUL_MAT as f32, or any cast is a graph output, the node is left untouched (normal f32 de-pad).
 *
 * Node removal is not possible here (the scheduler rebuilds the split from the original node range),
 * so the cast node persists but is neutralized to a no-op.
 */
static void ggml_backend_hsa_graph_optimize(ggml_backend_t /*backend*/, ggml_cgraph * cgraph) {
    const std::int32_t node_count = ggml_graph_n_nodes(cgraph);

    for (std::int32_t i = 0; i < node_count; ++i) {
        ggml_tensor * mm = ggml_graph_node(cgraph, i);
        if (!ggml_hsa_mul_mat_is_padded_gemm(*mm)) {
            continue;
        }
        if ((mm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            continue;
        }

        // Collect all consumers of mm; require at least one and every one a non-output
        // f32->bf16 convert-cast. Any f32 reader (matmul, add, output, or a cast to another dtype)
        // disqualifies: retyping mm would feed bf16 where f32 is expected.
        bool qualifies = true;
        bool has_consumer = false;
        for (std::int32_t j = 0; j < node_count && qualifies; ++j) {
            ggml_tensor * c = ggml_graph_node(cgraph, j);
            for (auto s = 0; s < GGML_MAX_SRC; ++s) {
                if (c->src[s] != mm) {
                    continue;
                }
                has_consumer = true;
                const bool is_bf16_cast = ggml_hsa_is_convert_copy(*c) &&
                                          c->src[0] != nullptr &&
                                          c->src[0]->type == GGML_TYPE_F32 &&
                                          c->type == GGML_TYPE_BF16 &&
                                          (c->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
                if (!is_bf16_cast) {
                    qualifies = false;
                }
                break; // a node lists mm in at most one meaningful src slot for this check
            }
        }
        if (!qualifies || !has_consumer) {
            continue;
        }

        // Retype the MUL_MAT result to bf16; the de-pad post-amble now narrows f32->bf16 directly.
        mm->type = GGML_TYPE_BF16;
        ggml_hsa_set_contiguous_strides(*mm);

        // For each cast consumer: rewire its downstream readers onto mm, then blank it to a no-op.
        for (std::int32_t j = 0; j < node_count; ++j) {
            ggml_tensor * cast = ggml_graph_node(cgraph, j);
            bool cast_of_mm = false;
            for (auto s = 0; s < GGML_MAX_SRC; ++s) {
                if (cast->src[s] == mm) {
                    cast_of_mm = true;
                    break;
                }
            }
            if (!cast_of_mm) {
                continue;
            }
            // rewire everything that reads the cast to read mm instead
            for (std::int32_t k = 0; k < node_count; ++k) {
                ggml_tensor * r = ggml_graph_node(cgraph, k);
                for (auto s = 0; s < GGML_MAX_SRC; ++s) {
                    if (r->src[s] == cast) {
                        r->src[s] = mm;
                    }
                }
            }
            cast->op = GGML_OP_NONE;
        }
    }
}
```

- [ ] **Step 2: Wire the hook into the iface**

At line 1920, change:
```cpp
    /* .graph_optimize      = */ nullptr,
```
to:
```cpp
    /* .graph_optimize      = */ ggml_backend_hsa_graph_optimize,
```

- [ ] **Step 3: Build**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
cmake --build build -j 2>&1 | tail -5
```
Expected: builds and links cleanly.

- [ ] **Step 4: Verify bit-exactness and perf unchanged**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
D=examples/mnist/data/MNIST/raw; M=examples/mnist/mnist-fc-f32.gguf
"$PWD/build/bin/mnist-eval-bf16" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
"$PWD/build/bin/mnist-eval" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
```
Expected: bf16 `test_loss=0.066342`, `test_acc=98.00`, ~62 µs/image; f32 `test_loss=0.066372`, `test_acc=98.00`, ~48 µs/image. The bf16 result now flows through the hook (retype + blank) rather than the old scan, and must be identical. If the bf16 loss changes, the rewrite is wrong — stop and debug (most likely a mis-rewired src or a wrongly-blanked cast).

- [ ] **Step 5: Op-gate selectivity check**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
"$PWD/build/bin/test-backend-ops-mnist" test -o MUL_MAT 2>&1 | tail -5
```
Expected: MUL_MAT cases PASS (3/3). Standalone MUL_MAT has no following cast, so the hook does not fire and the f32 de-pad path is exercised — confirms the guard is selective.

- [ ] **Step 6: Commit**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/ggml-hsa.cpp
git commit -m "$(cat <<'EOF'
Add graph_optimize hook to retype MUL_MAT to bf16 and blank the cast

Rewrite qualifying padded-GEMM MUL_MAT nodes to output bf16, rewire the
f32->bf16 cast consumers' readers onto the MUL_MAT, and neutralize each
orphaned cast to GGML_OP_NONE so the framework's empty-op skip drops it.
Bit-identical to the prior graph_compute fusion.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Delete the dead graph_compute fusion machinery

With the hook doing the rewrite, `ggml_hsa_fuse_mul_mat_narrow`, the `fusion_t` state, the `skip_dispatch` continue, and the `post_dst`/`narrow_dst` branch are all dead. Remove them.

**Files:**
- Modify: `src/ggml-hsa/ggml-hsa.cpp` (delete function ~1526-1608; delete call site + comment ~1647-1654; delete `skip_dispatch` continue ~1664-1671; simplify `post_dst` branch ~1760-1762)
- Modify: `src/ggml-hsa/common.hpp` (delete `fusion_t` struct + `fusion` member ~391-407)

**Interfaces:**
- Consumes: nothing new.
- Produces: no new symbols; removes `ggml_hsa_fuse_mul_mat_narrow`, `ggml_backend_hsa_tensor_extra::fusion_t`, and the `fusion` member.

- [ ] **Step 1: Delete the fusion function**

Delete the entire `ggml_hsa_fuse_mul_mat_narrow` function, including its doc comment (from the `/**` block beginning "Fuses each padded MUL_MAT's de-pad..." through the function's closing `}` — current lines ~1525-1608). Confirm the deletion boundary by locating the function before and after it (`ggml_backend_hsa_synchronize` above, `ggml_hsa_copy_padded_or_plain` below) so only this function is removed.

- [ ] **Step 2: Delete the call site in graph_compute**

Remove the fusion call and its comment block (current lines ~1647-1654):
```cpp
    // Fuse the per-layer f32->bf16 cast into the MUL_MAT de-pad post-amble. A padded-GEMM MUL_MAT
    // produces an f32 result that HSA_DEPAD strips into the f32 parent, which a following
    // convert-CPY then re-reads to write a bf16 copy for the next op. When the MUL_MAT's sole
    // consumer is exactly that cast, redirect the de-pad to narrow-and-write bf16 straight into the
    // cast's output buffer and skip the cast dispatch, removing a kernel launch and a full [M, N]
    // f32 memory round trip. The ggml graph is left untouched (no tensor type mutated); this is a
    // device-side rewrite only.
    ggml_hsa_fuse_mul_mat_narrow(ctx, cgraph, node_count);
```
Leave the blank line so the source-pointer fixup loop is directly followed by the main dispatch loop.

- [ ] **Step 3: Delete the skip_dispatch continue**

Remove this block from the main dispatch loop (current lines ~1664-1671):
```cpp
        // This node was fused into its producer's de-pad (only possible for a convert-CPY/DUP
        // consumer of a MUL_MAT), which already wrote the narrowed bf16 result here. Running it
        // would read the never-written f32 parent.
        if (tensor_extra.fusion.skip_dispatch) {
            continue;
        }
```
The `ggml_op_is_empty(node->op)` check earlier in the loop already skips the blanked (GGML_OP_NONE) cast, so no replacement is needed.

- [ ] **Step 4: Simplify the postprocess dispatch branch**

Replace the `post_dst` block (current lines ~1755-1763) — from the `if (use_device_transforms) {` comment through the `dispatch(...)` call — so it writes back into the node itself:

Old:
```cpp
        if (use_device_transforms) {
            // on-device result post-processing: transform the internal output back into the parent
            // tensor on-queue (e.g. de-pad), no drain. Reads the internal node, writes the parent —
            // or, when the de-pad was fused with a following f32->bf16 cast, narrows straight into
            // that consumer's buffer (skipping the separate cast dispatch).
            ggml_tensor * postprocess_src = &internal_node;
            ggml_tensor & post_dst =
                tensor_extra.fusion.narrow_dst ? *tensor_extra.fusion.narrow_dst : *node;
            status = tensor_extra.postprocess_kernel->dispatch(ctx, &postprocess_src, 1, post_dst);
```
New:
```cpp
        if (use_device_transforms) {
            // on-device result post-processing: transform the internal output back into the parent
            // tensor on-queue (e.g. de-pad), no drain. When the MUL_MAT was retyped to bf16 by
            // graph_optimize, the parent is bf16 and the de-pad narrows f32->bf16 in one pass.
            ggml_tensor * postprocess_src = &internal_node;
            status = tensor_extra.postprocess_kernel->dispatch(ctx, &postprocess_src, 1, *node);
```

- [ ] **Step 5: Delete the fusion_t struct from common.hpp**

Remove the `fusion_t` struct, its doc comment, and the `fusion` member (current lines ~391-407):
```cpp
    /// @brief State for the whole-graph MUL_MAT de-pad + f32->bf16 cast fusion (see
    /// @c ggml_hsa_fuse_mul_mat_narrow). Set on the MUL_MAT (narrow_dst, analyzed) and, when fusion
    /// applies, on the fused-away consumer CPY/DUP (skip_dispatch).
    struct fusion_t {
        /// @brief When non-null (this is a MUL_MAT whose de-pad post-amble was fused with the
        /// following f32->bf16 cast), the post-processing kernel narrows and writes the result into
        /// this consumer tensor instead of the f32 parent.
        ggml_tensor * narrow_dst{nullptr};
        /// @brief When true (this is a convert-CPY whose cast was fused into its producer's
        /// de-pad), graph compute skips this node entirely: the producer already wrote the narrowed
        /// bf16 result into this tensor, so re-running the copy would read the never-written f32
        /// parent and clobber it.
        bool skip_dispatch{false};
        /// @brief Latches the one-time fusion pre-pass so the whole-graph consumer scan and
        /// fused-kernel lookup run once, not on every forward pass.
        bool analyzed{false};
    } fusion;
```

- [ ] **Step 6: Build and confirm no dangling references**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
cmake --build build -j 2>&1 | tail -8
grep -rn "fusion\|narrow_dst\|skip_dispatch\|fuse_mul_mat_narrow" src/ggml-hsa/ggml-hsa.cpp src/ggml-hsa/common.hpp || echo "NO REMAINING REFERENCES"
```
Expected: clean build; the grep prints `NO REMAINING REFERENCES` (all fusion symbols gone).

- [ ] **Step 7: Full verification suite**

Run:
```bash
cd /home/ypapadop/workspace-raiders/ggml
source build/.venv/bin/activate
D=examples/mnist/data/MNIST/raw; M=examples/mnist/mnist-fc-f32.gguf
echo "== bf16 NPU =="; "$PWD/build/bin/mnist-eval-bf16" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
echo "== f32 NPU  =="; "$PWD/build/bin/mnist-eval" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" HSA0 2>&1 | grep -iE "test_loss|test_acc|us/image"
echo "== CPU      =="; "$PWD/build/bin/mnist-eval" "$M" "$D/t10k-images-idx3-ubyte" "$D/t10k-labels-idx1-ubyte" CPU 2>&1 | grep -iE "test_loss|test_acc|us/image"
echo "== MUL_MAT  =="; "$PWD/build/bin/test-backend-ops-mnist" test -o MUL_MAT 2>&1 | tail -3
echo "== depad    =="; "$PWD/build/bin/test-depad-hsa" 2>&1 | tail -3; echo "depad EXIT=$?"
```
Expected:
- bf16: `test_loss=0.066342`, `test_acc=98.00`, ~62 µs/image
- f32: `test_loss=0.066372`, `test_acc=98.00`, ~48 µs/image
- CPU: `test_loss≈0.06635`, `test_acc≈98.0`, ~2.8 µs/image
- MUL_MAT: PASS (3/3)
- depad: PASS, `EXIT=0`

- [ ] **Step 8: Commit**

```bash
cd /home/ypapadop/workspace-raiders/ggml
git add src/ggml-hsa/ggml-hsa.cpp src/ggml-hsa/common.hpp
git commit -m "$(cat <<'EOF'
Delete graph_compute MUL_MAT cast fusion machinery

Remove ggml_hsa_fuse_mul_mat_narrow, the fusion_t state
(narrow_dst/skip_dispatch/analyzed), the skip_dispatch continue, and the
post_dst branch, now that graph_optimize retypes the MUL_MAT and blanks
the cast up front. The framework's empty-op skip handles the orphaned
cast; the de-pad writes back into the (now bf16) node directly.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**
- Shared predicate `ggml_hsa_mul_mat_is_padded_gemm` → Task 1. ✓
- Relax prepare dst gate to accept bf16 → Task 1 Step 2 (folded in). ✓
- `ggml_backend_hsa_graph_optimize` hook (retype + rewire all cast consumers + blank to NONE) → Task 2. ✓
- All-consumers-are-casts + non-output guards → Task 2 Step 1. ✓
- Wire hook into iface → Task 2 Step 2. ✓
- Delete `fuse_mul_mat_narrow` + call site → Task 3 Steps 1-2. ✓
- Delete `fusion_t` + `fusion` member → Task 3 Step 5. ✓
- Delete `skip_dispatch` continue (replaced by empty-op skip) → Task 3 Step 3. ✓
- Delete `post_dst`/`narrow_dst` branch → Task 3 Step 4. ✓
- Tests: bf16/f32/CPU bit-exact + perf, MUL_MAT gate, depad gate → Task 3 Step 7 (and incrementally in Tasks 1-2). ✓
- depad.cc/py unchanged, no cache touch → honored (no task edits kernels). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code and exact commands with expected output. ✓

**3. Type consistency:** `ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor &)` defined in Task 1, consumed in Task 2 with the same signature. `ggml_backend_hsa_graph_optimize(ggml_backend_t, ggml_cgraph *)` matches the iface slot type (`void (*)(ggml_backend_t, struct ggml_cgraph *)`, ggml-backend-impl.h:139). `mm->type`/`set_contiguous_strides` mutation matches the de-pad's dtype-from-dst behavior asserted in the spec. ✓

**Note on the Task 1 `probe` shim:** `prepare_mul_mat` works on `node_t.tensor` copies, not the live graph node, so the predicate is fed a stack `ggml_tensor` mirroring op/type/shape with `src[0]/src[1]` pointed at the internal source tensors. This preserves the exact checks the predicate performs (op, src dtypes, trivial layout, batch) without depending on the live `src[]` wiring, which `node_t` does not carry.
