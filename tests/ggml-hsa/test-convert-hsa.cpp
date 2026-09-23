// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the HSA-only ggml_hsa_convert op: an element-wise dtype cast with no shape
// change (the on-device GGML_OP_CPY cast). Builds a real single-node op graph and computes it on the
// device. Covers f32->bf16 (round-to-nearest-even, bit-identical to the host reference), bf16->f32
// (exact widening), and the same-dtype plain copy.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"
#include "hsa-test-common.hpp"

namespace {

using hsa_test::cast_val;
using hsa_test::load_val;
using hsa_test::store_val;

enum class case_result { pass, fail, skip };

case_result run_case(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type, int64_t d0,
                     int64_t d1) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0, d1);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_convert(ctx.get(), src, dst_type);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        // Not every shape/dtype pair is streamable: a tensor whose byte size is not a whole
        // number of DMA words (a bf16 tensor with an odd element count) is left to the host copy
        // path rather than over-running the allocation. Treat as a skip, not a failure.
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    // varied, deterministic input pattern
    const int64_t n = d0 * d1;
    std::vector<uint8_t> src_bytes(ggml_nbytes(src));
    for (int64_t i = 0; i < n; ++i) {
        store_val(src_type, src_bytes.data(), i, static_cast<float>(i % 97) * 0.5f - 13.0f);
    }
    ggml_backend_tensor_set(src, src_bytes.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        // reference: read the source value (as stored in src_type), then cast to dst_type
        const float want = cast_val(dst_type, load_val(src_type, src_bytes.data(), i));
        const float got = load_val(dst_type, dst_bytes.data(), i);
        if (got != want) {
            printf("  mismatch at %lld: got %g want %g\n", (long long)i, got, want);
            ok = false;
        }
    }

    return ok ? case_result::pass : case_result::fail;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        int64_t d0, d1;
        const char * name;
    } cases[] = {
        {32, 1, "single row"},
        {64, 8, "wide"},
        {128, 4, "wider"},
        {48, 16, "many rows"},
        {768, 4, "gpt2 n_embd"},
        {500, 500, "square"},
        {1, 64, "single col"},
        // Odd element count: a bf16 tensor is then not a whole number of DMA words, so those
        // variants must be declined rather than rounded up into the next allocation. The f32->f32
        // variant of the same shape still runs, which is what distinguishes "correctly declined"
        // from "the whole shape broke".
        {15, 1, "odd numel"},
        {33, 3, "odd numel rows"},
    };

    struct {
        ggml_type src_type, dst_type;
        const char * label;
    } variants[] = {
        {GGML_TYPE_F32, GGML_TYPE_BF16, "HSA_CONVERT f32->bf16"},
        {GGML_TYPE_BF16, GGML_TYPE_F32, "HSA_CONVERT bf16->f32"},
        {GGML_TYPE_F32, GGML_TYPE_F32, "HSA_CONVERT f32->f32"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & v : variants) {
        for (const auto & c : cases) {
            const case_result r = run_case(backend, v.src_type, v.dst_type, c.d0, c.d1);
            const char * label = r == case_result::pass   ? "PASSED"
                                 : r == case_result::skip ? "SKIPPED"
                                                          : "FAILED";
            printf("%s %-14s: %s\n", v.label, c.name, label);
            any_fail = any_fail || (r == case_result::fail);
            passed += (r == case_result::pass);
            skipped += (r == case_result::skip);
        }
    }

    ggml_backend_free(backend);
    if (any_fail) {
        printf("FAILURES\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
