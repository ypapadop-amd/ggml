// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the source pre-processing / output post-processing paths of
// ggml_backend_hsa_tensor_extra. The device has no f16 kernels, so an op with f16 operands runs as
// bf16 internally: each f16 source is converted into its own internal buffer before the dispatch
// (sources.sync_mode) and the bf16 result is converted back into the f16 parent afterwards
// (node.sync_mode).
//
// Both source paths are exercised. With an even element count the HSA_CONVERT kernel builds and the
// conversion is dispatched on the device queue (sync_mode_t::device); with an odd element count the
// kernel legitimately declines -- a 2-byte tensor is then not a whole number of DMA words -- and the
// conversion falls back to the host copy path (sync_mode_t::host). Both must produce identical,
// correct results; that fallback is the point of the test.
//
// Inputs are small integers, which are exact in f16 and in bf16, and so are their sums. That makes
// the expected result an exact integer regardless of how many times the value is round-tripped
// through the two formats, so the comparison can be exact rather than tolerance-based.

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

using hsa_test::load_val;
using hsa_test::store_val;

enum class case_result { pass, fail, skip };

// Values chosen to be exactly representable in bf16 (8 significand bits), so no rounding occurs
// anywhere along f16 -> bf16 -> add -> bf16 -> f16.
float src0_val(int64_t i) { return static_cast<float>(i % 61) - 30.0f; }
float src1_val(int64_t i) { return static_cast<float>(i % 29) - 14.0f; }

case_result run_case(ggml_backend_t backend, int64_t d0, int64_t d1) {
    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, d0, d1);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, d0, d1);
    ggml_set_name(b, "b");
    ggml_tensor * dst = ggml_add(ctx.get(), a, b);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
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

    const int64_t n = d0 * d1;
    std::vector<uint8_t> a_bytes(ggml_nbytes(a));
    std::vector<uint8_t> b_bytes(ggml_nbytes(b));
    for (int64_t i = 0; i < n; ++i) {
        store_val(GGML_TYPE_F16, a_bytes.data(), i, src0_val(i));
        store_val(GGML_TYPE_F16, b_bytes.data(), i, src1_val(i));
    }
    ggml_backend_tensor_set(a, a_bytes.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bytes.data(), 0, ggml_nbytes(b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        const float want = src0_val(i) + src1_val(i);
        const float got = load_val(GGML_TYPE_F16, dst_bytes.data(), i);
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
        {64, 8, "even numel (on-queue convert)"},
        {256, 4, "even numel, larger"},
        {1, 64, "single col"},
        // Odd element count: a 2-byte tensor is not a whole number of DMA words, so HSA_CONVERT
        // declines and the source conversion falls back to the host copy path.
        {15, 1, "odd numel (host fallback)"},
        {33, 3, "odd numel rows (host fallback)"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.d0, c.d1);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("ADD f16 %-32s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    ggml_backend_free(backend);
    if (any_fail) {
        printf("FAILURES\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
