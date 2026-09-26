// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the GGML_OP_DIAG_MASK_INF kernel on the HSA backend.
// Builds a real op graph (ggml_diag_mask_inf), computes it on the device, and
// compares against a CPU reference matching ggml_compute_forward_diag_mask_f32:
// for row j of an [nc, nr, nz] tensor, column i is masked to -inf when
// i > n_past + j, and copied from the input otherwise.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

enum class case_result { pass, fail, skip };

case_result run_case(ggml_backend_t backend, int64_t nc, int64_t nr, int64_t nz, int n_past,
                     const char * name) {
    const int64_t n = nc * nr * nz;

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, nr, nz);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_diag_mask_inf(ctx.get(), src, n_past);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        // DIAG_MASK_INF is currently routed to the CPU fallback (see supports_op in ggml-hsa.cpp:
        // it faults the HSA queue inside integrated multi-op graphs such as GPT-2 attention, even
        // though the kernel is correct in isolation). Treat as a skip, not a failure.
        printf("  %-18s: op not supported (skipped)\n", name);
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-18s: graph allocation failed\n", name);
        return case_result::fail;
    }

    // Varied, deterministic input pattern.
    std::vector<float> src_host(n);
    for (int64_t i = 0; i < n; ++i) {
        src_host[i] = static_cast<float>(i % 97) * 0.5f - 13.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-18s: graph compute failed\n", name);
        return case_result::fail;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t k = 0; k < nz && ok; ++k) {
        for (int64_t j = 0; j < nr && ok; ++j) {
            for (int64_t i = 0; i < nc && ok; ++i) {
                const int64_t idx = (k * nr + j) * nc + i;
                const bool masked = i > n_past + j;
                const float want = masked ? -INFINITY : src_host[idx];
                const float got = dst_host[idx];
                const bool bad = masked ? !(std::isinf(got) && got < 0.0f) : (got != want);
                if (bad) {
                    printf("  %-18s: mismatch at [%lld,%lld,%lld] got %g want %g\n", name,
                           (long long)i, (long long)j, (long long)k, got, want);
                    ok = false;
                }
            }
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
        int64_t nc, nr, nz;
        int n_past;
        const char * name;
    } cases[] = {
        {32, 32, 1, 0, "square npast0"},
        {64, 32, 1, 0, "wide npast0"},
        {32, 64, 1, 0, "tall npast0"},
        {32, 32, 1, 8, "square npast8"},
        {64, 32, 1, 16, "wide npast16"},
        {48, 16, 1, 5, "npast5"},
        {32, 32, 4, 0, "3d npast0"},
        {64, 32, 3, 12, "3d npast12"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.nc, c.nr, c.nz, c.n_past, c.name);
        const char * label = r == case_result::pass ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("DIAG_MASK_INF %-18s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    ggml_backend_free(backend);
    if (any_fail) {
        printf("FAILURES\n");
        return 1;
    }
    if (skipped > 0 && passed == 0) {
        printf("ALL SKIPPED (DIAG_MASK_INF routed to CPU)\n");
    } else {
        printf("ALL PASSED\n");
    }
    return 0;
}
