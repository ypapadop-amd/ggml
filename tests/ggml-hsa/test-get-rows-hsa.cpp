// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the GGML_OP_GET_ROWS host kernel on the HSA backend.
// Builds a real op graph (ggml_get_rows), computes it on the device, and compares
// against a CPU reference matching ggml_compute_forward_get_rows: output row i is
// a copy of src0 row idx[i] (with the higher dims of src1 indexing src0's dims 2/3).

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

// 2D case: src0 [nc, nrows] f32, indices [n_idx] i32 -> dst [nc, n_idx] f32.
bool run_case(ggml_backend_t backend, int64_t nc, int64_t nrows, int64_t n_idx, const char * name) {
    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nc, nrows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx);
    ggml_set_name(src, "src");
    ggml_set_name(idx, "idx");
    ggml_tensor * dst = ggml_get_rows(ctx.get(), src, idx);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-18s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-18s: graph allocation failed\n", name);
        return false;
    }

    std::vector<float> src_host(nc * nrows);
    for (int64_t r = 0; r < nrows; ++r) {
        for (int64_t c = 0; c < nc; ++c) {
            src_host[r * nc + c] = static_cast<float>(r) * 100.0f + static_cast<float>(c);
        }
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    // Deterministic, varied indices (including repeats and the last row).
    std::vector<int32_t> idx_host(n_idx);
    for (int64_t i = 0; i < n_idx; ++i) {
        idx_host[i] = static_cast<int32_t>((i * 7 + 3) % nrows);
    }
    ggml_backend_tensor_set(idx, idx_host.data(), 0, ggml_nbytes(idx));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-18s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(nc * n_idx);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i = 0; i < n_idx && ok; ++i) {
        const int64_t row = idx_host[i];
        for (int64_t c = 0; c < nc && ok; ++c) {
            const float want = src_host[row * nc + c];
            const float got = dst_host[i * nc + c];
            if (got != want) {
                printf("  %-18s: mismatch at out row %lld (src row %lld) col %lld got %g want %g\n",
                       name, (long long)i, (long long)row, (long long)c, got, want);
                ok = false;
            }
        }
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        int64_t nc, nrows, n_idx;
        const char * name;
    } cases[] = {
        {16, 8, 4, "small"},
        {32, 64, 10, "gather 10"},
        {64, 128, 1, "single idx"},
        {768, 50, 12, "gpt2 embd"},   // n_embd rows, token-like gather
        {128, 16, 32, "repeats"},     // n_idx > nrows, forces repeats
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.nc, c.nrows, c.n_idx, c.name);
        printf("GET_ROWS %-18s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
