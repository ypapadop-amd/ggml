// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the GGML_OP_GET_ROWS host kernel on the HSA backend.
// Builds a real op graph (ggml_get_rows), computes it on the device, and compares
// against a CPU reference matching ggml_compute_forward_get_rows: output row i is
// a copy of src0 row idx[i] (with the higher dims of src1 indexing src0's dims 2/3).

#include <algorithm>
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

/// Stores @p in (nc*nr*nm floats) into @p out as @p type, and writes back the values as the
/// device will see them (i.e. round-tripped through @p type) so the reference compares exactly.
bool store_as(ggml_type type, std::vector<float> & in, std::vector<std::byte> & out) {
    const int64_t n = static_cast<int64_t>(in.size());
    switch (type) {
        case GGML_TYPE_F32:
            out.resize(n * sizeof(float));
            std::copy_n(reinterpret_cast<const std::byte *>(in.data()), out.size(), out.data());
            return true;
        case GGML_TYPE_F16: {
            out.resize(n * sizeof(ggml_fp16_t));
            auto * p = reinterpret_cast<ggml_fp16_t *>(out.data());
            ggml_fp32_to_fp16_row(in.data(), p, n);
            ggml_fp16_to_fp32_row(p, in.data(), n);
            return true;
        }
        case GGML_TYPE_BF16: {
            out.resize(n * sizeof(ggml_bf16_t));
            auto * p = reinterpret_cast<ggml_bf16_t *>(out.data());
            ggml_fp32_to_bf16_row(in.data(), p, n);
            ggml_bf16_to_fp32_row(p, in.data(), n);
            return true;
        }
        default:
            return false;
    }
}

// src0 [nc, nrows, nmat] of src_type, indices [n_idx, nmat] i32 -> dst [nc, n_idx, nmat] f32.
// With nmat > 1 this exercises the i11 / nb[2] path of the gather.
case_result run_case(ggml_backend_t backend, int64_t nc, int64_t nrows, int64_t n_idx,
                     int64_t nmat, ggml_type src_type, const char * name) {
    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};
    if (ctx == nullptr) {
        printf("  %-18s: could not create context\n", name);
        return case_result::fail;
    }

    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), src_type, nc, nrows, nmat);
    ggml_tensor * idx = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_idx, nmat);
    ggml_set_name(src, "src");
    ggml_set_name(idx, "idx");
    ggml_tensor * dst = ggml_get_rows(ctx.get(), src, idx);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        // Support is decided by the dtype combination the host gather accepts, so an
        // unsupported op here means this combination is not claimed rather than that the
        // test failed. Treat as a skip so a partially-supported backend still reports usefully.
        printf("  %-18s: op not supported (skipped)\n", name);
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-18s: graph allocation failed\n", name);
        return case_result::fail;
    }

    // Varied, deterministic source pattern; distinct per (matrix, row, column).
    std::vector<float> src_ref(nc * nrows * nmat);
    for (int64_t m = 0; m < nmat; ++m) {
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t c = 0; c < nc; ++c) {
                src_ref[(m * nrows + r) * nc + c] =
                    static_cast<float>(m) * 10000.0f + static_cast<float>(r) * 100.0f +
                    static_cast<float>(c);
            }
        }
    }
    std::vector<std::byte> src_bytes;
    if (!store_as(src_type, src_ref, src_bytes)) {
        printf("  %-18s: unhandled source type\n", name);
        return case_result::fail;
    }
    ggml_backend_tensor_set(src, src_bytes.data(), 0, ggml_nbytes(src));

    // Deterministic indices that always include row 0, the last row, and a repeat: the
    // stride-7 walk alone is a permutation whenever gcd(7, nrows) == 1, so it would leave
    // both boundary rows and the aliasing case untested for most shapes.
    std::vector<int32_t> idx_host(n_idx * nmat);
    for (int64_t m = 0; m < nmat; ++m) {
        for (int64_t i = 0; i < n_idx; ++i) {
            int64_t row = 0;
            if (i == 0 || n_idx == 1) {
                row = 0;
            } else if (i == 1) {
                row = nrows - 1;
            } else if (i == 2) {
                row = nrows - 1; // repeat of the previous index
            } else {
                row = (i * 7 + 3 + m) % nrows;
            }
            idx_host[m * n_idx + i] = static_cast<int32_t>(row);
        }
    }
    ggml_backend_tensor_set(idx, idx_host.data(), 0, ggml_nbytes(idx));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-18s: graph compute failed\n", name);
        return case_result::fail;
    }

    std::vector<float> dst_host(nc * n_idx * nmat);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t m = 0; m < nmat && ok; ++m) {
        for (int64_t i = 0; i < n_idx && ok; ++i) {
            const int64_t row = idx_host[m * n_idx + i];
            for (int64_t c = 0; c < nc && ok; ++c) {
                const float want = src_ref[(m * nrows + row) * nc + c];
                const float got = dst_host[(m * n_idx + i) * nc + c];
                if (got != want) {
                    printf("  %-18s: mismatch at mat %lld out row %lld (src row %lld) col %lld "
                           "got %g want %g\n",
                           name, (long long)m, (long long)i, (long long)row, (long long)c, got,
                           want);
                    ok = false;
                }
            }
        }
    }
    return ok ? case_result::pass : case_result::fail;
}

/// An index outside [0, nrows) must be rejected rather than read out of bounds.
case_result run_out_of_range_case(ggml_backend_t backend) {
    const char * name = "out of range";
    const int64_t nc = 16;
    const int64_t nrows = 8;
    const int64_t n_idx = 4;

    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};
    if (ctx == nullptr) {
        printf("  %-18s: could not create context\n", name);
        return case_result::fail;
    }

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nc, nrows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx);
    ggml_set_name(src, "src");
    ggml_set_name(idx, "idx");
    ggml_tensor * dst = ggml_get_rows(ctx.get(), src, idx);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-18s: op not supported (skipped)\n", name);
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-18s: graph allocation failed\n", name);
        return case_result::fail;
    }

    std::vector<float> src_host(nc * nrows, 0.0f);
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    // Last index is past the end of the table; the gather must fail the graph.
    std::vector<int32_t> idx_host{0, 1, 2, static_cast<int32_t>(nrows)};
    ggml_backend_tensor_set(idx, idx_host.data(), 0, ggml_nbytes(idx));

    printf("  %-18s: expecting a rejected index below\n", name);
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        printf("  %-18s: out-of-range index was accepted\n", name);
        return case_result::fail;
    }
    return case_result::pass;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        int64_t nc, nrows, n_idx, nmat;
        ggml_type src_type;
        const char * name;
    } cases[] = {
        {16, 8, 4, 1, GGML_TYPE_F32, "small"},
        {32, 64, 10, 1, GGML_TYPE_F32, "gather 10"},
        {64, 128, 1, 1, GGML_TYPE_F32, "single idx"},
        {768, 50, 12, 1, GGML_TYPE_F32, "gpt2 embd"},  // n_embd rows, token-like gather
        {128, 16, 32, 1, GGML_TYPE_F32, "repeats"},    // n_idx > nrows, forces repeats
        {32, 16, 6, 3, GGML_TYPE_F32, "3d f32"},       // nmat > 1: exercises the nb[2] path
        {64, 32, 8, 1, GGML_TYPE_F16, "f16 table"},    // f16 -> f32 conversion
        {64, 32, 8, 1, GGML_TYPE_BF16, "bf16 table"},  // bf16 -> f32 conversion
        {32, 16, 6, 3, GGML_TYPE_BF16, "3d bf16"},     // conversion on the nb[2] path
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r =
            run_case(backend, c.nc, c.nrows, c.n_idx, c.nmat, c.src_type, c.name);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("GET_ROWS %-18s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    {
        const case_result r = run_out_of_range_case(backend);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("GET_ROWS %-18s: %s\n", "out of range", label);
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
        printf("ALL SKIPPED (GET_ROWS not claimed by the backend)\n");
    } else {
        printf("ALL PASSED\n");
    }
    return 0;
}
