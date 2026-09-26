// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Verifies broadcasting for GGML_OP_ADD / GGML_OP_MUL on the HSA backend, i.e.
// src1 broadcast over src0/dst. These are the patterns GPT-2 relies on:
//   - bias add:  cur[n_embd, N] + b[n_embd]          (ADD bias fast path)
//   - LN affine: norm[n_embd, N] * w[n_embd]         (general broadcast, MUL)
//   - LN affine: (...)          + b[n_embd]          (ADD broadcast)
// Builds a real op graph, computes on device, and compares to a CPU reference.

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

enum class op_kind { add, sub, mul, div };

// broadcast: src0 [nc, nr, nz] op src1 [nc] (src1 reused over all rows/slices).
// !broadcast: src0 [nc, nr, nz] op src1 [nc, nr, nz], the plain element-wise path, which is a
// different kernel (ggml_op_add/sub/mul/div rather than the *_row / *_broadcast variants).
// Both land in dst [nc, nr, nz].
bool run_case(ggml_backend_t backend,
              op_kind kind,
              int64_t nc,
              int64_t nr,
              int64_t nz,
              const char * name,
              bool broadcast = true) {
    const int64_t n = nc * nr * nz;

    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src0 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, nr, nz);
    ggml_tensor * src1 = broadcast ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nc)
                                   : ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, nr, nz);
    ggml_set_name(src0, "src0");
    ggml_set_name(src1, "src1");
    ggml_tensor * dst = nullptr;
    switch (kind) {
        case op_kind::add:
            dst = ggml_add(ctx.get(), src0, src1);
            break;
        case op_kind::sub:
            dst = ggml_sub(ctx.get(), src0, src1);
            break;
        case op_kind::mul:
            dst = ggml_mul(ctx.get(), src0, src1);
            break;
        case op_kind::div:
            dst = ggml_div(ctx.get(), src0, src1);
            break;
    }
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-22s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-22s: graph allocation failed\n", name);
        return false;
    }

    std::vector<float> a_host(n);
    for (int64_t i = 0; i < n; ++i) {
        a_host[i] = static_cast<float>(i % 101) * 0.25f - 7.0f;
    }
    std::vector<float> b_host(broadcast ? nc : n);
    for (int64_t i = 0; i < (int64_t)b_host.size(); ++i) {
        b_host[i] = static_cast<float>(i % 13) * 0.5f + 1.0f;
    }
    ggml_backend_tensor_set(src0, a_host.data(), 0, ggml_nbytes(src0));
    ggml_backend_tensor_set(src1, b_host.data(), 0, ggml_nbytes(src1));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    const float tol = 1e-4f;
    bool ok = true;
    for (int64_t r = 0; r < nr * nz && ok; ++r) {
        for (int64_t i = 0; i < nc && ok; ++i) {
            const int64_t idx = r * nc + i;
            const int64_t b_idx = broadcast ? i : idx;
            float want = 0.0f;
            switch (kind) {
                case op_kind::add:
                    want = a_host[idx] + b_host[b_idx];
                    break;
                case op_kind::sub:
                    want = a_host[idx] - b_host[b_idx];
                    break;
                case op_kind::mul:
                    want = a_host[idx] * b_host[b_idx];
                    break;
                case op_kind::div:
                    want = a_host[idx] / b_host[b_idx];
                    break;
            }
            const float got = dst_host[idx];
            if (std::fabs(got - want) > tol) {
                printf("  %-22s: mismatch at row %lld col %lld got %g want %g\n", name,
                       (long long)r, (long long)i, got, want);
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
        op_kind kind;
        int64_t nc, nr, nz;
        const char * name;
        bool broadcast;
    } cases[] = {
        {op_kind::add, 32, 8, 1, "add bias 2d", true},
        {op_kind::add, 768, 4, 1, "add bias gpt2", true},
        {op_kind::mul, 32, 8, 1, "mul bcast 2d", true},
        {op_kind::mul, 768, 4, 1, "mul bcast gpt2", true},
        {op_kind::add, 64, 4, 3, "add bias 3d", true},
        {op_kind::mul, 64, 4, 3, "mul bcast 3d", true},
        {op_kind::sub, 32, 8, 1, "sub bcast 2d", true},
        {op_kind::sub, 768, 4, 1, "sub bcast gpt2", true},
        {op_kind::sub, 64, 4, 3, "sub bcast 3d", true},
        {op_kind::div, 32, 8, 1, "div bcast 2d", true},
        {op_kind::div, 768, 4, 1, "div bcast gpt2", true},
        {op_kind::div, 64, 4, 3, "div bcast 3d", true},
        // Row narrower than the 16-element f32 vector: exercises the scalar tail with
        // vend == 0, the case that rules out AIE_LOOP_MIN_ITERATION_COUNT.
        {op_kind::add, 10, 64, 1, "add bias narrow row", true},
        {op_kind::mul, 10, 64, 1, "mul bcast narrow row", true},
        // Plain element-wise (src1 same shape as src0): a different kernel from the broadcast
        // cases above, and the one the vectorized transform_vector_n body serves. 768*1024
        // matches the shape the ADD benchmark reports.
        {op_kind::add, 768, 64, 1, "add elementwise", false},
        {op_kind::sub, 768, 64, 1, "sub elementwise", false},
        {op_kind::mul, 768, 64, 1, "mul elementwise", false},
        {op_kind::div, 768, 64, 1, "div elementwise", false},
        {op_kind::add, 64, 4, 3, "add elementwise 3d", false},
        {op_kind::mul, 64, 4, 3, "mul elementwise 3d", false},
        // 700 elements: no multiple of the 16-element f32 vector divides it, so tiled_tile_size
        // falls back to max_tile_size and the tile lands below V. vend is then 0 and the whole
        // range runs on the scalar path -- confirmed by mutation-testing the vector body, which
        // this case does not catch. It covers the scalar formulation, not the vector/tail seam.
        {op_kind::add, 100, 7, 1, "add elementwise ragged", false},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.kind, c.nc, c.nr, c.nz, c.name, c.broadcast);
        printf("BROADCAST %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
