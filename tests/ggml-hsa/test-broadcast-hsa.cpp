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

enum class op_kind { add, mul };

// src0 [nc, nr, nz] op src1 [nc] (broadcast over all rows/slices) -> dst [nc, nr, nz].
bool run_case(ggml_backend_t backend, op_kind kind, int64_t nc, int64_t nr, int64_t nz,
              const char * name) {
    const int64_t n = nc * nr * nz;

    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src0 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, nr, nz);
    ggml_tensor * src1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, nc);  // broadcast operand
    ggml_set_name(src0, "src0");
    ggml_set_name(src1, "src1");
    ggml_tensor * dst = (kind == op_kind::add) ? ggml_add(ctx.get(), src0, src1)
                                               : ggml_mul(ctx.get(), src0, src1);
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
    std::vector<float> b_host(nc);
    for (int64_t i = 0; i < nc; ++i) {
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
            const float want =
                (kind == op_kind::add) ? a_host[idx] + b_host[i] : a_host[idx] * b_host[i];
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
    } cases[] = {
        {op_kind::add, 32, 8, 1, "add bias 2d"},
        {op_kind::add, 768, 4, 1, "add bias gpt2"},
        {op_kind::mul, 32, 8, 1, "mul bcast 2d"},
        {op_kind::mul, 768, 4, 1, "mul bcast gpt2"},
        {op_kind::add, 64, 4, 3, "add bias 3d"},
        {op_kind::mul, 64, 4, 3, "mul bcast 3d"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.kind, c.nc, c.nr, c.nz, c.name);
        printf("BROADCAST %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
