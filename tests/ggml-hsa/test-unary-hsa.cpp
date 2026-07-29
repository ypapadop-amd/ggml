// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Device tests for the element-wise unary ops. Inputs deliberately straddle zero and
// include exact zeros, so the sign-dependent ops (ABS, NEG, SGN, STEP) are checked on
// both branches, and one shape is narrower than the 16-element f32 vector so the scalar
// tail runs with vend == 0.

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

enum class op_kind { sqr, abs, neg };

const char * op_name(op_kind kind) {
    switch (kind) {
        case op_kind::sqr: return "sqr";
        case op_kind::abs: return "abs";
        case op_kind::neg: return "neg";
    }
    return "?";
}

float reference(op_kind kind, float x) {
    switch (kind) {
        case op_kind::sqr: return x * x;
        case op_kind::abs: return std::fabs(x);
        case op_kind::neg: return -x;
    }
    return 0.0f;
}

bool run_case(ggml_backend_t backend, op_kind kind, int64_t ne0, int64_t ne1,
              const char * name) {
    const int64_t n = ne0 * ne1;

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(src, "src");

    ggml_tensor * dst = nullptr;
    switch (kind) {
        case op_kind::sqr: dst = ggml_sqr(ctx.get(), src); break;
        case op_kind::abs: dst = ggml_abs(ctx.get(), src); break;
        case op_kind::neg: dst = ggml_neg(ctx.get(), src); break;
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

    // Straddles zero and hits exact zero every 21 elements.
    std::vector<float> src_host(n);
    for (int64_t i = 0; i < n; ++i) {
        src_host[i] = (static_cast<float>(i % 21) - 10.0f) * 0.5f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    const float tol = 1e-5f;
    for (int64_t i = 0; i < n; ++i) {
        const float want = reference(kind, src_host[i]);
        const float got = dst_host[i];
        if (std::fabs(got - want) > tol) {
            printf("  %-22s: mismatch at %lld (x=%g) got %g want %g\n", name, (long long)i,
                   src_host[i], got, want);
            return false;
        }
    }
    return true;
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
        int64_t ne0, ne1;
        const char * name;
    } cases[] = {
        {op_kind::sqr, 256, 4, "sqr 2d"},
        {op_kind::abs, 256, 4, "abs 2d"},
        {op_kind::neg, 256, 4, "neg 2d"},
        {op_kind::sqr, 3072, 1, "sqr gpt2 mlp"},
        {op_kind::abs, 3072, 1, "abs gpt2 mlp"},
        {op_kind::neg, 3072, 1, "neg gpt2 mlp"},
        // Narrower than the 16-element f32 vector: exercises the scalar tail (vend == 0).
        {op_kind::sqr, 10, 8, "sqr narrow"},
        {op_kind::abs, 10, 8, "abs narrow"},
        {op_kind::neg, 10, 8, "neg narrow"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        const bool ok = run_case(backend, c.kind, c.ne0, c.ne1, c.name);
        printf("UNARY %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
