// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for GGML_UNARY_OP_GELU on the HSA backend. Builds a real
// ggml_gelu graph, runs it on the device, and compares against the tanh
// approximation GGML uses:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// A relaxed tolerance absorbs the AIE scalar_exp/tanh approximation vs. libm.

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

float gelu_ref(float x) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    const float a = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(k * (x + a * x * x * x)));
}

bool run_case(ggml_backend_t backend, int64_t ne0, int64_t ne1, const char * name) {
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
    ggml_tensor * dst = ggml_gelu(ctx.get(), src);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-14s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-14s: graph allocation failed\n", name);
        return false;
    }

    // Spread inputs across the saturating and near-zero regions of GELU.
    std::vector<float> src_host(n);
    for (int64_t i = 0; i < n; ++i) {
        src_host[i] = (static_cast<float>(i % 201) - 100.0f) * 0.1f;  // [-10, 10]
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-14s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    const float tol = 2e-3f;
    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        const float want = gelu_ref(src_host[i]);
        const float got = dst_host[i];
        if (std::fabs(got - want) > tol) {
            printf("  %-14s: mismatch at %lld (x=%g) got %g want %g\n", name, (long long)i,
                   src_host[i], got, want);
            ok = false;
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
        int64_t ne0, ne1;
        const char * name;
    } cases[] = {
        {64, 1, "1d"},
        {768, 4, "gpt2 embd"},
        {3072, 8, "gpt2 mlp"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.ne0, c.ne1, c.name);
        printf("GELU %-14s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
