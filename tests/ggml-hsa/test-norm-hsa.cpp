// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the GGML_OP_NORM kernel on the HSA backend.
// Builds a real op graph (ggml_norm), computes it on the device, and compares
// against a CPU reference matching ggml_compute_forward_norm_f32: each row over
// dim 0 is normalized as y = (x - mean) / sqrt(variance + eps), where mean and
// variance are the population statistics over the ne00 row elements.

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

bool run_case(ggml_backend_t backend, int64_t nc, int64_t nr, int64_t nz, float eps,
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
    ggml_tensor * dst = ggml_norm(ctx.get(), src, eps);
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

    // Varied, deterministic input pattern with a per-row offset so rows differ.
    std::vector<float> src_host(n);
    for (int64_t r = 0; r < nr * nz; ++r) {
        for (int64_t i = 0; i < nc; ++i) {
            src_host[r * nc + i] = static_cast<float>(i % 97) * 0.5f - 13.0f +
                                   static_cast<float>(r) * 0.25f;
        }
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-18s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // CPU reference, per row.
    const float tol = 1e-3f;
    bool ok = true;
    for (int64_t r = 0; r < nr * nz && ok; ++r) {
        const float * x = src_host.data() + r * nc;

        double sum = 0.0;
        for (int64_t i = 0; i < nc; ++i) {
            sum += x[i];
        }
        const double mean = sum / static_cast<double>(nc);

        double var = 0.0;
        for (int64_t i = 0; i < nc; ++i) {
            const double v = x[i] - mean;
            var += v * v;
        }
        var /= static_cast<double>(nc);
        const double scale = 1.0 / std::sqrt(var + static_cast<double>(eps));

        for (int64_t i = 0; i < nc && ok; ++i) {
            const float want = static_cast<float>((x[i] - mean) * scale);
            const float got = dst_host[r * nc + i];
            if (std::fabs(got - want) > tol) {
                printf("  %-18s: mismatch at row %lld col %lld got %g want %g\n", name,
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

    const float eps = 1e-5f;

    struct {
        int64_t nc, nr, nz;
        const char * name;
    } cases[] = {
        {32, 1, 1, "single row"},
        {64, 8, 1, "wide"},
        {128, 4, 1, "wider"},
        {48, 16, 1, "many rows"},
        {768, 4, 1, "gpt2 n_embd"},
        {32, 8, 4, "3d"},
        {64, 4, 3, "3d wide"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.nc, c.nr, c.nz, eps, c.name);
        printf("NORM %-18s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
