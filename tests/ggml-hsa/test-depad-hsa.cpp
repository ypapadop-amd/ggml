// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the HSA-only ggml_hsa_depad op: gathers the top-left [d0, d1] sub-block out of
// a padded [d0pad, d1pad] source, converting to the destination dtype. Builds a real single-node op
// graph and computes it on the device. Covers f32->f32, bf16->bf16, and the fused f32->bf16 cast
// (the real MUL_MAT de-pad post-amble). The padded source regions carry a nonzero pattern that the
// gather must ignore.

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

bool run_case(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type, int64_t d0,
              int64_t d1, int64_t d0pad, int64_t d1pad) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0pad, d1pad);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_depad(ctx.get(), src, dst_type, d0, d1);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported\n");
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return false;
    }

    // fill the padded source with a varied pattern (including nonzero pad regions, which must be
    // ignored by the gather)
    std::vector<uint8_t> src_bytes(ggml_nbytes(src));
    for (int64_t i = 0; i < d0pad * d1pad; ++i) {
        store_val(src_type, src_bytes.data(), i, static_cast<float>(i) * 0.25f + 1.0f);
    }
    ggml_backend_tensor_set(src, src_bytes.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return false;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i1 = 0; i1 < d1 && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0 && ok; ++i0) {
            // reference: gather the sub-block value (as stored in src_type), then cast to dst_type
            const float src_v = load_val(src_type, src_bytes.data(), i1 * d0pad + i0);
            const float want = cast_val(dst_type, src_v);
            const float got = load_val(dst_type, dst_bytes.data(), i1 * d0 + i0);
            if (got != want) {
                printf("  mismatch at [%lld,%lld]: got %g want %g\n", (long long)i0, (long long)i1,
                       got, want);
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
        int64_t d0, d1, d0pad, d1pad;
        const char * name;
    } cases[] = {
        {128, 128, 128, 128, "exact (no pad)"},
        {40, 128, 64, 128, "d0 pad"},
        {128, 10, 128, 128, "d1 pad"},
        {40, 10, 64, 128, "d0+d1 pad"},
        // exact MNIST MUL_MAT output shapes: parent C [M,N] <- padded [Mpad,Npad]
        {500, 500, 512, 512, "mnist C c1"},
        {10, 500, 128, 512, "mnist C c2"},
        // large sizes that exceed a single strided-DMA descriptor's wrap limits
        // (regression guard for the linear-transfer + in-kernel narrowing design)
        {500, 4, 512, 128, "large d0"},
        {8, 500, 128, 512, "large d1"},
    };

    struct {
        ggml_type src_type, dst_type;
        const char * label;
    } variants[] = {
        {GGML_TYPE_F32, GGML_TYPE_F32, "HSA_DEPAD"},
        {GGML_TYPE_BF16, GGML_TYPE_BF16, "HSA_DEPAD bf16"},
        {GGML_TYPE_F32, GGML_TYPE_BF16, "HSA_DEPAD f32->bf16"},
    };

    bool all_ok = true;
    for (const auto & v : variants) {
        for (const auto & c : cases) {
            bool ok = run_case(backend, v.src_type, v.dst_type, c.d0, c.d1, c.d0pad, c.d1pad);
            printf("%s %-16s: %s\n", v.label, c.name, ok ? "PASSED" : "FAILED");
            all_ok = all_ok && ok;
        }
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
