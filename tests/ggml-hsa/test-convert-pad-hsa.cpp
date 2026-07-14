// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the internal HSA_CONVERT_PAD kernel: f32 [d0, d1] -> bf16 [d0pad, d1pad],
// converting the valid sub-block (round-to-nearest-even, matching the host reference) and leaving
// the padded regions zero. Exercises the real build+dispatch path via the internal test hook.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

bool run_case(ggml_backend_t backend, int64_t d0, int64_t d1, int64_t d0pad, int64_t d1pad) {
    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead() * 2 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d0pad, d1pad);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // fill source with a varied pattern; pre-zero the (padded) destination
    std::vector<float> src_host(d0 * d1);
    for (int64_t i = 0; i < d0 * d1; ++i) {
        src_host[i] = static_cast<float>(i % 97) * 0.5f - 13.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    std::vector<uint16_t> dst_zero(d0pad * d1pad, 0);
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, ggml_nbytes(dst));

    ggml_status status = ggml_hsa_test_dispatch_transform(backend, "HSA_CONVERT_PAD", src, dst);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  dispatch failed (status=%d)\n", (int)status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint16_t> dst_host(d0pad * d1pad);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i1 = 0; i1 < d1pad && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0pad && ok; ++i0) {
            uint16_t got = dst_host[i1 * d0pad + i0];
            uint16_t want = (i0 < d0 && i1 < d1) ? ggml_fp32_to_bf16(src_host[i1 * d0 + i0]).bits : 0;
            if (got != want) {
                printf("  mismatch at [%lld,%lld]: got 0x%04x want 0x%04x\n", (long long)i0,
                       (long long)i1, got, want);
                ok = false;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
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
        {32, 128, 32, 128, "exact (no pad)"},
        {40, 128, 64, 128, "d0 pad"},
        {32, 40, 32, 128, "d1 pad"},
        {40, 40, 64, 128, "d0+d1 pad"},
        // exact MNIST MUL_MAT operand shapes (parent f32 [K,M]/[K,N] -> padded bf16)
        {784, 500, 800, 512, "mnist A/B c1"},
        {500, 10, 512, 128, "mnist A c2"},
        {500, 500, 512, 512, "mnist B c2"},
        {784, 10, 800, 128, "mnist A c3"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.d0, c.d1, c.d0pad, c.d1pad);
        printf("HSA_CONVERT_PAD %-16s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
