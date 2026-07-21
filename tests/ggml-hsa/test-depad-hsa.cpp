// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the internal HSA_DEPAD kernel: f32 [d0pad, d1pad] -> f32 [d0, d1] (or bf16
// [d0pad, d1pad] -> bf16 [d0, d1]), gathering the top-left [d0, d1] sub-block out of a padded
// buffer. Exercises the real build+dispatch path via the internal test hook.

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

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0pad, d1pad);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0, d1);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // fill the padded source with a varied pattern (including nonzero pad regions, which must be
    // ignored by the gather)
    std::vector<float> src_host(d0pad * d1pad);
    for (int64_t i = 0; i < d0pad * d1pad; ++i) {
        src_host[i] = static_cast<float>(i) * 0.25f + 1.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    ggml_status status = ggml_hsa_test_dispatch_transform(backend, "HSA_DEPAD", src, dst);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  dispatch failed (status=%d)\n", (int)status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> dst_host(d0 * d1);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i1 = 0; i1 < d1 && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0 && ok; ++i0) {
            float got = dst_host[i1 * d0 + i0];
            float want = src_host[i1 * d0pad + i0];
            if (got != want) {
                printf("  mismatch at [%lld,%lld]: got %f want %f\n", (long long)i0, (long long)i1,
                       got, want);
                ok = false;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool run_case_bf16(ggml_backend_t backend, int64_t d0, int64_t d1, int64_t d0pad, int64_t d1pad) {
    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead() * 2 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d0pad, d1pad);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d0, d1);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // fill the padded source with a varied bf16 pattern (including nonzero pad regions, which
    // must be ignored by the gather)
    std::vector<uint16_t> src_host(d0pad * d1pad);
    for (int64_t i = 0; i < d0pad * d1pad; ++i) {
        src_host[i] = ggml_fp32_to_bf16(static_cast<float>(i) * 0.25f + 1.0f).bits;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    ggml_status status = ggml_hsa_test_dispatch_transform(backend, "HSA_DEPAD", src, dst);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  dispatch failed (status=%d)\n", (int)status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint16_t> dst_host(d0 * d1);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i1 = 0; i1 < d1 && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0 && ok; ++i0) {
            uint16_t got = dst_host[i1 * d0 + i0];
            uint16_t want = src_host[i1 * d0pad + i0];
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

// f32 padded source -> bf16 dense destination (the fused per-layer cast path used by the real
// MUL_MAT post-amble). The reference casts the gathered f32 sub-block to bf16 (round-to-nearest).
bool run_case_f32_to_bf16(ggml_backend_t backend, int64_t d0, int64_t d1, int64_t d0pad,
                          int64_t d1pad) {
    ggml_init_params params{
        /*.mem_size   =*/ggml_tensor_overhead() * 2 + 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d0pad, d1pad);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, d0, d1);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> src_host(d0pad * d1pad);
    for (int64_t i = 0; i < d0pad * d1pad; ++i) {
        src_host[i] = static_cast<float>(i) * 0.25f + 1.0f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    ggml_status status = ggml_hsa_test_dispatch_transform(backend, "HSA_DEPAD", src, dst);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  dispatch failed (status=%d)\n", (int)status);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint16_t> dst_host(d0 * d1);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i1 = 0; i1 < d1 && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0 && ok; ++i0) {
            uint16_t got = dst_host[i1 * d0 + i0];
            uint16_t want = ggml_fp32_to_bf16(src_host[i1 * d0pad + i0]).bits;
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

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.d0, c.d1, c.d0pad, c.d1pad);
        printf("HSA_DEPAD %-16s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    for (const auto & c : cases) {
        bool ok = run_case_bf16(backend, c.d0, c.d1, c.d0pad, c.d1pad);
        printf("HSA_DEPAD bf16 %-16s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    for (const auto & c : cases) {
        bool ok = run_case_f32_to_bf16(backend, c.d0, c.d1, c.d0pad, c.d1pad);
        printf("HSA_DEPAD f32->bf16 %-16s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
