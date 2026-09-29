// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for MUL_MAT accumulating over more than one K tile. The GEMM zeroes each core's
// C tile once and then accumulates one K tile per microkernel call, so every call after the first
// starts from a non-zero C. An accumulator loaded from the wrong address is invisible on a single
// K tile (every C block is still zero) and only shows up here. Shapes are bf16 x bf16 -> f32 and
// already tile multiples, so no padding or conversion kernel is involved and the result is the
// GEMM's alone. Inputs are small integers, so every product and partial sum is exact in bf16/f32
// and the result must match the host reference bit for bit.

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

case_result run_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K) {
    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, M);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, N);
    ggml_set_name(b, "b");
    ggml_tensor * c = ggml_mul_mat(ctx.get(), a, b); // [M, N] f32
    ggml_set_name(c, "c");

    if (!ggml_backend_supports_op(backend, c)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    // Different periods per operand keep neighbouring rows/columns distinct, so an accumulator
    // read from another C block cannot coincide with the right value.
    std::vector<float> fa(M * K);
    std::vector<float> fb(N * K);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            fa[i * K + k] = static_cast<float>((i * 7 + k * 3) % 5 - 2);
        }
    }
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            fb[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }
    std::vector<ggml_bf16_t> ha(fa.size());
    std::vector<ggml_bf16_t> hb(fb.size());
    ggml_fp32_to_bf16_row(fa.data(), ha.data(), static_cast<int64_t>(fa.size()));
    ggml_fp32_to_bf16_row(fb.data(), hb.data(), static_cast<int64_t>(fb.size()));
    ggml_backend_tensor_set(a, ha.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, hb.data(), 0, ggml_nbytes(b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<float> got(M * N);
    ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));

    int64_t mismatches = 0;
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            float want = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                want += fa[i * K + k] * fb[j * K + k];
            }
            if (got[j * M + i] != want) {
                if (mismatches < 8) {
                    printf("  mismatch at row %lld col %lld: got %g want %g\n", (long long)i,
                           (long long)j, got[j * M + i], want);
                }
                ++mismatches;
            }
        }
    }
    if (mismatches != 0) {
        printf("  %lld / %lld elements mismatched\n", (long long)mismatches, (long long)(M * N));
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

    // On aie2 the tile selector gives M=N=64 a 16x16 C tile per core (k = 256 at most), so K picks
    // the number of accumulating calls. The 32-wide cases keep the regular 4x4-expansion path
    // covered alongside the 16x16 one.
    struct {
        int64_t M, N, K;
        const char * name;
    } cases[] = {
        // single K tile: C starts at zero, so this passes even with misplaced accumulators
        {64, 64, 256, "16x16 tile, 1 K tile"},
        // the 16x16 C tile, accumulated across K tiles
        {64, 64, 512, "16x16 tile, 2 K tiles"},
        {64, 64, 1024, "16x16 tile, 4 K tiles"},
        // wider tiles, which stay on the 4x4-expansion path
        {128, 64, 512, "32x16 tile, 2 K tiles"},
        {64, 128, 512, "16x32 tile, 2 K tiles"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.M, c.N, c.K);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT bf16 %-22s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    ggml_backend_free(backend);

    if (any_fail) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
