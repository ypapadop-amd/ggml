// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for f32 MUL_MAT shapes that take the padded bf16 GEMM path. The operands are
// converted and zero-padded to the tile multiples, yet the result must land in the dense [M, N]
// destination. Each shape targets one part of the dense-destination C path
// (src/ggml-hsa/kernels/iron_kernels/gemm_c_plan.py).
//
// Inputs are small integers, so every product and partial sum is exact in bf16 and f32. The
// device must therefore match the host reference bit for bit.
//
// Every case runs several dispatches, because the GEMM's ping/pong parity must survive across
// them. Every case also checks that MUL_MAT ran on the HSA backend: a CPU fallback would pass
// silently.
//
// On failure the first mismatches are printed as (row, col, got, want), so a wrong DMA pattern
// shows up as a recognisable block rather than a bare FAILED.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

constexpr int kDispatches = 3;
constexpr int kMaxReported = 8;

bool run_case(ggml_backend_t hsa, ggml_backend_t cpu, int64_t M, int64_t N, int64_t K) {
    const std::size_t ctx_size = 8 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};
    if (!ctx) {
        printf("  ggml_init failed\n");
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_input(a);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_set_input(b);
    ggml_tensor * c = ggml_mul_mat(ctx.get(), a, b); // [M, N] f32
    ggml_set_output(c);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);

    ggml_backend_t backends[2] = {hsa, cpu};
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched{
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true),
        ggml_backend_sched_free};
    // Buffer-less inputs default to the last (CPU) backend, which would pull MUL_MAT with them.
    // Pin only the operands; MUL_MAT is still placed by the scheduler, subject to supports_op.
    ggml_backend_sched_set_tensor_backend(sched.get(), a, hsa);
    ggml_backend_sched_set_tensor_backend(sched.get(), b, hsa);
    if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        printf("  graph allocation failed\n");
        return false;
    }

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
    ggml_backend_tensor_set(a, fa.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, fb.data(), 0, ggml_nbytes(b));

    std::vector<float> want(M * N);
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += fa[i * K + k] * fb[j * K + k];
            }
            want[j * M + i] = acc;
        }
    }

    bool ok = true;
    for (int d = 0; d < kDispatches; ++d) {
        if (ggml_backend_sched_graph_compute(sched.get(), gf) != GGML_STATUS_SUCCESS) {
            printf("  dispatch %d: graph compute failed\n", d);
            return false;
        }
        if (ggml_backend_sched_get_tensor_backend(sched.get(), c) != hsa) {
            printf("  MUL_MAT did not run on the HSA backend\n");
            return false;
        }
        std::vector<float> got(M * N);
        ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));
        int64_t mismatches = 0;
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t i = 0; i < M; ++i) {
                const float g = got[j * M + i];
                const float w = want[j * M + i];
                if (std::memcmp(&g, &w, sizeof(float)) != 0) {
                    if (mismatches < kMaxReported) {
                        printf("  dispatch %d: mismatch at row %lld col %lld: got %g want %g\n", d,
                               (long long)i, (long long)j, g, w);
                    }
                    ++mismatches;
                }
            }
        }
        if (mismatches != 0) {
            printf("  dispatch %d: %lld / %lld elements mismatched\n", d, (long long)mismatches,
                   (long long)(M * N));
            ok = false;
        }
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_t probe = ggml_backend_hsa_init(0);
    if (probe == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }
    ggml_backend_free(probe);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        printf("CPU backend init failed\n");
        return 1;
    }

    struct {
        int64_t M, N, K;
        const char * name;
    } cases[] = {
        {512, 512, 512, "aligned, no clip"},
        {500, 500, 784, "MNIST fc1: partial core + partial column"},
        {10, 500, 500, "two column groups, partial r-group"},
        {10, 500, 784, "cores 1-3 all padding"},
        {392000, 8, 9, "conv1: 7 idle AIE columns, 490 row blocks"},
        {98000, 16, 72, "conv2: idle columns + row clip, 1021 row blocks"},
        {499, 500, 784, "odd M: partial core with an odd row count"},
        {1500, 2436, 64, "47 row blocks, straddling column per row block"},
        {1000, 260, 512, "3 column groups, straddling column 0"},
        {1000, 2000, 1000, "8 column groups, straddling column per row block"},
        {300, 16384, 64, "row-clipped last row block in 11 shim chunks"},
        {1500, 12000, 64, "uniform column issued per row block"},
    };

    int failed = 0;
    for (const auto & t : cases) {
        // A fresh HSA backend per case: freeing it evicts the case's cached kernels. Kept for the
        // whole run they exhaust the process's NPU hardware contexts, and the 14th case failed
        // with HSA_STATUS_ERROR_OUT_OF_RESOURCES whatever its shape.
        ggml_backend_t hsa = ggml_backend_hsa_init(0);
        const bool ok = hsa != nullptr && run_case(hsa, cpu, t.M, t.N, t.K);
        if (hsa == nullptr) {
            printf("  HSA backend init failed\n");
        } else {
            ggml_backend_free(hsa);
        }
        printf("MUL_MAT %lld,%lld,%lld %-50s: %s\n", (long long)t.M, (long long)t.N,
               (long long)t.K, t.name, ok ? "PASSED" : "FAILED");
        failed += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);

    if (failed != 0) {
        printf("%d FAILED\n", failed);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
