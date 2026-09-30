// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for f32 MUL_MAT shapes that take the padded bf16 GEMM path. The operands are
// converted and zero-padded to the tile multiples, yet the result must land in the dense [M, N]
// destination. Each shape targets one part of the dense-destination C path
// (src/ggml-hsa/kernels/iron_kernels/gemm_c_plan.py).
//
// Inputs are small integers, so every product and partial sum is exact in bf16 and f32. The
// device must therefore match the host reference bit for bit.
//
// With `cast`, the result feeds an f32->bf16 cast (followed by a cast back to f32 so the bf16
// tensor is not a graph output). The scheduler's graph_optimize then folds the cast into the
// GEMM when M is even, and the reference rounds with ggml's own f32->bf16 conversion.
//
// Cast cases re-allocate the graph between dispatches (see the comment in run_case).
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

bool run_case(ggml_backend_t hsa, ggml_backend_t cpu, int64_t M, int64_t N, int64_t K, bool cast) {
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
    ggml_tensor * out = c;
    if (cast) {
        out = ggml_cast(ctx.get(), ggml_cast(ctx.get(), c, GGML_TYPE_BF16), GGML_TYPE_F32);
    }
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);

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
            want[j * M + i] = cast ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(acc)) : acc;
        }
    }

    bool ok = true;
    for (int d = 0; d < kDispatches; ++d) {
        // Recomputing an already-allocated graph whose MUL_MAT was folded to bf16 by graph_optimize
        // returns stale results from the second dispatch (pre-existing, reproduces on 2053fe89), so
        // cast cases re-allocate per dispatch. The GEMM kernel and its device state persist either
        // way, so cross-dispatch ping/pong parity is still exercised.
        if (cast && d > 0) {
            ggml_backend_sched_reset(sched.get());
            ggml_backend_sched_set_tensor_backend(sched.get(), a, hsa);
            ggml_backend_sched_set_tensor_backend(sched.get(), b, hsa);
            if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
                printf("  dispatch %d: graph allocation failed\n", d);
                return false;
            }
            ggml_backend_tensor_set(a, fa.data(), 0, ggml_nbytes(a));
            ggml_backend_tensor_set(b, fb.data(), 0, ggml_nbytes(b));
        }
        if (ggml_backend_sched_graph_compute(sched.get(), gf) != GGML_STATUS_SUCCESS) {
            printf("  dispatch %d: graph compute failed\n", d);
            return false;
        }
        if (ggml_backend_sched_get_tensor_backend(sched.get(), c) != hsa) {
            printf("  MUL_MAT did not run on the HSA backend\n");
            return false;
        }
        // graph_optimize retypes the MUL_MAT to bf16 when it folds the cast, except for an odd M:
        // the DMA cannot write bf16 columns that start mid-word, so the MUL_MAT must stay f32.
        if (cast && M % 2 == 0 && c->type != GGML_TYPE_BF16) {
            printf("  dispatch %d: bf16 fold did not happen (MUL_MAT type is %s)\n", d,
                   ggml_type_name(c->type));
            return false;
        }
        if (cast && M % 2 != 0 && c->type != GGML_TYPE_F32) {
            printf("  dispatch %d: odd-M MUL_MAT was folded (MUL_MAT type is %s)\n", d,
                   ggml_type_name(c->type));
            return false;
        }
        std::vector<float> got(M * N);
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
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
    ggml_backend_t hsa = ggml_backend_hsa_init(0);
    if (hsa == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        printf("CPU backend init failed\n");
        ggml_backend_free(hsa);
        return 1;
    }

    struct {
        int64_t M, N, K;
        bool cast;
        const char * name;
    } cases[] = {
        {512, 512, 512, false, "aligned, no clip"},
        {500, 500, 784, false, "MNIST fc1: partial core + partial column"},
        {10, 500, 500, false, "two column groups, partial r-group"},
        {10, 500, 784, false, "cores 1-3 all padding"},
        {392000, 8, 9, false, "conv1: 7 idle AIE columns, 490 row blocks"},
        {98000, 16, 72, false, "conv2: idle columns + row clip, 1021 row blocks"},
        {512, 512, 512, true, "aligned, bf16 destination"},
        {500, 500, 784, true, "fc1, bf16 destination"},
        {10, 500, 500, true, "two column groups, bf16 destination"},
        {10, 500, 784, true, "cores 1-3 padding, bf16 destination"},
        {392000, 8, 9, true, "conv1, bf16 destination"},
        {98000, 16, 72, true, "conv2, bf16 destination"},
        {499, 500, 784, true, "odd M: bf16 fold refused, stays f32"},
    };

    int failed = 0;
    for (const auto & t : cases) {
        const bool ok = run_case(hsa, cpu, t.M, t.N, t.K, t.cast);
        printf("MUL_MAT %lld,%lld,%lld%s %-45s: %s\n", (long long)t.M, (long long)t.N,
               (long long)t.K, t.cast ? " +cast" : "", t.name, ok ? "PASSED" : "FAILED");
        failed += ok ? 0 : 1;
    }
    ggml_backend_free(cpu);
    ggml_backend_free(hsa);

    if (failed != 0) {
        printf("%d FAILED\n", failed);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
