// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// The AIE runtime gives each distinct PDI a queue runs one of the 32 compute units of the queue's
// hardware context and never frees it, so a 33rd PDI would suspend the queue. The backend replaces
// its queue before that happens (ggml_hsa_reserve_pdi). This test runs one graph of more distinct
// kernels than two queues can hold -- NEG at as many element counts, each its own PDI -- on one
// backend, several times, and checks every result bit for bit: the limit is crossed mid-graph with
// packets still pending, and again on every recompute.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

// More than twice the 32 PDIs one queue can hold.
constexpr int kKernels = 70;
constexpr int kComputes = 2;

// Element count of kernel i; distinct counts give distinct kernels (unary ops are flattened).
int64_t numel(int i) { return 64 * (i + 1); }

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    const std::size_t ctx_size =
        2 * kKernels * ggml_tensor_overhead() + ggml_graph_overhead_custom(4 * kKernels, false);
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    std::vector<ggml_tensor *> srcs(kKernels);
    std::vector<ggml_tensor *> dsts(kKernels);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 4 * kKernels, false);
    for (int i = 0; i < kKernels; ++i) {
        srcs[i] = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, numel(i));
        ggml_set_input(srcs[i]);
        dsts[i] = ggml_neg(ctx.get(), srcs[i]);
        ggml_set_output(dsts[i]);
        if (!ggml_backend_supports_op(backend, dsts[i])) {
            printf("NEG of %lld elements not supported; skipping.\n", (long long)numel(i));
            ggml_backend_free(backend);
            return 0;
        }
        ggml_build_forward_expand(gf, dsts[i]);
    }

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("graph allocation failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    bool ok = true;
    for (int c = 0; c < kComputes; ++c) {
        // A different input per compute, so a stale result from the previous one cannot pass.
        // Kept on the host: the allocator may run NEG in place over its input.
        std::vector<std::vector<float>> ins(kKernels);
        for (int i = 0; i < kKernels; ++i) {
            std::vector<float> & in = ins[i];
            in.resize(numel(i));
            for (int64_t e = 0; e < numel(i); ++e) {
                in[e] = static_cast<float>((e * 7 + i * 3 + c) % 19 - 9);
            }
            ggml_backend_tensor_set(srcs[i], in.data(), 0, ggml_nbytes(srcs[i]));
        }

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("compute %d: graph compute failed\n", c);
            ok = false;
            break;
        }

        int failed_kernels = 0;
        for (int i = 0; i < kKernels; ++i) {
            const std::vector<float> & in = ins[i];
            std::vector<float> out(numel(i));
            ggml_backend_tensor_get(dsts[i], out.data(), 0, ggml_nbytes(dsts[i]));
            for (int64_t e = 0; e < numel(i); ++e) {
                const float want = -in[e];
                if (std::memcmp(&want, &out[e], sizeof(float)) != 0) {
                    printf("compute %d: kernel %d (%lld elements): element %lld got %g want %g\n",
                           c, i, (long long)numel(i), (long long)e, out[e], want);
                    ++failed_kernels;
                    break;
                }
            }
        }
        if (failed_kernels != 0) {
            printf("compute %d: %d / %d kernels wrong\n", c, failed_kernels, kKernels);
            ok = false;
            break;
        }
    }
    ggml_backend_free(backend);

    printf("%d distinct kernels x %d computes on one backend: %s\n", kKernels, kComputes,
           ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
