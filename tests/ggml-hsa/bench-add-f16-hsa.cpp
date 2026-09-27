// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Timing harness for a chain of f16 element-wise ADDs on the HSA backend.
//
// The device has no f16 kernels, so an f16 ADD runs as bf16 internally: each f16 source is
// converted into an internal buffer before the dispatch and the bf16 result is converted back into
// the f16 parent afterwards. The point of interest is the output conversion. On the host path it
// has to drain the queue after every node -- the host may not touch a buffer the device is still
// using -- which flushes the packet batch and serializes the chain. With the bf16 -> f16 transform
// kernel it runs on-queue instead, so a chain of N adds submits as one batch.
//
// A chain, not a single add, because a single-node graph drains at the end regardless: the cost of
// the extra drains is only visible when there is surrounding work they could have batched with.
//
// Reports min / median / max over the timed iterations rather than a single number, so a
// before/after comparison can be read against the run-to-run spread.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

constexpr int chain_length = 8;
constexpr int warmup_iterations = 20;
constexpr int timed_iterations = 200;

// Runs `chain_length` chained f16 adds over [d0, d1] tensors and reports the per-graph wall time.
// Returns false if the shape is not supported or a compute fails.
bool run_chain(ggml_backend_t backend, int64_t d0, int64_t d1) {
    const std::size_t tensor_count = chain_length + 2;
    const std::size_t ctx_size = tensor_count * ggml_tensor_overhead() +
                                 ggml_graph_overhead_custom(tensor_count, false);
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, d0, d1);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, d0, d1);
    ggml_set_name(b, "b");

    ggml_tensor * out = a;
    for (int i = 0; i < chain_length; ++i) {
        out = ggml_add(ctx.get(), out, b);
    }
    ggml_set_name(out, "out");

    if (!ggml_backend_supports_op(backend, out)) {
        printf("  [%5lld x %-4lld] op not supported\n", (long long)d0, (long long)d1);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), tensor_count, /*grads =*/false);
    ggml_build_forward_expand(gf, out);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  [%5lld x %-4lld] graph allocation failed\n", (long long)d0, (long long)d1);
        return false;
    }

    // Small magnitudes so the repeated adds stay well inside the f16 range.
    const int64_t n = d0 * d1;
    std::vector<uint16_t> bits(n);
    for (int64_t i = 0; i < n; ++i) {
        bits[i] = ggml_fp32_to_fp16(static_cast<float>(i % 17) * 0.25f);
    }
    ggml_backend_tensor_set(a, bits.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, bits.data(), 0, ggml_nbytes(b));

    // Warm up: the first compute pays the one-time JIT compile and kernel load.
    for (int i = 0; i < warmup_iterations; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  [%5lld x %-4lld] warm-up compute failed\n", (long long)d0, (long long)d1);
            return false;
        }
    }

    std::vector<double> samples_us;
    samples_us.reserve(timed_iterations);
    for (int i = 0; i < timed_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  [%5lld x %-4lld] compute failed\n", (long long)d0, (long long)d1);
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        samples_us.push_back(
            std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples_us.begin(), samples_us.end());
    const double min = samples_us.front();
    const double median = samples_us[samples_us.size() / 2];
    const double max = samples_us.back();
    printf("  [%5lld x %-4lld] %8lld elems  min %9.1f us  median %9.1f us  max %9.1f us  "
           "(%.2f us/add)\n",
           (long long)d0, (long long)d1, (long long)n, min, median, max,
           median / chain_length);
    return true;
}

// Times a bare HSA_CONVERT dispatch, so the add numbers above can be attributed. The f32 -> bf16
// direction is vectorized, the rest are scalar; comparing them says whether a direction's cost is
// the conversion arithmetic or the streaming around it.
bool run_convert(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type, const char * label,
                 int64_t d0, int64_t d1) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0, d1);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_convert(ctx.get(), src, dst_type);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-12s op not supported\n", label);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-12s graph allocation failed\n", label);
        return false;
    }

    for (int i = 0; i < warmup_iterations; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-12s warm-up compute failed\n", label);
            return false;
        }
    }

    std::vector<double> samples_us;
    samples_us.reserve(timed_iterations);
    for (int i = 0; i < timed_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  %-12s compute failed\n", label);
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        samples_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    std::sort(samples_us.begin(), samples_us.end());
    const double median = samples_us[samples_us.size() / 2];
    printf("  %-12s min %8.1f us  median %8.1f us  max %8.1f us  (%.1f ns/elem)\n", label,
           samples_us.front(), median, samples_us.back(),
           median * 1000.0 / static_cast<double>(d0 * d1));
    return true;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    printf("f16 ADD chain of %d, %d warm-up + %d timed iterations\n", chain_length,
           warmup_iterations, timed_iterations);

    struct {
        int64_t d0, d1;
    } shapes[] = {
        {1024, 1}, {4096, 1}, {1024, 16}, {4096, 16}, {4096, 64},
    };
    for (const auto & s : shapes) {
        run_chain(backend, s.d0, s.d1);
    }

    printf("bare HSA_CONVERT, 512 x 512 (262144 elements)\n");
    run_convert(backend, GGML_TYPE_F32, GGML_TYPE_BF16, "f32->bf16", 512, 512);
    run_convert(backend, GGML_TYPE_BF16, GGML_TYPE_F32, "bf16->f32", 512, 512);
    run_convert(backend, GGML_TYPE_F16, GGML_TYPE_BF16, "f16->bf16", 512, 512);
    run_convert(backend, GGML_TYPE_BF16, GGML_TYPE_F16, "bf16->f16", 512, 512);

    ggml_backend_free(backend);
    return 0;
}
