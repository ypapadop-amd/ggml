// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Device tests for CLAMP. The result is selected, never computed, so it is compared bit for bit
// with the CPU backend's MAX(MIN(x, max), min), evaluated here on the host.
//
// The backend flattens the tensor, and clamp.py streams it in tiles of the largest divisor of the
// count that fits L1, split across up to one worker per column for large counts. A tile runs the
// vector loop (with a minimum-trip-count copy from 4 vectors up) and a scalar tail for the rest.
// The shapes below pick each of those. Inputs cover the full exponent range, signed zeros,
// infinities, subnormals, values equal to the bounds and +NaN. A NaN with the sign bit set is left
// out: the kernel clamps it to min, where the CPU gives max (see clamp.cc).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

float cpu_clamp(float x, float lo, float hi) {
    const float m = x < hi ? x : hi;
    return m > lo ? m : lo;
}

uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

bool run_case(ggml_backend_t backend, int64_t n, float lo, float hi, const char * name) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * dst = ggml_clamp(ctx.get(), src, lo, hi);

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-26s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-26s: graph allocation failed\n", name);
        return false;
    }

    // Random sign, exponent and mantissa (exponents 2^-17 .. 2^16, so about half the values fall
    // inside a +-0.5 range), with the special values spread through the run so they land in
    // different lanes and, for the longer cases, in different tiles.
    const float inf = std::numeric_limits<float>::infinity();
    const float specials[] = {0.0f,
                              -0.0f,
                              inf,
                              -inf,
                              lo,
                              hi,
                              std::nextafter(lo, -inf),
                              std::nextafter(hi, inf),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::denorm_min(),
                              -std::numeric_limits<float>::denorm_min()};
    const int64_t n_specials = static_cast<int64_t>(sizeof specials / sizeof specials[0]);
    std::vector<float> src_host(n);
    uint64_t rng = 0x243F6A8885A308D3ULL;
    for (int64_t i = 0; i < n; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t r = static_cast<uint32_t>(rng >> 32);
        const uint32_t bits = (r & 0x807FFFFFu) | ((110u + (r >> 24) % 34u) << 23);
        std::memcpy(&src_host[i], &bits, sizeof bits);
    }
    const int64_t stride = n / n_specials > 0 ? n / n_specials : 1;
    for (int64_t s = 0; s < n_specials && s * stride < n; ++s) {
        src_host[s * stride + (s % 3)] = specials[s];
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-26s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    int64_t wrong = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (bits_of(dst_host[i]) != bits_of(cpu_clamp(src_host[i], lo, hi))) {
            if (first < 0) {
                first = i;
            }
            ++wrong;
        }
    }

    printf("  %-26s: %lld of %lld differ", name, (long long)wrong, (long long)n);
    if (first >= 0) {
        printf(" (first at %lld: x=%.9g got %.9g want %.9g)", (long long)first, src_host[first],
               dst_host[first], cpu_clamp(src_host[first], lo, hi));
    }
    printf("\n");
    return wrong == 0;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    const float inf = std::numeric_limits<float>::infinity();
    struct {
        int64_t n;
        float lo, hi;
        const char * name;
    } cases[] = {
        // Whole-vector tiles of 2048: the vector loop's minimum-trip-count copy.
        {8192, -0.5f, 0.5f, "vector"},
        // 48 = 3 vectors in one tile: the vector loop's short copy.
        {48, -0.5f, 0.5f, "vector short"},
        // One tile of 40: 2 vectors and an 8-element tail.
        {40, -0.5f, 0.5f, "vector + tail"},
        // 3 tiles of 683: not whole vectors, so tiles start unaligned, each with a tail.
        {2049, -0.5f, 0.5f, "unaligned tiles"},
        // One tile of 19, all tail; and a prime above the L1 cap, tiles of 1.
        {19, -0.5f, 0.5f, "scalar"},
        {4099, -0.5f, 0.5f, "scalar tiles of 1"},
        // Fanned out across workers: 4 bands of 128 tiles, and 2 uneven bands (129 + 128).
        {1048576, -0.5f, 0.5f, "4 workers"},
        {526336, -0.5f, 0.5f, "2 workers uneven"},
        // min > max: min everywhere, on both paths.
        {8192, 0.5f, -0.5f, "vector min>max"},
        {19, 0.5f, -0.5f, "scalar min>max"},
        // One-sided clamps with a zero bound. A +inf bound does not build (the design's MLIR
        // cannot carry it as a constant), so the top is the largest finite float instead.
        {8192, 0.0f, std::numeric_limits<float>::max(), "vector [0, max]"},
        {19, 0.0f, std::numeric_limits<float>::max(), "scalar [0, max]"},
        {8192, -inf, -0.0f, "vector [-inf, -0]"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        const bool ok = run_case(backend, c.n, c.lo, c.hi, c.name);
        printf("CLAMP %-26s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
