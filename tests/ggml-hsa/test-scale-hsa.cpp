// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the GGML_OP_SCALE kernel on the HSA backend: out = in * s + b.
//
// The backend flattens SCALE to one run of numel elements, and the design picks its tile (the
// largest divisor of numel that fits L1) and its worker count from numel, so the cases below are
// chosen by element count: tiles of whole 16-lane vectors, tiles that are not (unaligned, with a
// scalar tail), tiles narrower than a vector (tail only), tiles of one element, and counts large
// enough that the design splits the tiles over several workers.
//
// Inputs are random full-mantissa floats, since low-entropy inputs hide the aie2 emulated fp32
// multiply's drift. The reference is computed in double. aie2 has no native fp32 multiply or
// add, so the result is gated at 2^-21 of max(|x * s|, |b|): the multiply is within 1 ulp of its
// product, the add within 1 ulp of its larger operand, which bounds their sum below 2^-22.
//
// With "perf N REPS" it times REPS computes of one SCALE node of N elements instead (us/run).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

struct graph {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf{
        nullptr, ggml_backend_buffer_free};
    ggml_cgraph * gf = nullptr;
    ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;
};

// Builds dst = src * s + b over n f32 elements and allocates both on the backend. Returns false if
// the backend does not take the op or the allocation fails.
bool build(ggml_backend_t backend, int64_t n, float s, float b, graph & g) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    g.ctx.reset(ggml_init(params));
    g.src = ggml_new_tensor_1d(g.ctx.get(), GGML_TYPE_F32, n);
    ggml_set_name(g.src, "src");
    g.dst = ggml_scale_bias(g.ctx.get(), g.src, s, b);
    ggml_set_name(g.dst, "dst");
    if (!ggml_backend_supports_op(backend, g.dst)) {
        return false;
    }
    g.gf = ggml_new_graph(g.ctx.get());
    ggml_build_forward_expand(g.gf, g.dst);
    // src and dst get their own storage: a graph allocator would run the op in place, which
    // the poisoned output below would then overwrite the input of.
    g.buf.reset(ggml_backend_alloc_ctx_tensors(g.ctx.get(), backend));
    return g.buf != nullptr;
}

std::vector<float> random_inputs(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> mant(1.0f, 2.0f);
    std::uniform_int_distribution<int> expo(-8, 8);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) {
        const float m = std::ldexp(mant(rng), expo(rng));
        v[i] = (rng() & 1) != 0 ? -m : m;
    }
    return v;
}

bool run_case(ggml_backend_t backend, int64_t n, float s, float b, uint32_t seed) {
    graph g;
    if (!build(backend, n, s, b, g)) {
        printf("  n=%-7lld s=%g b=%g: not supported or allocation failed\n", (long long)n, s, b);
        return false;
    }

    const std::vector<float> in = random_inputs(n, seed);
    ggml_backend_tensor_set(g.src, in.data(), 0, ggml_nbytes(g.src));

    // Poison the output so an element the kernel never writes cannot pass by chance.
    const std::vector<float> poison(n, NAN);
    ggml_backend_tensor_set(g.dst, poison.data(), 0, ggml_nbytes(g.dst));

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        printf("  n=%-7lld s=%g b=%g: graph compute failed\n", (long long)n, s, b);
        return false;
    }

    std::vector<float> out(n);
    ggml_backend_tensor_get(g.dst, out.data(), 0, ggml_nbytes(g.dst));

    constexpr double tol = 1.0 / (1 << 21);
    double worst = 0.0;
    int64_t bad = 0;
    int64_t first_bad = -1;
    for (int64_t i = 0; i < n; ++i) {
        const double prod = static_cast<double>(in[i]) * s;
        const double want = prod + b;
        const double mag = std::max(std::fabs(prod), std::fabs(static_cast<double>(b)));
        const double err = mag > 0.0 ? std::fabs(out[i] - want) / mag : std::fabs(out[i] - want);
        if (!(err <= tol)) { // NaN-safe: a NaN error fails
            if (first_bad < 0) {
                first_bad = i;
            }
            ++bad;
        }
        if (err > worst || std::isnan(err)) {
            worst = err;
        }
    }
    if (bad > 0) {
        printf("  n=%-7lld s=%g b=%g: %lld bad, first at %lld (got %g want %g), worst %.3g\n",
               (long long)n, s, b, (long long)bad, (long long)first_bad, out[first_bad],
               static_cast<double>(in[first_bad]) * s + b, worst);
        return false;
    }
    printf("  n=%-7lld s=%g b=%g: worst %.3g of max(|x*s|,|b|)\n", (long long)n, s, b, worst);
    return true;
}

int run_perf(ggml_backend_t backend, int64_t n, int reps) {
    graph g;
    if (!build(backend, n, 0.37f, -1.25f, g)) {
        printf("SCALE n=%lld: not supported\n", (long long)n);
        return 1;
    }
    const std::vector<float> in = random_inputs(n, 1);
    ggml_backend_tensor_set(g.src, in.data(), 0, ggml_nbytes(g.src));
    for (int i = 0; i < 5; ++i) { // warm up (and JIT on the first)
        ggml_backend_graph_compute(backend, g.gf);
    }
    std::vector<double> us(reps);
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            printf("compute failed\n");
            return 1;
        }
        us[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count();
    }
    std::sort(us.begin(), us.end());
    printf("SCALE n=%lld reps=%d us/run min %.1f median %.1f max %.1f\n", (long long)n, reps,
           us.front(), us[reps / 2], us.back());
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    if (argc == 4 && std::string_view(argv[1]) == "perf") {
        const int rc = run_perf(backend, std::atoll(argv[2]), std::atoi(argv[3]));
        ggml_backend_free(backend);
        return rc;
    }

    struct {
        int64_t n;
        float s, b;
    } cases[] = {
        {16 * 1024, 2.0f, 0.0f},      // 8 tiles of 2048: whole vectors
        {100000, 0.37f, -1.25f},      // ggml's own test shape: 50 tiles of 2000
        {16 * 1000 * 7, -3.1f, 0.5f}, // 56 tiles of 2000
        {10, 0.37f, -1.25f},          // one tile of 10: narrower than a vector, tail only
        {24, 1.5f, 0.25f},            // one vector and a tail
        {1000, -0.75f, 3.0f},         // one unaligned tile: 62 vectors and a tail
        {1009, 0.37f, -1.25f},        // prime: one tile of 1009
        {6147, -0.75f, 3.0f},         // 9 unaligned tiles of 683
        {16, 1.0f, 0.0f},             // exactly one vector
        {4099 * 3, 1.5f, 0.25f},      // 4099 is prime: 4099 tiles of 3, tail only
        {4099, -3.1f, 0.5f},          // prime above the L1 cap: 4099 tiles of 1
        // Large enough to fan out over several workers: 4 equal bands; 2 bands of 129 and 128
        // tiles; 65537 (prime) tiles of 16 over 4 uneven bands; 600 unaligned tiles of 2047.
        {1 << 20, 0.37f, -1.25f},
        {2048 * 257, -3.1f, 0.5f},
        {16 * 65537, 1.5f, 0.25f},
        {2047 * 600, -0.75f, 3.0f},
    };

    bool any_fail = false;
    uint32_t seed = 1;
    for (const auto & c : cases) {
        const bool ok = run_case(backend, c.n, c.s, c.b, seed++);
        printf("SCALE n=%-7lld: %s\n", (long long)c.n, ok ? "PASSED" : "FAILED");
        any_fail = any_fail || !ok;
    }

    ggml_backend_free(backend);
    printf(any_fail ? "FAILURES\n" : "ALL PASSED\n");
    return any_fail ? 1 : 0;
}
