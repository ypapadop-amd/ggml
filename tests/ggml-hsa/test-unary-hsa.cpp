// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Device tests for the element-wise unary ops: the vectorized SQR, ABS, NEG, SGN and STEP.
// Inputs deliberately straddle zero and include exact zeros, so the sign-dependent ops are
// checked on all their branches, and one shape is narrower than the 16-element f32 vector so
// the scalar tail runs with vend == 0.

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

enum class op_kind { sqr, abs, neg, sgn, step };

float reference(op_kind kind, float x) {
    switch (kind) {
        case op_kind::sqr:
            return x * x;
        case op_kind::abs:
            return std::fabs(x);
        case op_kind::neg:
            return -x;
        case op_kind::sgn:
            return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        case op_kind::step:
            return (x > 0.0f) ? 1.0f : 0.0f;
    }
    return 0.0f;
}

bool run_case(ggml_backend_t backend, op_kind kind, int64_t ne0, int64_t ne1, const char * name) {
    const int64_t n = ne0 * ne1;

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_set_name(src, "src");

    ggml_tensor * dst = nullptr;
    switch (kind) {
        case op_kind::sqr:
            dst = ggml_sqr(ctx.get(), src);
            break;
        case op_kind::abs:
            dst = ggml_abs(ctx.get(), src);
            break;
        case op_kind::neg:
            dst = ggml_neg(ctx.get(), src);
            break;
        case op_kind::sgn:
            dst = ggml_sgn(ctx.get(), src);
            break;
        case op_kind::step:
            dst = ggml_step(ctx.get(), src);
            break;
    }
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-22s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-22s: graph allocation failed\n", name);
        return false;
    }

    // Straddles zero and hits exact zero every 21 elements.
    std::vector<float> src_host(n);
    for (int64_t i = 0; i < n; ++i) {
        src_host[i] = (static_cast<float>(i % 21) - 10.0f) * 0.5f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    const float tol = 1e-5f;
    for (int64_t i = 0; i < n; ++i) {
        const float want = reference(kind, src_host[i]);
        const float got = dst_host[i];
        if (std::fabs(got - want) > tol) {
            printf("  %-22s: mismatch at %lld (x=%g) got %g want %g\n", name, (long long)i,
                   src_host[i], got, want);
            return false;
        }
    }
    return true;
}

// Bit-exact ABS/NEG/SGN/STEP check over the signed zeros, infinities and NaNs that the value-based
// comparison above cannot distinguish (-0.0 == +0.0 numerically, and NaN fails any tolerance
// test). ABS must match std::fabs and NEG must match -x down to the sign bit; SGN and STEP are
// built from aie::lt/aie::gt plus aie::select, whose treatment of NaN (every comparison false)
// and of -0.0 (not greater than, not less than +0.0) has to agree with the scalar tail, so they
// are checked here too rather than only through the tolerance-based comparison.
//
// The element count selects which code path runs: an f32 tile is a whole 16-lane vector only when
// the count is a multiple of 16, otherwise the tile shrinks below the vector width and the kernel
// runs entirely scalar. So 32 covers the vector path and 19 the scalar one; both must agree with
// the host.
bool run_sign_case(ggml_backend_t backend, op_kind kind, int64_t ne0, const char * name) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ne0);
    ggml_set_name(src, "src");
    ggml_tensor * dst = nullptr;
    switch (kind) {
        case op_kind::abs:
            dst = ggml_abs(ctx.get(), src);
            break;
        case op_kind::neg:
            dst = ggml_neg(ctx.get(), src);
            break;
        case op_kind::sgn:
            dst = ggml_sgn(ctx.get(), src);
            break;
        case op_kind::step:
            dst = ggml_step(ctx.get(), src);
            break;
        default:
            printf("  %-22s: unsupported op for sign case\n", name);
            return false;
    }
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-22s: op not supported\n", name);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-22s: graph allocation failed\n", name);
        return false;
    }

    // Leading entries are the interesting bit patterns; the rest are ordinary values so the whole
    // tile is covered.
    const float specials[] = {
        +0.0f,
        -0.0f,
        -1.5f,
        2.5f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::quiet_NaN(),
    };
    const auto n_specials = static_cast<int64_t>(sizeof(specials) / sizeof(specials[0]));

    std::vector<float> src_host(ne0);
    for (int64_t i = 0; i < ne0; ++i) {
        src_host[i] = (i < n_specials) ? specials[i] : (static_cast<float>(i % 21) - 10.0f) * 0.5f;
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(ne0);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    for (int64_t i = 0; i < ne0; ++i) {
        const float x = src_host[i];
        float want = 0.0f;
        switch (kind) {
            case op_kind::abs:
                want = std::fabs(x);
                break;
            case op_kind::neg:
                want = -x;
                break;
            // Matches the kernel's scalar tail exactly, including NaN (neither comparison
            // holds, so 0) and -0.0 (likewise 0, with a positive zero's bit pattern).
            case op_kind::sgn:
                want = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
                break;
            case op_kind::step:
                want = (x > 0.0f) ? 1.0f : 0.0f;
                break;
            default:
                break;
        }
        uint32_t got_bits = 0;
        uint32_t want_bits = 0;
        std::memcpy(&got_bits, &dst_host[i], sizeof(got_bits));
        std::memcpy(&want_bits, &want, sizeof(want_bits));
        if (got_bits != want_bits) {
            uint32_t src_bits = 0;
            std::memcpy(&src_bits, &src_host[i], sizeof(src_bits));
            printf("  %-22s: bit mismatch at %lld (x=0x%08x) got 0x%08x want 0x%08x\n", name,
                   (long long)i, src_bits, got_bits, want_bits);
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        op_kind kind;
        int64_t ne0, ne1;
        const char * name;
    } cases[] = {
        {op_kind::sqr, 256, 4, "sqr 2d"},
        {op_kind::abs, 256, 4, "abs 2d"},
        {op_kind::neg, 256, 4, "neg 2d"},
        {op_kind::sgn, 256, 4, "sgn 2d"},
        {op_kind::step, 256, 4, "step 2d"},
        {op_kind::sqr, 3072, 1, "sqr gpt2 mlp"},
        {op_kind::abs, 3072, 1, "abs gpt2 mlp"},
        {op_kind::neg, 3072, 1, "neg gpt2 mlp"},
        {op_kind::sgn, 3072, 1, "sgn gpt2 mlp"},
        {op_kind::step, 3072, 1, "step gpt2 mlp"},
        // A narrow 2D shape. Note this does NOT reach the scalar path: 10*8 == 80 is a multiple
        // of 16, so the tile is a full vector. The signbit cases below cover the scalar path.
        {op_kind::sqr, 10, 8, "sqr narrow"},
        {op_kind::abs, 10, 8, "abs narrow"},
        {op_kind::neg, 10, 8, "neg narrow"},
        {op_kind::sgn, 10, 8, "sgn narrow"},
        {op_kind::step, 10, 8, "step narrow"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        const bool ok = run_case(backend, c.kind, c.ne0, c.ne1, c.name);
        printf("UNARY %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    // Bit-exact sign handling. 32 elements tile to a full 16-lane vector (vector path); 19 does
    // not, so the kernel runs entirely scalar. Both paths must match the host bit for bit.
    struct {
        op_kind kind;
        int64_t ne0;
        const char * name;
    } sign_cases[] = {
        {op_kind::abs, 32, "abs signbit vector"}, {op_kind::neg, 32, "neg signbit vector"},
        {op_kind::sgn, 32, "sgn signbit vector"}, {op_kind::step, 32, "step signbit vector"},
        {op_kind::abs, 19, "abs signbit scalar"}, {op_kind::neg, 19, "neg signbit scalar"},
        {op_kind::sgn, 19, "sgn signbit scalar"}, {op_kind::step, 19, "step signbit scalar"},
    };
    for (const auto & c : sign_cases) {
        const bool ok = run_sign_case(backend, c.kind, c.ne0, c.name);
        printf("UNARY %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
