// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Device tests for the element-wise unary ops: the vectorized SQR, ABS and NEG, plus the
// still-scalar SGN and STEP. Inputs deliberately straddle zero and include exact zeros, so
// the sign-dependent ops are checked on all their branches.
//
// One shape has an element count that is not a multiple of the 16-element f32 vector, which
// is what drives the kernel down the scalar tail. Note it is the *total* count that matters,
// not ne0: the backend flattens unary tensors to (nelements, 1, 1, 1) before dispatch (see
// ggml_hsa_flatten_tensor), so a 10x8 tensor reaches the kernel as one 80-element run and
// takes the vector path.

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
        case op_kind::sqr: return x * x;
        case op_kind::abs: return std::fabs(x);
        case op_kind::neg: return -x;
        case op_kind::sgn: return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
        case op_kind::step: return (x > 0.0f) ? 1.0f : 0.0f;
    }
    return 0.0f;
}

// Maps a float onto a monotonically increasing integer, so the difference between two keys
// is a true distance in representable floats. Subtracting the raw int32 bit patterns instead
// overflows whenever the two values straddle zero -- reference(neg, +0.0f) is -0.0f, i.e.
// INT32_MIN, so a device result of +0.0f would give 0 - INT32_MIN, wrapping to a negative
// "ulp" that never exceeds the bound and reports the mismatch as 0 ulp.
int64_t ulp_key(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return (b & 0x80000000u) != 0 ? static_cast<int64_t>(0x80000000u) - static_cast<int64_t>(b)
                                  : static_cast<int64_t>(b);
}

// Distance in representable floats. +0.0 and -0.0 map to the same key and so compare equal;
// their sign is covered separately by the bit-exact signbit cases below.
int64_t ulp_distance(float want, float got) {
    const int64_t d = ulp_key(got) - ulp_key(want);
    return d < 0 ? -d : d;
}

bool run_case(ggml_backend_t backend, op_kind kind, int64_t ne0, int64_t ne1,
              const char * name) {
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
        case op_kind::sqr: dst = ggml_sqr(ctx.get(), src); break;
        case op_kind::abs: dst = ggml_abs(ctx.get(), src); break;
        case op_kind::neg: dst = ggml_neg(ctx.get(), src); break;
        case op_kind::sgn: dst = ggml_sgn(ctx.get(), src); break;
        case op_kind::step: dst = ggml_step(ctx.get(), src); break;
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

    // Random sign, exponent and mantissa. Covering the full 23-bit mantissa is what makes
    // this able to see the aie2 fp32 emulation drift: a low-entropy input (a handful of
    // distinct mantissas scaled by powers of two) measures 0 ulp on every op and hides it.
    // Element 0 is forced to exactly zero so the sign-dependent ops keep that branch.
    std::vector<float> src_host(n);
    uint64_t rng = 0x243F6A8885A308D3ULL;
    for (int64_t i = 0; i < n; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t r = static_cast<uint32_t>(rng >> 32);
        const uint32_t bits = (r & 0x807FFFFFu) | ((110u + (r >> 24) % 34u) << 23);
        std::memcpy(&src_host[i], &bits, 4);
    }
    src_host[0] = 0.0f;
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // None of these ops can cancel, so the distance in representable floats (ulp) between
    // the device result and the CPU reference is a direct measure of arithmetic drift.
    // Measured on aie2 (NPU1): ABS/NEG/SGN/STEP are exact -- they are sign and comparison
    // work, not arithmetic -- and SQR reaches 3 ulp because aie::mul on fp32 lowers to the
    // emulated bf16 triple-product (see the aie2 shim in aie_kernel_math.h), which costs
    // about 1 ulp, doubled by squaring. Bounds are per-op so a regression on the exact ops
    // cannot hide behind SQR's allowance.
    const int64_t max_ulp = (kind == op_kind::sqr) ? 4 : 0;

    int64_t worst_ulp = 0;
    int64_t worst_i = -1;
    for (int64_t i = 0; i < n; ++i) {
        const float want = reference(kind, src_host[i]);
        const float got = dst_host[i];
        const int64_t ulp = ulp_distance(want, got);
        if (ulp > worst_ulp) { worst_ulp = ulp; worst_i = i; }
    }

    const bool ok = worst_ulp <= max_ulp;
    printf("  %-22s: worst %lld ulp (allowed %lld)", name, (long long)worst_ulp,
           (long long)max_ulp);
    if (worst_i >= 0) {
        printf(" at x=%.9g (got %.9g want %.9g)", src_host[worst_i], dst_host[worst_i],
               reference(kind, src_host[worst_i]));
    }
    printf("\n");
    return ok;
}

// Bit-exact ABS/NEG check over the signed zeros, infinities and NaNs that the value-based
// comparison above cannot distinguish (-0.0 == +0.0 numerically, and NaN fails any tolerance
// test). ABS must match std::fabs and NEG must match -x down to the sign bit.
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
    ggml_tensor * dst = (kind == op_kind::abs) ? ggml_abs(ctx.get(), src) : ggml_neg(ctx.get(), src);
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
        +0.0f, -0.0f, -1.5f, 2.5f, std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
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
        const float want = (kind == op_kind::abs) ? std::fabs(src_host[i]) : -src_host[i];
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
        // of 16, so the tile is a full vector.
        {op_kind::sqr, 10, 8, "sqr narrow"},
        {op_kind::abs, 10, 8, "abs narrow"},
        {op_kind::neg, 10, 8, "neg narrow"},
        {op_kind::sgn, 10, 8, "sgn narrow"},
        {op_kind::step, 10, 8, "step narrow"},
        // 10*3 == 30 elements once flattened, not a multiple of the 16-element f32 vector,
        // so the tile drops below V and the whole run goes through the scalar tail
        // (vend == 0) -- the case that rules out AIE_LOOP_MIN_ITERATION_COUNT. The signbit
        // cases below cover the scalar path for ABS/NEG bit patterns as well.
        {op_kind::sqr, 10, 3, "sqr tail"},
        {op_kind::abs, 10, 3, "abs tail"},
        {op_kind::neg, 10, 3, "neg tail"},
        {op_kind::sgn, 10, 3, "sgn tail"},
        {op_kind::step, 10, 3, "step tail"},
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
        {op_kind::abs, 32, "abs signbit vector"},
        {op_kind::neg, 32, "neg signbit vector"},
        {op_kind::abs, 19, "abs signbit scalar"},
        {op_kind::neg, 19, "neg signbit scalar"},
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
