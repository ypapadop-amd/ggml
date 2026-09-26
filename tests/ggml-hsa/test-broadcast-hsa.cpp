// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Verifies broadcasting for GGML_OP_ADD / SUB / MUL / DIV on the HSA backend, i.e. src1
// broadcast over src0/dst. These are the patterns GPT-2 relies on:
//   - bias add:  cur[n_embd, N] + b[n_embd]          (ADD bias fast path)
//   - LN affine: norm[n_embd, N] * w[n_embd]         (general broadcast, MUL)
//   - LN affine: (...)          + b[n_embd]          (ADD broadcast)
// Builds a real op graph, computes on device, and compares to a CPU reference.
//
// Covers both dispatch paths in binary_op(): a src1 of exactly one row (src1_nr == 1) takes
// the vectorized ggml_op_*_row kernels, and a src1 of several rows (src1_nr > 1) falls back
// to the generic ggml_op_*_broadcast kernels, which recompute the src1 index per element.
// (The dtype-mismatched src1 path also lands in that fallback, but is not covered here:
// every case below builds src1 as F32.)

#include <cmath>
#include <cstddef>
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

enum class op_kind { add, sub, mul, div };

// Maps a float onto a monotonically increasing integer, so the difference between two keys
// is a true distance in representable floats. Subtracting the raw int32 bit patterns instead
// overflows whenever the two values straddle zero -- exactly the cancellation case this test
// provokes, which would make the reported ulp figure meaningless.
int64_t ulp_key(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof b);
    return (b & 0x80000000u) != 0 ? static_cast<int64_t>(0x80000000u) - static_cast<int64_t>(b)
                                  : static_cast<int64_t>(b);
}

int64_t ulp_distance(float want, float got) {
    const int64_t d = ulp_key(got) - ulp_key(want);
    return d < 0 ? -d : d;
}

// src0 [nc, nr, nz] op src1 [nc, src1_nr, 1] (repeated over the remaining rows/slices)
// -> dst [nc, nr, nz]. src1_nr must divide nr; src1_nr == 1 is the single-row fast path.
bool run_case(ggml_backend_t backend, op_kind kind, int64_t nc, int64_t nr, int64_t nz,
              int64_t src1_nr, const char * name) {
    const int64_t n = nc * nr * nz;
    const int64_t n1 = nc * src1_nr;

    // ggml_add/ggml_mul assert ggml_can_repeat, which needs nr % src1_nr == 0.
    // Check it here so a malformed table entry names itself instead of aborting
    // inside GGML before any output.
    if (src1_nr <= 0 || nr % src1_nr != 0) {
        printf("  %-22s: bad case: nr=%lld not a multiple of src1_nr=%lld\n", name,
               (long long)nr, (long long)src1_nr);
        return false;
    }

    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src0 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, nr, nz);
    // Broadcast operand. src1_nr == 1 gives ne == (nc,1,1,1), the single-row shape.
    ggml_tensor * src1 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, nc, src1_nr, 1);
    ggml_set_name(src0, "src0");
    ggml_set_name(src1, "src1");
    ggml_tensor * dst = nullptr;
    switch (kind) {
        case op_kind::add: dst = ggml_add(ctx.get(), src0, src1); break;
        case op_kind::sub: dst = ggml_sub(ctx.get(), src0, src1); break;
        case op_kind::mul: dst = ggml_mul(ctx.get(), src0, src1); break;
        case op_kind::div: dst = ggml_div(ctx.get(), src0, src1); break;
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

    uint64_t rng = 0x13198A2E03707344ULL;
    auto rnd = [&rng]() {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t r = static_cast<uint32_t>(rng >> 32);
        const uint32_t bits = (r & 0x807FFFFFu) | ((110u + (r >> 24) % 34u) << 23);
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    };
    std::vector<float> a_host(n);
    for (int64_t i = 0; i < n; ++i) { a_host[i] = rnd(); }
    std::vector<float> b_host(n1);
    for (int64_t i = 0; i < n1; ++i) { b_host[i] = rnd(); }
    ggml_backend_tensor_set(src0, a_host.data(), 0, ggml_nbytes(src0));
    ggml_backend_tensor_set(src1, b_host.data(), 0, ggml_nbytes(src1));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-22s: graph compute failed\n", name);
        return false;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // The denominator is per-op, because the two families fail differently.
    //
    // ADD and SUB on random operands cancel, and after cancellation a half-ulp
    // rounding difference in the sum is an arbitrarily large number of ulp of the
    // (tiny) result -- SUB peaks at 64 ulp here while being correctly rounded to
    // within half an ulp of its inputs. For those, max(|a|,|b|,|want|) is the
    // denominator that separates real drift from cancellation.
    //
    // MUL and DIV do not cancel, so that denominator would be far too generous:
    // a quotient many binades below max(|a|,|b|) could be wrong by orders of
    // magnitude and still land inside an operand-scaled bound. Those are gated on
    // |want| instead, which is a true relative error.
    //
    // Bound is 2 * 2^-23 in both cases. Measured on aie2 (NPU1) every op sits at
    // or below 1.192e-07 == 2^-23, i.e. within one ulp, which is the best aie2 can
    // do since it has no native fp32 ALU. The old absolute 1e-4 tolerance was
    // ~1000x looser than this at these magnitudes and could not see any of it.
    constexpr float max_rel = 2.0f / 8388608.0f; // 2 * 2^-23

    bool ok = true;
    int64_t worst_ulp = 0;
    float worst_rel = 0.0f;
    for (int64_t r = 0; r < nr * nz; ++r) {
        // src1 has ne2 == 1, so only the row index wraps: GGML repeats src1 row
        // (r % src1_nr) across every dst row.
        const int64_t b_row = (r % nr) % src1_nr;
        for (int64_t i = 0; i < nc; ++i) {
            const int64_t idx  = r * nc + i;
            const int64_t bidx = b_row * nc + i;
            float want = 0.0f;
            switch (kind) {
                case op_kind::add: want = a_host[idx] + b_host[bidx]; break;
                case op_kind::sub: want = a_host[idx] - b_host[bidx]; break;
                case op_kind::mul: want = a_host[idx] * b_host[bidx]; break;
                case op_kind::div: want = a_host[idx] / b_host[bidx]; break;
            }
            const float got = dst_host[idx];

            // A NaN or Inf fails every ordered comparison below, so screen it
            // explicitly: `NaN > max_rel` is false and would otherwise report
            // PASSED with a clean 0.000e+00 worst-rel line.
            if (!std::isfinite(got) && std::isfinite(want)) {
                if (ok) {
                    printf("  %-22s: non-finite at row %lld col %lld got %g want %.9g\n",
                           name, (long long)r, (long long)i, got, want);
                }
                ok = false;
                continue;
            }

            const int64_t ulp = ulp_distance(want, got);
            if (ulp > worst_ulp) { worst_ulp = ulp; }

            // ADD and SUB cancel, so their error is only meaningful against the
            // operand scale. MUL and DIV do not cancel: their result can sit
            // many binades below max(|a|,|b|), where an operand-scaled bound is
            // thousands of ulp of the result and lets grossly wrong values pass.
            // Gate those on the result instead.
            const bool cancels = (kind == op_kind::add || kind == op_kind::sub);
            const float scale =
                cancels ? std::fmax(std::fmax(std::fabs(a_host[idx]), std::fabs(b_host[bidx])),
                                    std::fabs(want))
                        : std::fabs(want);
            const float rel = scale > 0.0f ? std::fabs(got - want) / scale : 0.0f;
            if (rel > worst_rel) { worst_rel = rel; }

            if (rel > max_rel && ok) {
                printf("  %-22s: mismatch at row %lld col %lld got %.9g want %.9g (%.3e rel)\n",
                       name, (long long)r, (long long)i, got, want, rel);
                ok = false;
            }
        }
    }
    printf("  %-22s: worst %lld ulp, %.3e rel-to-%s (allowed %.3e)\n", name,
           (long long)worst_ulp, worst_rel,
           (kind == op_kind::add || kind == op_kind::sub) ? "operand" : "result", max_rel);
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
        op_kind kind;
        int64_t nc, nr, nz, src1_nr;
        const char * name;
    } cases[] = {
        {op_kind::add, 32, 8, 1, 1, "add bias 2d"},
        {op_kind::add, 768, 4, 1, 1, "add bias gpt2"},
        {op_kind::mul, 32, 8, 1, 1, "mul bcast 2d"},
        {op_kind::mul, 768, 4, 1, 1, "mul bcast gpt2"},
        {op_kind::add, 64, 4, 3, 1, "add bias 3d"},
        {op_kind::mul, 64, 4, 3, 1, "mul bcast 3d"},
        {op_kind::sub, 32, 8, 1, 1, "sub bcast 2d"},
        {op_kind::sub, 768, 4, 1, 1, "sub bcast gpt2"},
        {op_kind::sub, 64, 4, 3, 1, "sub bcast 3d"},
        {op_kind::div, 32, 8, 1, 1, "div bcast 2d"},
        {op_kind::div, 768, 4, 1, 1, "div bcast gpt2"},
        {op_kind::div, 64, 4, 3, 1, "div bcast 3d"},
        // Row narrower than the 16-element f32 vector: exercises the scalar tail with
        // vend == 0, the case that rules out AIE_LOOP_MIN_ITERATION_COUNT. All four ops,
        // since each has its own tail body.
        {op_kind::add, 10, 64, 1, 1, "add bias narrow row"},
        {op_kind::sub, 10, 64, 1, 1, "sub bcast narrow row"},
        {op_kind::mul, 10, 64, 1, 1, "mul bcast narrow row"},
        {op_kind::div, 10, 64, 1, 1, "div bcast narrow row"},
        // src1 spans several rows, so src1_is_row is false and dispatch falls back from the
        // vectorized *_row kernels to the generic *_broadcast ones.
        {op_kind::add, 64, 8, 2, 4, "add bcast rows"},
        {op_kind::sub, 64, 8, 2, 4, "sub bcast rows"},
        {op_kind::mul, 64, 8, 2, 4, "mul bcast rows"},
        {op_kind::div, 64, 8, 2, 4, "div bcast rows"},
    };

    bool all_ok = true;
    for (const auto & c : cases) {
        bool ok = run_case(backend, c.kind, c.nc, c.nr, c.nz, c.src1_nr, c.name);
        printf("BROADCAST %-22s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        all_ok = all_ok && ok;
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
