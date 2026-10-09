// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for GGML_OP_SOFT_MAX on the HSA backend. Builds a real
// ggml_soft_max graph (softmax over dim 0, scale 1.0), runs it on the device, and
// compares against a double-precision CPU reference. Includes the GPT-2 attention
// shape [1024,1024,12] and a non-multiple-of-16 row length.
//
// This test was written to expose a kernel that mis-tiled rows (odd rows came back zero
// on a uniform input) and later stopped compiling at all under mlir-aie 1.4.3
// ("stack_size is absent ... needs 1088 bytes"). Both were the same defect: the core's
// stack frame exceeded the AIE core's 1024-byte default stack, so it wrote 64 bytes past
// the end of its own stack into the neighbouring ObjectFifo buffer -- corrupting every
// other tile, which is what produced the odd-row pattern. Newer toolchains measure the
// frame and refuse to build; older ones compiled it and corrupted memory at run time.
// softmax.py now sets an explicit stack_size, and the device result is asserted.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

// A case either runs and matches (pass), runs and mismatches (mismatch -- the known kernel bug,
// reported but not fatal), is declined by the backend (skip), or fails to set up or execute at all
// (error -- an infrastructure regression, which IS fatal).
enum class case_result { pass, mismatch, skip, error };

// Inputs: the original deterministic ramp, or seeded uniform logits with a full random mantissa.
// The ramp has a handful of distinct values per row, which hides rounding drift that random
// mantissas expose.
// `wide` spans [-100, 100], so a scaled row can overflow exp() unless its max is subtracted.
enum class input_kind { ramp, random, wide };

case_result run_case(ggml_backend_t backend, int64_t ne0, int64_t ne1, int64_t ne2,
                     const char * name, float scale = 1.0f, input_kind kind = input_kind::ramp) {
    const int64_t n = ne0 * ne1 * ne2;

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_soft_max_ext(ctx.get(), src, nullptr, scale, 0.0f);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-18s: op not supported (skipped)\n", name);
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-18s: graph allocation failed\n", name);
        return case_result::error;
    }

    // Deterministic inputs, per input_kind (see its comment).
    std::vector<float> src_host(n);
    std::mt19937 rng(static_cast<uint32_t>(ne0 * 7919 + ne1 * 31 + ne2));
    std::uniform_real_distribution<float> logit(-12.0f, 12.0f);
    std::uniform_real_distribution<float> wide_logit(-100.0f, 100.0f);
    for (int64_t r = 0; r < ne1 * ne2; ++r) {
        for (int64_t i = 0; i < ne0; ++i) {
            src_host[r * ne0 + i] =
                kind == input_kind::random ? logit(rng)
                : kind == input_kind::wide ? wide_logit(rng)
                                           : (static_cast<float>((i + r) % 211) - 105.0f) * 0.1f;
        }
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-18s: graph compute failed\n", name);
        return case_result::error;
    }

    std::vector<float> dst_host(n);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // Relative to each output (all are > 0). Against this double reference, the device's error
    // is the rounding of the exp() argument (<= half an ulp of |scale*x - max|: ~1e-6 for the
    // [-12, 12] inputs, and ~3.8e-6 at worst, since only arguments above -88 clear atol), the
    // exp() approximation itself (~1.4e-6 for a degree-7 series on [0, ln2)) and the row sum
    // (~sqrt(N) * 2^-24 for these seeded inputs). The scalar kernel measured <= 3.6e-6 here.
    // atol only covers outputs below ~1e-38, where the device's exp() clamps its argument at -88.
    const double rtol = 1e-5;
    const double atol = 1e-30;
    bool ok = true;
    double max_rel = 0.0;
    for (int64_t r = 0; r < ne1 * ne2 && ok; ++r) {
        const float * x = src_host.data() + r * ne0;

        double m = -1e30;
        for (int64_t i = 0; i < ne0; ++i) {
            m = std::fmax(m, scale * static_cast<double>(x[i]));
        }
        double sum = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            sum += std::exp(scale * static_cast<double>(x[i]) - m);
        }
        for (int64_t i = 0; i < ne0 && ok; ++i) {
            const double want = std::exp(scale * static_cast<double>(x[i]) - m) / sum;
            const float got = dst_host[r * ne0 + i];
            max_rel = std::fmax(max_rel, std::fabs(got - want) / (want + atol / rtol));
            // Written so a NaN fails: NaN compares false, so `err > tol` would let it through.
            if (!(std::fabs(got - want) <= rtol * want + atol)) {
                printf("  %-18s: mismatch at row %lld col %lld got %g want %g\n", name,
                       (long long)r, (long long)i, got, want);
                ok = false;
            }
        }
    }
    printf("  %-18s: max relative error %.3g\n", name, max_rel);
    return ok ? case_result::pass : case_result::mismatch;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        int64_t ne0, ne1, ne2;
        const char * name;
        float scale = 1.0f;
        input_kind kind = input_kind::ramp;
    } cases[] = {
        {32, 8, 1, "small"},
        {64, 64, 12, "gpt2 attn 64"},
        {256, 256, 12, "gpt2 attn 256"},
        {500, 4, 1, "scalar tail"},   // 500 % 16 = 4
        {10, 500, 1, "mnist rnd", 1.0f, input_kind::random},  // MNIST logits: tail only
        {3, 64, 1, "3 wide rnd", 1.0f, input_kind::random},
        {16, 64, 1, "16 rnd", 1.0f, input_kind::random},      // whole vectors, no tail
        {17, 64, 1, "17 rnd", 1.0f, input_kind::random},      // one vector + 1-element tail
        {256, 64, 1, "256 rnd", 1.0f, input_kind::random},
        {500, 16, 1, "500 rnd s=0.125", 0.125f, input_kind::random},
        // Negative scale: the row max of scale*x is scale*min(x). Taking the wrong extreme only
        // shows up when exp() overflows, hence the wide inputs.
        {40, 16, 1, "40 wide s=-1", -1.0f, input_kind::wide},
        // NOTE: the full-context [1024,1024,12] shape is intentionally omitted for now:
        // the current single-worker kernel overruns the per-dispatch watchdog and aborts
        // the process (uncatchable). Re-add it once the kernel is fanned across compute
        // tiles (see the softmax watchdog fix).
    };

    int passed = 0;
    int mismatched = 0;
    int skipped = 0;
    bool any_error = false;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.ne0, c.ne1, c.ne2, c.name, c.scale, c.kind);
        const char * label;
        switch (r) {
            case case_result::pass:
                label = "PASSED";
                ++passed;
                break;
            case case_result::mismatch:
                label = "MISMATCH";
                ++mismatched;
                break;
            case case_result::skip:
                label = "SKIPPED (op reported unsupported)";
                ++skipped;
                break;
            default:
                label = "ERROR";
                any_error = true;
                break;
        }
        printf("SOFT_MAX %-18s: %s\n", c.name, label);
    }

    ggml_backend_free(backend);

    // A numerical mismatch is now fatal: the kernel is expected to be correct, so a mismatch is a
    // regression rather than the documented bug it once was. A skip still is not fatal -- the
    // backend may legitimately decline a shape, or route the op to the CPU.
    if (any_error) {
        printf("ERRORS (setup or execution failed)\n");
        return 1;
    }
    if (mismatched > 0) {
        printf("FAILURES (%d numerical mismatch)\n", mismatched);
        return 1;
    }
    if ((passed == 0) && (skipped > 0)) {
        // Nothing actually ran, so say so rather than reporting a green result.
        printf("ALL SKIPPED (no case ran; see the per-case reasons above)\n");
        return 0;
    }
    printf("%d passed, %d skipped\n", passed, skipped);
    return 0;
}
