// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the cache of converted constant MUL_MAT sources. An f32 MUL_MAT at a shape
// that is not a tile multiple runs as a padded bf16 GEMM: each operand is converted and
// zero-padded into an internal buffer first. For a graph-constant weight that conversion is done
// once and reused across computes, so the cache must notice every way the weight's contents can
// change. Each case computes C = W x X twice (or more) and checks every result against a host
// reference; inputs are small integers, so every product and partial sum is exact in bf16/f32 and
// the result must match bit for bit.
//
// The weight lives in a buffer of its own, as model weights do, apart from the input and the
// result, so writes to those do not touch the weight's buffer.

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

enum class case_result { pass, fail, skip };

enum class case_kind {
    // W is a trainable parameter (ggml_set_param), rewritten in place with ggml_backend_tensor_set
    // between the computes, as an optimizer step updates it at the same address.
    param_rewrite,
    // W is a plain constant (no flags), rewritten in place with ggml_backend_tensor_set.
    const_rewrite,
    // W is a plain constant, overwritten in place by a CPY computed on the HSA backend in another
    // graph (a KV-cache write is such a CPY).
    const_graph_write,
    // W is a plain constant that is not rewritten: the computes must reuse the cached conversion.
    const_cached,
};

const char * kind_name(case_kind kind) {
    switch (kind) {
        case case_kind::param_rewrite:
            return "param, tensor_set rewrite";
        case case_kind::const_rewrite:
            return "const, tensor_set rewrite";
        case case_kind::const_graph_write:
            return "const, CPY graph rewrite";
        case case_kind::const_cached:
            return "const, not rewritten";
    }
    return "?";
}

using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using gallocr_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

ctx_ptr make_ctx(std::size_t n_tensors) {
    ggml_init_params params{
        /*.mem_size   =*/n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return {ggml_init(params), ggml_free};
}

// Weight [K, M] in ggml layout; @p seed selects one of several distinct small-integer patterns.
std::vector<float> make_weight(int64_t M, int64_t K, int seed) {
    std::vector<float> w(M * K);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            w[i * K + k] = static_cast<float>((i * (7 + seed) + k * (3 + 2 * seed) + seed) % 5 - 2);
        }
    }
    return w;
}

// Checks C [M, N] against W [K, M] x X [K, N].
bool check(const char * label,
           const std::vector<float> & got,
           const std::vector<float> & w,
           const std::vector<float> & x,
           int64_t M,
           int64_t N,
           int64_t K) {
    int64_t mismatches = 0;
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            float want = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                want += w[i * K + k] * x[j * K + k];
            }
            if (got[j * M + i] != want) {
                if (mismatches < 4) {
                    printf("  %s: mismatch at row %lld col %lld: got %g want %g\n", label,
                           (long long)i, (long long)j, got[j * M + i], want);
                }
                ++mismatches;
            }
        }
    }
    if (mismatches != 0) {
        printf("  %s: %lld / %lld elements mismatched\n", label, (long long)mismatches,
               (long long)(M * N));
        return false;
    }
    return true;
}

case_result run_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, case_kind kind) {
    // the weight, in a buffer of its own
    ctx_ptr wctx = make_ctx(1);
    ggml_tensor * w = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_name(w, "w");
    if (kind == case_kind::param_rewrite) {
        ggml_set_param(w);
    }
    buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), backend), ggml_backend_buffer_free};
    if (wbuf == nullptr) {
        printf("  weight allocation failed\n");
        return case_result::fail;
    }

    // C = W x X
    ctx_ptr ctx = make_ctx(4);
    ggml_tensor * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * c = ggml_mul_mat(ctx.get(), w, x); // [M, N] f32
    ggml_set_name(c, "c");
    ggml_set_output(c);

    if (!ggml_backend_supports_op(backend, c)) {
        printf("  MUL_MAT not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);
    gallocr_ptr galloc{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
                       ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    // W' -> W, computed by the backend in a graph of its own; the source lives in that graph's
    // buffer, so only the CPY itself writes the weight's buffer
    ctx_ptr cctx = make_ctx(2);
    ggml_cgraph * gc = nullptr;
    gallocr_ptr cpy_galloc{nullptr, ggml_gallocr_free};
    ggml_tensor * w_src = nullptr;
    if (kind == case_kind::const_graph_write) {
        w_src = ggml_new_tensor_2d(cctx.get(), GGML_TYPE_F32, K, M);
        ggml_set_name(w_src, "w_src");
        ggml_set_input(w_src);
        ggml_tensor * cpy = ggml_cpy(cctx.get(), w_src, w);
        if (!ggml_backend_supports_op(backend, cpy)) {
            printf("  CPY not supported (skipped)\n");
            return case_result::skip;
        }
        gc = ggml_new_graph(cctx.get());
        ggml_build_forward_expand(gc, cpy);
        cpy_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
        if (!ggml_gallocr_alloc_graph(cpy_galloc.get(), gc)) {
            printf("  CPY graph allocation failed\n");
            return case_result::fail;
        }
    }

    std::vector<float> hx(N * K);
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }
    const std::vector<float> w1 = make_weight(M, K, 0);
    const std::vector<float> w2 = make_weight(M, K, 1);
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(w, w1.data(), 0, ggml_nbytes(w));

    std::vector<float> got(M * N);
    auto compute = [&](const char * label, const std::vector<float> & want_w) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  %s: graph compute failed\n", label);
            return false;
        }
        ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));
        return check(label, got, want_w, hx, M, N, K);
    };

    bool ok = compute("compute 1", w1);

    switch (kind) {
        case case_kind::param_rewrite:
        case case_kind::const_rewrite:
            // same tensor, same address, new contents
            ggml_backend_tensor_set(w, w2.data(), 0, ggml_nbytes(w));
            ok = compute("compute 2 (after rewrite)", w2) && ok;
            break;
        case case_kind::const_graph_write:
            ggml_backend_tensor_set(w_src, w2.data(), 0, ggml_nbytes(w_src));
            if (ggml_backend_graph_compute(backend, gc) != GGML_STATUS_SUCCESS) {
                printf("  CPY graph compute failed\n");
                return case_result::fail;
            }
            ok = compute("compute 2 (after CPY rewrite)", w2) && ok;
            break;
        case case_kind::const_cached:
            ok = compute("compute 2", w1) && ok;
            // Shows that the second compute reused the cached conversion without a test hook:
            // HSA buffers are host-mapped, so a store straight to the weight's memory, bypassing
            // every ggml write path, is something the cache cannot see. A compute that still
            // returns the first weight's result therefore did not re-convert the weight.
            std::memcpy(w->data, w2.data(), ggml_nbytes(w));
            ok = compute("compute 3 (after untracked store; expects cached W)", w1) && ok;
            break;
    }

    return ok ? case_result::pass : case_result::fail;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    // M x N x K. Neither is a tile multiple, so both run as padded GEMMs with converted operands:
    // the MNIST fully-connected layers (fc1: 784 -> 500, fc2: 500 -> 10) at a batch of 500.
    struct {
        int64_t M, N, K;
    } shapes[] = {
        {500, 500, 784},
        {10, 500, 500},
    };
    const case_kind kinds[] = {case_kind::param_rewrite, case_kind::const_rewrite,
                               case_kind::const_graph_write, case_kind::const_cached};

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & s : shapes) {
        for (const case_kind kind : kinds) {
            const case_result r = run_case(backend, s.M, s.N, s.K, kind);
            const char * label = r == case_result::pass   ? "PASSED"
                                 : r == case_result::skip ? "SKIPPED"
                                                          : "FAILED";
            printf("MUL_MAT f32 %lldx%lldx%lld %-26s: %s\n", (long long)s.M, (long long)s.N,
                   (long long)s.K, kind_name(kind), label);
            any_fail = any_fail || (r == case_result::fail);
            passed += (r == case_result::pass);
            skipped += (r == case_result::skip);
        }
    }
    ggml_backend_free(backend);

    if (any_fail) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
