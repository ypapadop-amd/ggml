// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the cache of converted constant MUL_MAT sources. An f32 MUL_MAT at a shape
// that is not a tile multiple runs as a padded bf16 GEMM: each operand is converted and
// zero-padded into an internal buffer first. For a graph-constant weight that conversion is done
// once and reused across computes (a constant must not be rewritten in place after its first use).
// Each case computes C = W x X several times and checks every result against a host reference;
// inputs are small integers, so every product and partial sum is exact in bf16/f32 and the result
// must match bit for bit.
//
// The weight lives in a buffer of its own, as model weights do, apart from the input and the
// result. An f16 ADD case checks that the cache is not specific to MUL_MAT. A last case places the
// weight in the graph's own compute buffer, where it is not cached: it is rewritten between
// computes at the same address and every compute must see its current contents. So is a plain view
// of a weight, which has no extra to keep a converted copy in. The last case checks that a
// conversion that never ran, because its queue was suspended, is not reused by another backend,
// nor is one made by a backend that waited on work that never ran.

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

// A graph C = op(X, W) over a weight W that lives elsewhere, with its own context and allocator.
struct graph_t {
    ctx_ptr ctx{nullptr, ggml_free};
    gallocr_ptr galloc{nullptr, ggml_gallocr_free};
    ggml_tensor * x = nullptr;
    ggml_tensor * c = nullptr;
    ggml_cgraph * gf = nullptr;
};

// Builds and allocates C = @p op(ctx, X) for an input X of @p x_type and shape [@p ne0, @p ne1].
// gf is null if the op is unsupported or the allocation failed.
template <typename Op>
graph_t make_graph(ggml_backend_t backend, ggml_type x_type, int64_t ne0, int64_t ne1, Op op) {
    graph_t g;
    g.ctx = make_ctx(4);
    g.x = ggml_new_tensor_2d(g.ctx.get(), x_type, ne0, ne1);
    ggml_set_name(g.x, "x");
    ggml_set_input(g.x);
    g.c = op(g.ctx.get(), g.x);
    ggml_set_name(g.c, "c");
    ggml_set_output(g.c);
    if (!ggml_backend_supports_op(backend, g.c)) {
        return g;
    }
    ggml_cgraph * gf = ggml_new_graph(g.ctx.get());
    ggml_build_forward_expand(gf, g.c);
    g.galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (ggml_gallocr_alloc_graph(g.galloc.get(), gf)) {
        g.gf = gf;
    }
    return g;
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

case_result run_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K) {
    // the weight, in a buffer of its own
    ctx_ptr wctx = make_ctx(1);
    ggml_tensor * w = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_name(w, "w");
    buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), backend), ggml_backend_buffer_free};
    if (wbuf == nullptr) {
        printf("  weight allocation failed\n");
        return case_result::fail;
    }

    // C = W x X, [M, N] f32
    auto build = [&] {
        return make_graph(backend, GGML_TYPE_F32, K, N, [w](ggml_context * ctx, ggml_tensor * x) {
            return ggml_mul_mat(ctx, w, x);
        });
    };
    graph_t g = build();
    if (!ggml_backend_supports_op(backend, g.c)) {
        printf("  MUL_MAT not supported (skipped)\n");
        return case_result::skip;
    }
    if (g.gf == nullptr) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<float> hx(N * K);
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }
    const std::vector<float> w1 = make_weight(M, K, 0);
    const std::vector<float> w2 = make_weight(M, K, 1);
    ggml_backend_tensor_set(w, w1.data(), 0, ggml_nbytes(w));

    std::vector<float> got(M * N);
    auto compute = [&](const char * label, const std::vector<float> & want_w) {
        ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            printf("  %s: graph compute failed\n", label);
            return false;
        }
        ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
        return check(label, got, want_w, hx, M, N, K);
    };

    bool ok = compute("compute 1", w1);

    ok = compute("compute 2", w1) && ok;
    // Shows that the second compute reused the cached conversion without a test hook: HSA buffers
    // are host-mapped, so a store straight to the weight's memory, bypassing every ggml write path,
    // is something the cache cannot see. A compute that still returns the first weight's result
    // therefore did not re-convert the weight.
    std::memcpy(w->data, w2.data(), ggml_nbytes(w));
    ok = compute("compute 3 (after a store to W; expects cached W)", w1) && ok;
    // The converted weight lives with W, not with the graph that first used it: a new graph over
    // the same W, with its own allocation, still gets the first weight's result.
    g = build();
    ok = g.gf != nullptr && compute("compute 4 (new graph; expects cached W)", w1) && ok;

    return ok ? case_result::pass : case_result::fail;
}

// The cache is not specific to MUL_MAT: any source that needs an internal buffer is cached when it
// is a constant. Here an f16 ADD, whose sources are converted to bf16, adds a constant weight W to
// an input X; inputs are small integers, so the result is exact in f16 and bf16.
case_result run_add_case(ggml_backend_t backend, int64_t n) {
    ctx_ptr wctx = make_ctx(1);
    ggml_tensor * w = ggml_new_tensor_1d(wctx.get(), GGML_TYPE_F16, n);
    ggml_set_name(w, "w");
    buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), backend), ggml_backend_buffer_free};
    if (wbuf == nullptr) {
        printf("  weight allocation failed\n");
        return case_result::fail;
    }

    auto build = [&] {
        return make_graph(backend, GGML_TYPE_F16, n, 1, [w](ggml_context * ctx, ggml_tensor * x) {
            return ggml_add(ctx, x, w);
        });
    };
    graph_t g = build();
    if (!ggml_backend_supports_op(backend, g.c)) {
        printf("  ADD not supported (skipped)\n");
        return case_result::skip;
    }
    if (g.gf == nullptr) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    auto to_f16 = [n](auto && fn) {
        std::vector<ggml_fp16_t> v(n);
        for (int64_t i = 0; i < n; ++i) {
            v[i] = ggml_fp32_to_fp16(static_cast<float>(fn(i)));
        }
        return v;
    };
    const auto hx = to_f16([](int64_t i) { return i % 7 - 3; });
    const auto w1 = to_f16([](int64_t i) { return i % 5 - 2; });
    const auto w2 = to_f16([](int64_t i) { return i % 3 + 10; });
    ggml_backend_tensor_set(w, w1.data(), 0, ggml_nbytes(w));

    std::vector<ggml_fp16_t> got(n);
    auto compute = [&](const char * label, const std::vector<ggml_fp16_t> & want_w) {
        // the allocator may place C over X (ADD can run in place), so X is set before every compute
        ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            printf("  %s: graph compute failed\n", label);
            return false;
        }
        ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
        int64_t mismatches = 0;
        for (int64_t i = 0; i < n; ++i) {
            const float want = ggml_fp16_to_fp32(hx[i]) + ggml_fp16_to_fp32(want_w[i]);
            if (ggml_fp16_to_fp32(got[i]) != want) {
                if (mismatches < 4) {
                    printf("  %s: mismatch at %lld: got %g want %g\n", label, (long long)i,
                           ggml_fp16_to_fp32(got[i]), want);
                }
                ++mismatches;
            }
        }
        if (mismatches != 0) {
            printf("  %s: %lld / %lld elements mismatched\n", label, (long long)mismatches,
                   (long long)n);
        }
        return mismatches == 0;
    };

    bool ok = compute("compute 1", w1);
    ok = compute("compute 2", w1) && ok;
    // as in run_case: a store the cache cannot see, so the cached W must still be used, also by a
    // new graph over the same W
    std::memcpy(w->data, w2.data(), ggml_nbytes(w));
    ok = compute("compute 3 (after a store to W; expects cached W)", w1) && ok;
    g = build();
    ok = g.gf != nullptr && compute("compute 4 (new graph; expects cached W)", w1) && ok;

    return ok ? case_result::pass : case_result::fail;
}

// A leaf the graph allocator places in its compute buffer is not a constant, even when it is not
// flagged as an input: such leaves are commonly set before every compute at the same address, and
// the allocator recycles their extras on the next graph allocation. Here W is such a leaf,
// rewritten between computes, and every compute must use its current contents.
case_result run_compute_buffer_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K) {
    graph_t g;
    g.ctx = make_ctx(4);
    ggml_tensor * w = ggml_new_tensor_2d(g.ctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_name(w, "w");
    g.x = ggml_new_tensor_2d(g.ctx.get(), GGML_TYPE_F32, K, N);
    ggml_set_name(g.x, "x");
    ggml_set_input(g.x);
    g.c = ggml_mul_mat(g.ctx.get(), w, g.x);
    ggml_set_name(g.c, "c");
    ggml_set_output(g.c);
    if (!ggml_backend_supports_op(backend, g.c)) {
        printf("  MUL_MAT not supported (skipped)\n");
        return case_result::skip;
    }
    g.gf = ggml_new_graph(g.ctx.get());
    ggml_build_forward_expand(g.gf, g.c);
    g.galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!ggml_gallocr_alloc_graph(g.galloc.get(), g.gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<float> hx(N * K);
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }

    std::vector<float> got(M * N);
    auto compute = [&](const char * label, const std::vector<float> & hw) {
        ggml_backend_tensor_set(w, hw.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            printf("  %s: graph compute failed\n", label);
            return false;
        }
        ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
        return check(label, got, hw, hx, M, N, K);
    };

    bool ok = compute("compute 1", make_weight(M, K, 0));
    ok = compute("compute 2 (W rewritten; expects the new W)", make_weight(M, K, 1)) && ok;

    return ok ? case_result::pass : case_result::fail;
}

// A plain view of a weight (ggml_view_tensor) is a leaf too, but HSA buffers give views no extra
// to keep a converted copy in, so it is not cached: every compute converts it.
case_result run_view_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K) {
    ctx_ptr wctx = make_ctx(1);
    ggml_tensor * w = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_name(w, "w");
    buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), backend), ggml_backend_buffer_free};
    if (wbuf == nullptr) {
        printf("  weight allocation failed\n");
        return case_result::fail;
    }

    graph_t g;
    g.ctx = make_ctx(4);
    g.x = ggml_new_tensor_2d(g.ctx.get(), GGML_TYPE_F32, K, N);
    ggml_set_name(g.x, "x");
    ggml_set_input(g.x);
    g.c = ggml_mul_mat(g.ctx.get(), ggml_view_tensor(g.ctx.get(), w), g.x);
    ggml_set_name(g.c, "c");
    ggml_set_output(g.c);
    if (!ggml_backend_supports_op(backend, g.c)) {
        printf("  MUL_MAT not supported (skipped)\n");
        return case_result::skip;
    }
    g.gf = ggml_new_graph(g.ctx.get());
    // W is expanded too: the allocator sizes its hash set from the graph's tensors but also hashes
    // the view's source, which a graph this small has no room for otherwise
    ggml_build_forward_expand(g.gf, w);
    ggml_build_forward_expand(g.gf, g.c);
    g.galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!ggml_gallocr_alloc_graph(g.galloc.get(), g.gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<float> hx(N * K);
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }

    std::vector<float> got(M * N);
    auto compute = [&](const char * label, const std::vector<float> & hw) {
        ggml_backend_tensor_set(w, hw.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            printf("  %s: graph compute failed\n", label);
            return false;
        }
        ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
        return check(label, got, hw, hx, M, N, K);
    };

    bool ok = compute("compute 1", make_weight(M, K, 0));
    ok = compute("compute 2 (W rewritten; expects the new W)", make_weight(M, K, 1)) && ok;

    return ok ? case_result::pass : case_result::fail;
}

// Suspends @p backend 's queue: a queue holds at most 32 distinct kernels and the next dispatch
// fails, so this casts at distinct sizes until a compute fails. Returns false if none did.
bool suspend_queue(ggml_backend_t backend) {
    for (int64_t i = 1; i <= 64; ++i) {
        graph_t g = make_graph(backend, GGML_TYPE_F32, 1024 * i, 1, [](ggml_context * ctx,
                                                                       ggml_tensor * x) {
            return ggml_cast(ctx, x, GGML_TYPE_BF16);
        });
        if (g.gf == nullptr) {
            return false;
        }
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
            return true;
        }
    }
    return false;
}

// A constant's conversion that never ran must not be trusted later. On a suspended queue the
// conversion is only written to the queue, so it seems to succeed, but it never runs and the
// compute fails. A second backend, on a queue of its own, must then convert the weight again
// rather than read a copy that was never written.
case_result run_failed_conversion_case(int64_t M, int64_t N, int64_t K) {
    ggml_backend_t failing = ggml_backend_hsa_init(0);
    ggml_backend_t fresh = ggml_backend_hsa_init(0);
    if (failing == nullptr || fresh == nullptr) {
        ggml_backend_free(failing);
        ggml_backend_free(fresh);
        printf("  backend initialization failed\n");
        return case_result::fail;
    }
    case_result result = case_result::fail;
    {
        ctx_ptr wctx = make_ctx(1);
        ggml_tensor * w = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, K, M);
        ggml_set_name(w, "w");
        buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), fresh),
                        ggml_backend_buffer_free};
        graph_t g =
            make_graph(fresh, GGML_TYPE_F32, K, N, [w](ggml_context * ctx, ggml_tensor * x) {
                return ggml_mul_mat(ctx, w, x);
            });
        std::vector<float> hx(N * K);
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t k = 0; k < K; ++k) {
                hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
            }
        }
        const std::vector<float> hw = make_weight(M, K, 0);
        if (wbuf == nullptr || g.gf == nullptr) {
            printf("  allocation failed\n");
        } else if (!ggml_backend_supports_op(fresh, g.c)) {
            printf("  MUL_MAT not supported (skipped)\n");
            result = case_result::skip;
        } else if (!suspend_queue(failing)) {
            printf("  could not suspend a queue (skipped)\n");
            result = case_result::skip;
        } else {
            ggml_backend_tensor_set(w, hw.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
            if (ggml_backend_graph_compute(failing, g.gf) == GGML_STATUS_SUCCESS) {
                printf("  compute on a suspended queue succeeded\n");
            } else if (ggml_backend_graph_compute(fresh, g.gf) != GGML_STATUS_SUCCESS) {
                printf("  compute on a fresh backend failed\n");
            } else {
                std::vector<float> got(M * N);
                ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
                result = check("compute on a fresh backend", got, hw, hx, M, N, K)
                             ? case_result::pass
                             : case_result::fail;
            }
        }
    }
    ggml_backend_free(failing);
    ggml_backend_free(fresh);
    return result;
}

// A backend that waited on another queue's work that never ran computes unsound results, so its
// conversions must not be published either. The consumer waits on an event of a suspended
// producer, then computes with W (and fails). W is then changed by a store the cache cannot see,
// so a fresh backend returns the new W's result only if the consumer published nothing.
case_result run_failed_dependency_case(int64_t M, int64_t N, int64_t K) {
    ggml_backend_t producer = ggml_backend_hsa_init(0);
    ggml_backend_t consumer = ggml_backend_hsa_init(0);
    ggml_backend_t fresh = ggml_backend_hsa_init(0);
    ggml_backend_event_t event =
        consumer != nullptr ? ggml_backend_event_new(ggml_backend_get_device(consumer)) : nullptr;
    case_result result = case_result::fail;
    if (producer == nullptr || consumer == nullptr || fresh == nullptr || event == nullptr) {
        printf("  backend or event initialization failed\n");
    } else {
        ctx_ptr wctx = make_ctx(1);
        ggml_tensor * w = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, K, M);
        ggml_set_name(w, "w");
        buffer_ptr wbuf{ggml_backend_alloc_ctx_tensors(wctx.get(), fresh),
                        ggml_backend_buffer_free};
        graph_t g =
            make_graph(fresh, GGML_TYPE_F32, K, N, [w](ggml_context * ctx, ggml_tensor * x) {
                return ggml_mul_mat(ctx, w, x);
            });
        std::vector<float> hx(N * K);
        for (int64_t j = 0; j < N; ++j) {
            for (int64_t k = 0; k < K; ++k) {
                hx[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
            }
        }
        const std::vector<float> w1 = make_weight(M, K, 0);
        const std::vector<float> w2 = make_weight(M, K, 1);
        if (wbuf == nullptr || g.gf == nullptr) {
            printf("  allocation failed\n");
        } else if (!ggml_backend_supports_op(fresh, g.c)) {
            printf("  MUL_MAT not supported (skipped)\n");
            result = case_result::skip;
        } else if (!suspend_queue(producer)) {
            printf("  could not suspend a queue (skipped)\n");
            result = case_result::skip;
        } else {
            ggml_backend_event_record(event, producer);
            ggml_backend_event_wait(consumer, event);
            ggml_backend_tensor_set(w, w1.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(g.x, hx.data(), 0, ggml_nbytes(g.x));
            if (ggml_backend_graph_compute(consumer, g.gf) == GGML_STATUS_SUCCESS) {
                printf("  compute after a failed dependency succeeded\n");
            } else {
                // a store straight to W's host-mapped memory, which the cache cannot see
                std::memcpy(w->data, w2.data(), ggml_nbytes(w));
                if (ggml_backend_graph_compute(fresh, g.gf) != GGML_STATUS_SUCCESS) {
                    printf("  compute on a fresh backend failed\n");
                } else {
                    std::vector<float> got(M * N);
                    ggml_backend_tensor_get(g.c, got.data(), 0, ggml_nbytes(g.c));
                    result = check("compute on a fresh backend (expects the new W)", got, w2, hx,
                                   M, N, K)
                                 ? case_result::pass
                                 : case_result::fail;
                }
            }
        }
    }
    if (event != nullptr) {
        ggml_backend_event_free(event);
    }
    ggml_backend_free(producer);
    ggml_backend_free(consumer);
    ggml_backend_free(fresh);
    return result;
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

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & s : shapes) {
        const case_result r = run_case(backend, s.M, s.N, s.K);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 %lldx%lldx%lld cached constant: %s\n", (long long)s.M,
               (long long)s.N, (long long)s.K, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    for (const auto & s : shapes) {
        const case_result r = run_compute_buffer_case(backend, s.M, s.N, s.K);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 %lldx%lldx%lld weight in compute buffer: %s\n", (long long)s.M,
               (long long)s.N, (long long)s.K, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    for (const auto & s : shapes) {
        const case_result r = run_view_case(backend, s.M, s.N, s.K);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 %lldx%lldx%lld view of a weight: %s\n", (long long)s.M,
               (long long)s.N, (long long)s.K, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    for (const int64_t n : {1024, 4096}) {
        const case_result r = run_add_case(backend, n);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("ADD f16 %lld cached constant: %s\n", (long long)n, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    ggml_backend_free(backend);

    {
        const case_result r = run_failed_conversion_case(500, 500, 784);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 500x500x784 conversion on a suspended queue: %s\n", label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    {
        const case_result r = run_failed_dependency_case(500, 500, 784);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 500x500x784 conversion after a failed dependency: %s\n", label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    if (any_fail) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
