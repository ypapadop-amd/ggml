// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the recycling of HSA tensor extras across graph allocations. Checks that a
// rebuilt graph recycles the first graph's extras, that a graph allocated again keeps its extras,
// and that each node has an extra of its own. The ops are f16 ADDs, whose sources use internal
// buffers; inputs are small integers, so results must match exactly.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <set>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using gallocr_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

constexpr int64_t n = 1024;

ctx_ptr make_ctx(std::size_t n_tensors) {
    ggml_init_params params{
        /*.mem_size   =*/n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return {ggml_init(params), ggml_free};
}

std::vector<ggml_fp16_t> pattern(int seed) {
    std::vector<ggml_fp16_t> v(n);
    for (int64_t i = 0; i < n; ++i) {
        v[i] = ggml_fp32_to_fp16(static_cast<float>((i * (3 + seed) + seed) % 9 - 4));
    }
    return v;
}

// C = X + Y on f16 tensors of a context of its own.
struct add_graph {
    ctx_ptr ctx = make_ctx(3);
    ggml_tensor * x = nullptr;
    ggml_tensor * y = nullptr;
    ggml_tensor * c = nullptr;
    ggml_cgraph * gf = nullptr;

    add_graph() {
        x = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
        y = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
        ggml_set_input(x);
        ggml_set_input(y);
        c = ggml_add(ctx.get(), x, y);
        ggml_set_output(c);
        gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, c);
    }

    // the extras of the graph's tensors in the allocator's buffer
    std::set<const void *> extras() const { return {x->extra, y->extra, c->extra}; }
};

// Sets the inputs of C = X + Y, computes the graph and checks C.
bool compute_add(ggml_backend_t backend,
                 ggml_cgraph * gf,
                 ggml_tensor * x,
                 ggml_tensor * y,
                 ggml_tensor * c,
                 int seed,
                 const char * label) {
    const auto hx = pattern(seed);
    const auto hy = pattern(seed + 1);
    ggml_backend_tensor_set(x, hx.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(y, hy.data(), 0, ggml_nbytes(y));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %s: graph compute failed\n", label);
        return false;
    }
    std::vector<ggml_fp16_t> got(n);
    ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));
    int64_t mismatches = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float want = ggml_fp16_to_fp32(hx[i]) + ggml_fp16_to_fp32(hy[i]);
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
}

// Rebuilds C = X + Y over one allocator: each rebuild must recycle the first graph's extras.
bool run_rebuild_case(ggml_backend_t backend) {
    gallocr_ptr galloc{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
                       ggml_gallocr_free};
    std::set<const void *> first_extras;
    bool ok = true;
    for (int rebuild = 0; rebuild < 5; ++rebuild) {
        add_graph g;
        if (!ggml_gallocr_alloc_graph(galloc.get(), g.gf)) {
            printf("  rebuild %d: graph allocation failed\n", rebuild);
            return false;
        }
        if (rebuild == 0) {
            first_extras = g.extras();
        } else if (g.extras() != first_extras) {
            printf("  rebuild %d: the graph got new extras instead of recycling the first ones\n",
                   rebuild);
            ok = false;
        }
        char label[32];
        snprintf(label, sizeof(label), "rebuild %d", rebuild);
        ok = compute_add(backend, g.gf, g.x, g.y, g.c, rebuild, label) && ok;
    }
    return ok;
}

// Allocates one graph twice: the second allocation resets the buffer but initializes no tensor, so
// every tensor must keep its extra intact.
bool run_realloc_case(ggml_backend_t backend) {
    gallocr_ptr galloc{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
                       ggml_gallocr_free};
    add_graph g;
    if (!ggml_gallocr_alloc_graph(galloc.get(), g.gf)) {
        printf("  graph allocation failed\n");
        return false;
    }
    const auto extras = g.extras();
    bool ok = compute_add(backend, g.gf, g.x, g.y, g.c, 0, "first allocation");
    if (!ggml_gallocr_alloc_graph(galloc.get(), g.gf)) {
        printf("  second graph allocation failed\n");
        return false;
    }
    if (g.extras() != extras) {
        printf("  the second allocation changed the tensors' extras\n");
        ok = false;
    }
    return compute_add(backend, g.gf, g.x, g.y, g.c, 1, "second allocation") && ok;
}

// Two same-shaped ADDs in one graph have extras of their own.
bool run_distinct_case(ggml_backend_t backend) {
    ctx_ptr ctx = make_ctx(6);
    ggml_tensor * x1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
    ggml_tensor * y1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
    ggml_tensor * x2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
    ggml_tensor * y2 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, n);
    for (ggml_tensor * t : {x1, y1, x2, y2}) {
        ggml_set_input(t);
    }
    ggml_tensor * c1 = ggml_add(ctx.get(), x1, y1);
    ggml_tensor * c2 = ggml_add(ctx.get(), x2, y2);
    ggml_set_output(c1);
    ggml_set_output(c2);
    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c1);
    ggml_build_forward_expand(gf, c2);
    gallocr_ptr galloc{ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
                       ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return false;
    }
    bool ok = true;
    if (c1->extra == nullptr || c1->extra == c2->extra) {
        printf("  the two ADDs do not have extras of their own\n");
        ok = false;
    }
    ok = compute_add(backend, gf, x1, y1, c1, 10, "C1") && ok;
    return compute_add(backend, gf, x2, y2, c2, 20, "C2") && ok;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    struct {
        bool (*run)(ggml_backend_t);
        const char * name;
    } cases[] = {
        {run_rebuild_case, "rebuilt graph recycles its extras"},
        {run_realloc_case, "reallocated graph keeps its extras"},
        {run_distinct_case, "each node has an extra of its own"},
    };
    bool any_fail = false;
    for (const auto & c : cases) {
        const bool ok = c.run(backend);
        printf("%s: %s\n", c.name, ok ? "PASSED" : "FAILED");
        any_fail = any_fail || !ok;
    }

    ggml_backend_free(backend);
    if (any_fail) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
