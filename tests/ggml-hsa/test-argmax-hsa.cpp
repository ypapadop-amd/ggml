// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for GGML_OP_ARGMAX on the HSA backend, over deterministic rows rather than
// random data: exact ties, NaN in the first lane, NaN elsewhere, an all-NaN row, and rows whose
// maximum sits in the lane the vector path pads from (lane 0).
//
// Random inputs never produce a tie and never produce a NaN, so they exercise none of the cases
// where a vectorized reduction can legitimately disagree with a scalar scan. Each row below is
// checked against ggml_vec_argmax_f32 (ggml-cpu/vec.h), which is the reference the backend has
// to match:
//
//     float max = -INFINITY; int idx = 0;
//     for (i) { max = MAX(max, x[i]); if (max == x[i]) { idx = i; } }
//
// Two properties of that reference are easy to get wrong and are what most of these rows pin
// down. It keeps the LAST index equal to the maximum, not the first. And because MAX(a,b) is
// (a > b ? a : b), a NaN operand is not propagated -- it is replaced by the next element -- so
// a leading NaN does not poison the scan.


#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

/// CPU reference: ggml_vec_argmax_f32 from ggml-cpu/vec.h, transcribed.
int32_t reference_argmax(const std::vector<float> & x) {
    float max = -kInf;
    int32_t idx = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        max = (max > x[i]) ? max : x[i];
        if (max == x[i]) {
            idx = static_cast<int32_t>(i);
        }
    }
    return idx;
}

struct test_row {
    const char *       name;
    std::vector<float> values;
};

/// Runs ARGMAX over `rows` (all the same length) as one [nc, nr] tensor and checks every row.
bool run_rows(ggml_backend_t backend, const std::vector<test_row> & rows) {
    const int64_t nc = static_cast<int64_t>(rows[0].values.size());
    const int64_t nr = static_cast<int64_t>(rows.size());

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nc, nr);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_argmax(ctx.get(), src);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  ARGMAX [%lld x %lld]: op not supported\n", (long long)nc, (long long)nr);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  ARGMAX [%lld x %lld]: graph allocation failed\n", (long long)nc, (long long)nr);
        return false;
    }

    std::vector<float> host(static_cast<std::size_t>(nc * nr));
    for (int64_t r = 0; r < nr; ++r) {
        for (int64_t c = 0; c < nc; ++c) {
            host[static_cast<std::size_t>(r * nc + c)] = rows[r].values[c];
        }
    }
    ggml_backend_tensor_set(src, host.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  ARGMAX [%lld x %lld]: graph compute failed\n", (long long)nc, (long long)nr);
        return false;
    }

    std::vector<int32_t> got(static_cast<std::size_t>(nr));
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));

    bool all_ok = true;
    for (int64_t r = 0; r < nr; ++r) {
        const int32_t want = reference_argmax(rows[r].values);
        const bool    ok   = got[static_cast<std::size_t>(r)] == want;
        printf("  %-26s: got %2d want %2d  %s\n", rows[r].name, got[static_cast<std::size_t>(r)],
               want, ok ? "OK" : "MISMATCH");
        all_ok = all_ok && ok;
    }
    return all_ok;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    // 10 wide: the MNIST class count, and under the 16-lane f32 vector, so the vector path pads.
    const std::vector<test_row> rows10 = {
        {"plain max at 4", {0.f, 1.f, 2.f, 3.f, 9.f, 3.f, 2.f, 1.f, 0.f, -1.f}},
        {"max at 0", {9.f, 1.f, 2.f, 3.f, 4.f, 3.f, 2.f, 1.f, 0.f, -1.f}},
        {"max at last", {0.f, 1.f, 2.f, 3.f, 4.f, 3.f, 2.f, 1.f, 0.f, 9.f}},
        {"tie at 2 and 7", {0.f, 1.f, 9.f, 3.f, 4.f, 3.f, 2.f, 9.f, 0.f, -1.f}},
        {"tie at 0 and 5", {9.f, 1.f, 2.f, 3.f, 4.f, 9.f, 2.f, 1.f, 0.f, -1.f}},
        {"all equal", {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f}},
        {"NaN at 0, max at 3", {kNaN, 1.f, 2.f, 7.f, 4.f, 3.f, 2.f, 1.f, 0.f, -1.f}},
        {"NaN at 0, max at 0*", {kNaN, -1.f, -2.f, -3.f, -4.f, -5.f, -6.f, -7.f, -8.f, -9.f}},
        {"NaN in middle", {0.f, 1.f, kNaN, 3.f, 8.f, 3.f, 2.f, 1.f, 0.f, -1.f}},
        {"all NaN", {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}},
        {"+inf at 6", {0.f, 1.f, 2.f, 3.f, 4.f, 3.f, kInf, 1.f, 0.f, -1.f}},
        {"all -inf", {-kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf}},
        {"negatives only", {-5.f, -4.f, -3.f, -2.f, -1.5f, -2.f, -3.f, -4.f, -5.f, -6.f}},
        {"signed zeros", {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}},
    };

    // 16 wide: exactly one vector, so the vector path pads nothing.
    const std::vector<test_row> rows16 = {
        {"16-wide max at 11",
         {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 99.f, 10.f, 9.f, 8.f, 7.f}},
        {"16-wide tie 3 and 12",
         {0.f, 1.f, 2.f, 9.f, 4.f, 5.f, 6.f, 7.f, 8.f, 3.f, 2.f, 1.f, 9.f, 0.f, -1.f, -2.f}},
        {"16-wide NaN at 0",
         {kNaN, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f}},
    };

    bool all_ok = true;
    printf("ARGMAX 10-wide (vector path pads lanes 10..15 from lane 0):\n");
    all_ok = run_rows(backend, rows10) && all_ok;
    printf("ARGMAX 16-wide (no padding):\n");
    all_ok = run_rows(backend, rows16) && all_ok;

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
