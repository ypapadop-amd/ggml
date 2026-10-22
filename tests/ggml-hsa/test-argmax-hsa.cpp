// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for GGML_OP_ARGMAX on the HSA backend, over deterministic rows rather than
// random data: exact ties, NaN in the first lane, NaN elsewhere, an all-NaN row, and rows whose
// maximum sits at either end of the row.
//
// Random inputs never produce a tie and never produce a NaN, so they exercise none of the cases
// where an argmax implementation can legitimately disagree with the reference. Each row below is
// checked against ggml_vec_argmax_f32 (ggml-cpu/vec.h), which is the reference the backend has
// to match:
//
//     float max = -INFINITY; int idx = 0;
//     for (i) { max = MAX(max, x[i]); if (max == x[i]) { idx = i; } }
//
// Two properties of that reference are easy to get wrong and are what most of these rows pin
// down. It keeps the LAST index equal to the maximum, not the first. And because MAX(a,b) is
// (a > b ? a : b) and a > NaN is false, a NaN BECOMES the running maximum and is then displaced
// by the next element -- so a NaN discards the maximum before it, and the answer is decided by
// the elements after the last NaN rather than being "the largest non-NaN element".


#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
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

/// A row of n values -(i % 7) (all below 1), with the given (column, value) pairs set.
test_row sparse_row(const char * name, std::size_t n,
                    std::initializer_list<std::pair<std::size_t, float>> set) {
    test_row row{name, std::vector<float>(n)};
    for (std::size_t i = 0; i < n; ++i) {
        row.values[i] = -static_cast<float>(i % 7);
    }
    for (const auto & [col, value] : set) {
        row.values[col] = value;
    }
    return row;
}

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

    // 10 wide: the MNIST class count.
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
        // A NaN discards the maximum before it: MAX(a, NaN) is NaN because a > NaN is
        // false, and the next element then displaces the NaN. So these do NOT report the
        // largest non-NaN element -- the answer comes from after the last NaN.
        {"NaN discards earlier max", {1.f, kNaN, -1.f, -2.f, -3.f, -4.f, -5.f, -6.f, -7.f, -8.f}},
        {"two NaNs, max before 2nd", {9.f, kNaN, 1.f, 2.f, kNaN, 3.f, 4.f, 5.f, 6.f, 7.f}},
        {"trailing NaN", {0.f, 1.f, 2.f, 3.f, 9.f, 3.f, 2.f, 1.f, 0.f, kNaN}},
        {"all NaN", {kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}},
        {"+inf at 6", {0.f, 1.f, 2.f, 3.f, 4.f, 3.f, kInf, 1.f, 0.f, -1.f}},
        {"all -inf", {-kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf, -kInf}},
        {"negatives only", {-5.f, -4.f, -3.f, -2.f, -1.5f, -2.f, -3.f, -4.f, -5.f, -6.f}},
        {"signed zeros", {-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f}},
        // -0 == +0, so -0 after a +0 maximum ties it and takes the index. "signed zeros" above
        // ends in +0 and so cannot tell a kernel that orders -0 below +0 from one that doesn't.
        {"-0 ties +0 max", {0.f, -0.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f}},
    };

    // 16 wide: a different row length, to check nothing depends on the MNIST shape.
    const std::vector<test_row> rows16 = {
        {"16-wide max at 11",
         {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 99.f, 10.f, 9.f, 8.f, 7.f}},
        {"16-wide tie 3 and 12",
         {0.f, 1.f, 2.f, 9.f, 4.f, 5.f, 6.f, 7.f, 8.f, 3.f, 2.f, 1.f, 9.f, 0.f, -1.f, -2.f}},
        {"16-wide NaN at 0",
         {kNaN, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f}},
    };

    // 3 wide, 4 rows: the tile holds fewer elements than a vector, so every row takes the
    // per-row search with loads cut at the tile's end, which the wider rows reach only for
    // their last row.
    const std::vector<test_row> rows3 = {
        {"3-wide max at 1", {0.f, 9.f, 1.f}},
        {"3-wide tie 0 and 2", {9.f, 1.f, 9.f}},
        {"3-wide NaN discards max", {9.f, kNaN, -1.f}},
        {"3-wide -0 ties +0 max", {0.f, -0.f, -1.f}},
    };

    // 40 wide: two whole vectors and a tail of 8 per row, so a lane sees several columns.
    const std::vector<test_row> rows40 = {
        sparse_row("40-wide max at 0", 40, {{0, 9.f}}),
        sparse_row("40-wide max in tail at 35", 40, {{35, 9.f}}),
        sparse_row("40-wide tie 3, 19 (one lane)", 40, {{3, 9.f}, {19, 9.f}}),
        sparse_row("40-wide tie 5, 26 (two lanes)", 40, {{5, 9.f}, {26, 9.f}}),
        sparse_row("40-wide tie 30 and 38 (tail)", 40, {{30, 9.f}, {38, 9.f}}),
        sparse_row("40-wide -0 at 39 ties +0", 40, {{2, 0.f}, {39, -0.f}}),
        sparse_row("40-wide NaN at 20 discards", 40, {{2, 9.f}, {20, kNaN}}),
        sparse_row("40-wide NaN in tail at 33", 40, {{2, 9.f}, {33, kNaN}, {36, 0.5f}}),
        {"40-wide all -inf", std::vector<float>(40, -kInf)},
    };

    // 32 wide: whole vectors and no tail.
    const std::vector<test_row> rows32 = {
        sparse_row("32-wide tie 15 and 16", 32, {{15, 9.f}, {16, 9.f}}),
        sparse_row("32-wide max at 31", 32, {{31, 9.f}}),
        sparse_row("32-wide NaN at 31", 32, {{4, 9.f}, {31, kNaN}}),
    };

    // 1 wide: every row's answer is 0, whatever it holds.
    const std::vector<test_row> rows1 = {
        {"1-wide 3", {3.f}},      {"1-wide NaN", {kNaN}}, {"1-wide -inf", {-kInf}},
        {"1-wide -0", {-0.f}},    {"1-wide +inf", {kInf}}, {"1-wide -2", {-2.f}},
    };

    // 16 and 13 wide, 20 rows: a tile of whole rows, so on aie2p the first 16 rows are reduced
    // together as a tree that pairs adjacent column runs, 1 | 1, 2 | 2, 4 | 4, 8 | 8. Each tie
    // below straddles one of those pairings, and the reference wants the later column.
    const auto tree_rows = [](std::size_t n) {
        std::vector<test_row> rows = {
            sparse_row("tie 0, 1", n, {{0, 9.f}, {1, 9.f}}),
            sparse_row("tie 1, 2", n, {{1, 9.f}, {2, 9.f}}),
            sparse_row("tie 3, 4", n, {{3, 9.f}, {4, 9.f}}),
            sparse_row("tie 2, 5", n, {{2, 9.f}, {5, 9.f}}),
            sparse_row("tie 7, 8", n, {{7, 9.f}, {8, 9.f}}),
            sparse_row("tie 0, 8", n, {{0, 9.f}, {8, 9.f}}),
            sparse_row("tie 6, 9, 11", n, {{6, 9.f}, {9, 9.f}, {11, 9.f}}),
            sparse_row("tie 0 and last", n, {{0, 9.f}, {n - 1, 9.f}}),
            sparse_row("tie 4, last - 1", n, {{4, 9.f}, {n - 2, 9.f}}),
            sparse_row("max at last", n, {{n - 1, 9.f}}),
            sparse_row("max at 0", n, {{0, 9.f}}),
            sparse_row("-0 at 12 ties +0", n, {{3, 0.f}, {12, -0.f}}),
            sparse_row("+0 at 12 ties -0", n, {{3, -0.f}, {12, 0.f}}),
            sparse_row("-NaN at 5 discards", n, {{2, 9.f}, {5, -kNaN}}),
            {"all -inf", std::vector<float>(n, -kInf)},
            {"all equal", std::vector<float>(n, 5.f)},
            // Rows 16-19: past the tree's block.
            sparse_row("NaN at 1 discards", n, {{0, 9.f}, {1, kNaN}}),
            sparse_row("tie 1, 2 (row 17)", n, {{1, 9.f}, {2, 9.f}}),
            sparse_row("+inf at 10", n, {{10, kInf}}),
            sparse_row("tie 5, 6 (row 19)", n, {{5, 9.f}, {6, 9.f}}),
        };
        return rows;
    };
    const std::vector<test_row> rows16x20 = tree_rows(16);
    const std::vector<test_row> rows13x20 = tree_rows(13);

    // 100 wide: 6 whole vectors and a tail of 4, so a row runs the whole-vector loop at its
    // promised minimum trip count and over.
    const std::vector<test_row> rows100 = {
        sparse_row("100-wide max at 0", 100, {{0, 9.f}}),
        sparse_row("100-wide max at 95 (whole)", 100, {{95, 9.f}}),
        sparse_row("100-wide max in tail at 98", 100, {{98, 9.f}}),
        sparse_row("100-wide tie 3, 83 (one lane)", 100, {{3, 9.f}, {83, 9.f}}),
        sparse_row("100-wide tie 70, 5 (two lanes)", 100, {{5, 9.f}, {70, 9.f}}),
        sparse_row("100-wide tie 17, 97 (tail)", 100, {{17, 9.f}, {97, 9.f}}),
        sparse_row("100-wide -0 at 64 ties +0", 100, {{2, 0.f}, {64, -0.f}}),
        sparse_row("100-wide NaN at 50 discards", 100, {{2, 9.f}, {50, kNaN}}),
        sparse_row("100-wide -NaN at 80", 100, {{90, 9.f}, {80, -kNaN}, {85, 0.5f}}),
        {"100-wide all -inf", std::vector<float>(100, -kInf)},
    };

    bool all_ok = true;
    printf("ARGMAX 10-wide:\n");
    all_ok = run_rows(backend, rows10) && all_ok;
    printf("ARGMAX 16-wide:\n");
    all_ok = run_rows(backend, rows16) && all_ok;
    printf("ARGMAX 3-wide:\n");
    all_ok = run_rows(backend, rows3) && all_ok;
    printf("ARGMAX 40-wide:\n");
    all_ok = run_rows(backend, rows40) && all_ok;
    printf("ARGMAX 32-wide:\n");
    all_ok = run_rows(backend, rows32) && all_ok;
    printf("ARGMAX 1-wide:\n");
    all_ok = run_rows(backend, rows1) && all_ok;
    printf("ARGMAX 16-wide x 20:\n");
    all_ok = run_rows(backend, rows16x20) && all_ok;
    printf("ARGMAX 13-wide x 20:\n");
    all_ok = run_rows(backend, rows13x20) && all_ok;
    printf("ARGMAX 100-wide:\n");
    all_ok = run_rows(backend, rows100) && all_ok;

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
