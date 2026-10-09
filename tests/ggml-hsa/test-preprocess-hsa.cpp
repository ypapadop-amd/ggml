// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the source pre-processing / output post-processing paths of
// ggml_backend_hsa_tensor_extra. The device has no f16 kernels, so an op with f16 operands runs as
// bf16 internally: each f16 source is converted into its own internal buffer before the dispatch
// (per source) and the bf16 result is converted back into the f16 parent afterwards
// (node.transform == convert).
//
// Both source paths are exercised. With an even element count the HSA_CONVERT kernel builds and the
// conversion is dispatched on the device queue (a preprocess kernel); with an odd element count the
// kernel legitimately declines -- a 2-byte tensor is then not a whole number of DMA words -- and
// the conversion falls back to the host copy path (no kernel). Both must produce identical, correct
// results; that fallback is the point of the test.
//
// The padded-GEMM MUL_MAT has its own host fallback: HSA_CONVERT_PAD declines an f16 source, so f16
// operands are scattered into their zero-padded bf16 internal buffers on the host. The internal
// buffers are not zeroed at allocation, so that scatter must write the padding itself. Fresh device
// allocations happen to read as zero, so the case first dirties the storage: a larger MUL_MAT with
// no zero inputs runs on the same allocator, and the padded case's extra, recycled from it, takes
// over its internal storage, so a scatter that skips the padding leaves nonzero values there.
//
// Inputs are small integers, which are exact in f16 and in bf16, and so are their sums. That makes
// the expected result an exact integer regardless of how many times the value is round-tripped
// through the two formats, so the comparison can be exact rather than tolerance-based.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"
#include "hsa-test-common.hpp"

namespace {

using hsa_test::load_val;
using hsa_test::store_val;

enum class case_result { pass, fail, skip };

// Values chosen to be exactly representable in bf16 (8 significand bits), so no rounding occurs
// anywhere along f16 -> bf16 -> add -> bf16 -> f16.
float src0_val(int64_t i) { return static_cast<float>(i % 61) - 30.0f; }
float src1_val(int64_t i) { return static_cast<float>(i % 29) - 14.0f; }

// b defaults to a's shape; a smaller b ([b_d0, b_d1], each dividing a's) is broadcast. Each
// source's path is checked separately, so a and b may take different ones.
case_result run_case(ggml_backend_t backend,
                     int64_t d0,
                     int64_t d1,
                     bool expect_on_queue,
                     int64_t b_d0 = 0,
                     int64_t b_d1 = 0,
                     int expect_b_on_queue = -1) {
    b_d0 = b_d0 != 0 ? b_d0 : d0;
    b_d1 = b_d1 != 0 ? b_d1 : d1;
    const bool expect_b = expect_b_on_queue < 0 ? expect_on_queue : expect_b_on_queue != 0;

    const std::size_t ctx_size = 5 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, d0, d1);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, b_d0, b_d1);
    ggml_set_name(b, "b");
    ggml_tensor * dst = ggml_add(ctx.get(), a, b);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    // Which path the sources take is not observable from the result: the host copy and the
    // on-queue convert produce identical values, so the numerical check below would pass even if
    // the device path never ran. Assert the thing that selects the path instead. A source runs on
    // the device queue exactly when it has a preprocess kernel, and that kernel is
    // the element-wise f16 -> bf16 convert built for this source; probing the same transform as a
    // standalone op therefore reports exactly whether that kernel builds.
    for (auto [src, expect] : {std::pair{a, expect_on_queue}, std::pair{b, expect_b}}) {
        ggml_tensor * convert_probe = ggml_hsa_convert(ctx.get(), src, GGML_TYPE_BF16);
        ggml_set_name(convert_probe, "convert_probe");
        const bool on_queue = ggml_backend_supports_op(backend, convert_probe);
        if (on_queue != expect) {
            printf("  expected %s pre-processing of %s, got %s\n", expect ? "on-queue" : "host",
                   ggml_get_name(src), on_queue ? "on-queue" : "host");
            return case_result::fail;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    const int64_t n = d0 * d1;
    std::vector<uint8_t> a_bytes(ggml_nbytes(a));
    std::vector<uint8_t> b_bytes(ggml_nbytes(b));
    for (int64_t i = 0; i < n; ++i) {
        store_val(GGML_TYPE_F16, a_bytes.data(), i, src0_val(i));
    }
    for (int64_t i = 0; i < b_d0 * b_d1; ++i) {
        store_val(GGML_TYPE_F16, b_bytes.data(), i, src1_val(i));
    }
    ggml_backend_tensor_set(a, a_bytes.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bytes.data(), 0, ggml_nbytes(b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        const int64_t j = (i % d0) % b_d0 + ((i / d0) % b_d1) * b_d0;
        const float want = src0_val(i) + src1_val(j);
        const float got = load_val(GGML_TYPE_F16, dst_bytes.data(), i);
        if (got != want) {
            printf("  mismatch at %lld: got %g want %g\n", (long long)i, got, want);
            ok = false;
        }
    }

    return ok ? case_result::pass : case_result::fail;
}

// f16 [K, M] x f16 [K, N] -> f32 [M, N] MUL_MAT on the padded-GEMM path, with K, M and N off the
// tile multiples so every operand is padded in both dimensions. Both operands take the host
// fallback (asserted), so the padding gaps the GEMM reads are written only by the host scatter.
case_result run_mul_mat_case(ggml_backend_t backend, int64_t K, int64_t M, int64_t N) {
    const std::size_t ctx_size = 8 * ggml_tensor_overhead() + 2 * ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, K, M);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, K, N);
    ggml_set_name(b, "b");
    ggml_tensor * dst = ggml_mul_mat(ctx.get(), a, b);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    // As above, assert the path rather than infer it: the operand pre-processing is on-queue
    // exactly when HSA_CONVERT_PAD builds for the operand, and it declines an f16 source whatever
    // the padded shape, so probing any padded shape reports the path.
    for (ggml_tensor * src : {a, b}) {
        ggml_tensor * probe = ggml_hsa_convert_pad(ctx.get(), src, GGML_TYPE_BF16, GGML_PAD(K, 8),
                                                   GGML_PAD(src->ne[1], 64));
        ggml_set_name(probe, "convert_pad_probe");
        if (ggml_backend_supports_op(backend, probe)) {
            printf("  expected host pre-processing of %s, got on-queue\n", ggml_get_name(src));
            return case_result::fail;
        }
    }

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};

    // Dirty the internal storage first: a MUL_MAT at tile multiples on aie2 and aie2p, with no zero
    // inputs, whose operands' bf16 copies alone outsize all of the padded case's internal storage.
    // It is larger, so the allocator keeps its buffer for the padded case and recycles its extras;
    // the padded MUL_MAT takes the same extra slot and reuses its storage.
    {
        const int64_t dK = 64, dM = 512, dN = 128;
        ggml_tensor * da = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, dK, dM);
        ggml_tensor * db = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, dK, dN);
        ggml_tensor * ddst = ggml_mul_mat(ctx.get(), da, db);
        if (!ggml_backend_supports_op(backend, ddst)) {
            printf("  dirtying MUL_MAT not supported\n");
            return case_result::fail;
        }
        ggml_cgraph * dgf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(dgf, ddst);
        if (!ggml_gallocr_alloc_graph(galloc.get(), dgf)) {
            printf("  dirtying graph allocation failed\n");
            return case_result::fail;
        }
        for (ggml_tensor * t : {da, db}) {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                store_val(GGML_TYPE_F16, bytes.data(), i, static_cast<float>(i % 7 + 1));
            }
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        if (ggml_backend_graph_compute(backend, dgf) != GGML_STATUS_SUCCESS) {
            printf("  dirtying graph compute failed\n");
            return case_result::fail;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> a_bytes(ggml_nbytes(a));
    std::vector<uint8_t> b_bytes(ggml_nbytes(b));
    for (int64_t i = 0; i < K * M; ++i) {
        store_val(GGML_TYPE_F16, a_bytes.data(), i, src0_val(i));
    }
    for (int64_t i = 0; i < K * N; ++i) {
        store_val(GGML_TYPE_F16, b_bytes.data(), i, src1_val(i));
    }
    ggml_backend_tensor_set(a, a_bytes.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_bytes.data(), 0, ggml_nbytes(b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<float> dst_host(M * N);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // small integers: every product and partial sum is exact in bf16 inputs / f32 accumulation
    bool ok = true;
    for (int64_t n = 0; n < N && ok; ++n) {
        for (int64_t m = 0; m < M && ok; ++m) {
            float want = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                want += src0_val(m * K + k) * src1_val(n * K + k);
            }
            const float got = dst_host[n * M + m];
            if (got != want) {
                printf("  mismatch at [%lld,%lld]: got %g want %g\n", (long long)m, (long long)n,
                       got, want);
                ok = false;
            }
        }
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

    struct {
        int64_t d0, d1;
        const char * name;
        bool expect_on_queue;
    } cases[] = {
        {64, 8, "even numel (on-queue convert)", true},
        {256, 4, "even numel, larger", true},
        {1, 64, "single col", true},
        // Odd element count: a 2-byte tensor is not a whole number of DMA words, so HSA_CONVERT
        // declines and the source conversion falls back to the host copy path.
        {15, 1, "odd numel (host fallback)", false},
        {33, 3, "odd numel rows (host fallback)", false},
    };

    // Mixed paths in one node: a converts on-queue, the broadcast b (odd numel) on the host.
    // Exercises the lazy drain before the first host copy.
    struct {
        int64_t d0, d1, b_d0, b_d1;
        const char * name;
    } mixed_cases[] = {
        {6, 2, 3, 1, "a on-queue, b host"},
        {30, 4, 15, 1, "a on-queue, b host, larger"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.d0, c.d1, c.expect_on_queue);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("ADD f16 %-32s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    for (const auto & c : mixed_cases) {
        const case_result r = run_case(backend, c.d0, c.d1, true, c.b_d0, c.b_d1, 0);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("ADD f16 %-32s: %s\n", c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    {
        const case_result r = run_mul_mat_case(backend, 37, 20, 10);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f16 %-28s: %s\n", "padded GEMM (host scatter)", label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    ggml_backend_free(backend);
    if (any_fail) {
        printf("FAILURES\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
