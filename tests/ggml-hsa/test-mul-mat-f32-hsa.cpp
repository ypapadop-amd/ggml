// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for f32 x f32 MUL_MAT at shapes that are not tile multiples. On aie2 the GEMM
// streams an f32 B unpadded and converts it on the core, and writes C in place when it is tall
// enough: it shifts its last column group and row block back to end at N and M, and zeroes the K
// tail the last K tile reads past each B column. Narrow or short shapes run over padding instead and
// clip the result to the dense destination. Each case checks the whole result against a host reference; inputs are small
// integers, so every product and partial sum is exact in bf16/f32 and the result must match bit
// for bit.

#include <cmath>
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

enum class case_result { pass, fail, skip };

// Runs C = A x B for A [K, M] and B [K, N] (ggml layout). With @p nan_col >= 0, B's first element
// in that column is NaN: that column's result is NaN, and no other column may be affected -- the
// last K tile of column nan_col - 1 reads it past its own end and must zero it.
case_result run_case(ggml_backend_t backend, int64_t M, int64_t N, int64_t K, int64_t nan_col) {
    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_set_name(b, "b");
    ggml_set_input(b);
    ggml_tensor * c = ggml_mul_mat(ctx.get(), a, b); // [M, N] f32
    ggml_set_name(c, "c");

    if (!ggml_backend_supports_op(backend, c)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    // Different periods per operand keep neighbouring rows/columns distinct, so a result written
    // to or read from the wrong row or column cannot coincide with the right value.
    std::vector<float> fa(M * K);
    std::vector<float> fb(N * K);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            fa[i * K + k] = static_cast<float>((i * 7 + k * 3) % 5 - 2);
        }
    }
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            fb[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
    }
    if (nan_col >= 0) {
        fb[nan_col * K] = std::numeric_limits<float>::quiet_NaN();
    }
    ggml_backend_tensor_set(a, fa.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, fb.data(), 0, ggml_nbytes(b));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<float> got(M * N);
    ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));

    int64_t mismatches = 0;
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t i = 0; i < M; ++i) {
            float want = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                want += fa[i * K + k] * fb[j * K + k];
            }
            const float g = got[j * M + i];
            const bool same = std::isnan(want) ? std::isnan(g) : g == want;
            if (!same) {
                if (mismatches < 8) {
                    printf("  mismatch at row %lld col %lld: got %g want %g\n", (long long)i,
                           (long long)j, g, want);
                }
                ++mismatches;
            }
        }
    }
    if (mismatches != 0) {
        printf("  %lld / %lld elements mismatched\n", (long long)mismatches, (long long)(M * N));
        return case_result::fail;
    }
    return case_result::pass;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    // On aie2 a column group and a row block are at least 64 wide/tall, and K pads to 8.
    struct {
        int64_t M, N, K;
        int64_t nan_col;
        const char * name;
    } cases[] = {
        // tile multiples: nothing to shift, pad or clip
        {512, 512, 512, -1, "aligned"},
        // shifted last column group and row block, C written in place
        {500, 500, 784, -1, "mnist fc1"},
        {100, 64, 256, -1, "shifted row block"},
        {129, 300, 1000, -1, "shifted rows and cols"},
        {1000, 70, 40, -1, "many row blocks"},
        // K tail zeroed on the core, including the last column's read into the buffer slack
        {64, 100, 500, -1, "K tail"},
        {64, 70, 257, -1, "K tail, odd K"},
        {64, 100, 500, 50, "K tail, NaN in next col"},
        // M below one row block: B unpadded, last row block clipped
        {10, 500, 500, -1, "mnist fc2"},
        {33, 257, 129, -1, "short M"},
        // N below one column group: B padded, last column group clipped
        {300, 8, 256, -1, "narrow N"},
        // more than 64 column groups for the largest-volume tile (8x256x32 on aie2p): a shim BD
        // iterates at most 64 times, so the tile must be chosen to keep the group count at most 64
        {32, 16640, 512, -1, "many column groups"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        const case_result r = run_case(backend, c.M, c.N, c.K, c.nan_col);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 %4lldx%4lldx%4lld %-24s: %s\n", (long long)c.M, (long long)c.N,
               (long long)c.K, c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }
    ggml_backend_free(backend);

    if (any_fail) {
        printf("SOME FAILED\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
