// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for f32 x f32 MUL_MAT at shapes that are not tile multiples. The GEMM streams an
// f32 B unpadded and converts it on the core, and writes C in place when it is tall enough: it
// shifts its last column group and row block back to end at N and M, and zeroes the K tail the last
// K tile reads past each B column. Narrow or short shapes fall back to padding and de-padding. Each
// case checks the whole result against a host reference; inputs are small integers, so every
// product and partial sum is exact in bf16/f32 and the result must match bit for bit. Two more
// cases place B after C, so the kernel is chosen before B's buffer is known, and check that a B
// without read slack is refused rather than read past.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

enum class case_result { pass, fail, skip };

// Fills A [K, M] and B [K, N] with small integers. Different periods per operand keep neighbouring
// rows/columns distinct, so a result written to or read from the wrong row or column cannot
// coincide with the right value.
void fill_operands(std::vector<float> & fa, std::vector<float> & fb, int64_t M, int64_t N,
                   int64_t K) {
    fa.resize(M * K);
    fb.resize(N * K);
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
}

// Checks C [M, N] against A x B bit for bit (a NaN must stay a NaN).
case_result check_result(const std::vector<float> & got,
                         const std::vector<float> & fa,
                         const std::vector<float> & fb,
                         int64_t M,
                         int64_t N,
                         int64_t K) {
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

    std::vector<float> fa;
    std::vector<float> fb;
    fill_operands(fa, fb, M, N, K);
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

    return check_result(got, fa, fb, M, N, K);
}

// The GEMM's kernel is chosen when C is placed (init_tensor), from B's buffer at that time. Placing
// A and C before B, which has no buffer yet, selects the kernel that reads an f32 B unpadded and
// reads its last K tile past each column. Then B is placed: in an HSA buffer, whose allocation has
// read slack, the GEMM must compute C; in a buffer without slack, graph_compute must refuse to
// dispatch it rather than read past B. The test uses a CPU buffer for that, standing in for one
// imported from another device (which needs HIP). The device cannot reach a CPU buffer at all, so
// a failed compute alone proves nothing; what shows the refusal is that nothing was dispatched and
// the backend still runs a GEMM afterwards, where a dispatch would have suspended its queue. K = 500
// is not a multiple of 8, and N = 200 is at least one column group on aie2 and aie2p.
case_result run_out_of_order_case(ggml_backend_t backend, bool b_in_hsa_buffer) {
    const int64_t M = 64;
    const int64_t N = 200;
    const int64_t K = 500;

    const std::size_t ctx_size = 3 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_mul_mat(ctx.get(), a, b);
    if (!ggml_backend_supports_op(backend, c)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    auto alloc = [](ggml_backend_buffer_type_t buft, std::initializer_list<ggml_tensor *> tensors) {
        std::size_t size = 0;
        for (ggml_tensor * t : tensors) {
            size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t),
                             ggml_backend_buft_get_alignment(buft));
        }
        buffer_ptr buf{ggml_backend_buft_alloc_buffer(buft, size), ggml_backend_buffer_free};
        if (buf != nullptr) {
            ggml_tallocr talloc = ggml_tallocr_new(buf.get());
            for (ggml_tensor * t : tensors) {
                if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
                    buf.reset();
                    break;
                }
            }
        }
        return buf;
    };

    ggml_backend_buffer_type_t hsa_buft = ggml_backend_get_default_buffer_type(backend);
    buffer_ptr ac_buf = alloc(hsa_buft, {a, c});
    buffer_ptr b_buf = alloc(b_in_hsa_buffer ? hsa_buft : ggml_backend_cpu_buffer_type(), {b});
    if (ac_buf == nullptr || b_buf == nullptr) {
        printf("  allocation failed\n");
        return case_result::fail;
    }

    std::vector<float> fa;
    std::vector<float> fb;
    fill_operands(fa, fb, M, N, K);
    ggml_backend_tensor_set(a, fa.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, fb.data(), 0, ggml_nbytes(b));

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);
    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (!b_in_hsa_buffer) {
        if (status == GGML_STATUS_SUCCESS) {
            printf("  graph compute dispatched a GEMM that reads past B's buffer\n");
            return case_result::fail;
        }
        if (run_case(backend, M, N, K, -1) != case_result::pass) {
            printf("  the backend failed a GEMM after the refused one: it was dispatched\n");
            return case_result::fail;
        }
        return case_result::pass;
    }
    if (status != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }
    std::vector<float> got(M * N);
    ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));
    return check_result(got, fa, fb, M, N, K);
}

} // namespace

int main() {
    if (ggml_backend_t probe = ggml_backend_hsa_init(0); probe == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    } else {
        ggml_backend_free(probe);
    }

    // A column group is at least 64 wide on aie2 and 128 on aie2p; a row block is at least 64 tall
    // on aie2 and 32 on aie2p; K pads to 8 on both. Shapes below a column group take the padded
    // path, so the K-tail cases are repeated with N past 128 to reach the unpadded path on aie2p.
    struct {
        int64_t M, N, K;
        int64_t nan_col;
        const char * name;
    } cases[] = {
        // tile multiples: nothing to shift, pad or de-pad
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
        {64, 200, 500, -1, "K tail, wide N"},
        {64, 140, 257, -1, "K tail, odd K, wide N"},
        {64, 200, 500, 150, "K tail, NaN, wide N"},
        // a row block on aie2p but not on aie2: C in place on aie2p, de-padded on aie2
        {40, 200, 256, -1, "short M, wide N"},
        // M below one row block: B unpadded, C padded and de-padded
        {10, 500, 500, -1, "mnist fc2"},
        {33, 257, 129, -1, "short M"},
        // N below one column group: B padded as before
        {300, 8, 256, -1, "narrow N"},
        // more than 64 column groups for the largest-volume tile (8x256x32 on aie2p): a shim BD
        // iterates at most 64 times, so the tile must be chosen to keep the group count at most 64
        {32, 16640, 512, -1, "many column groups"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & c : cases) {
        // Each case runs on a backend, and so a queue, of its own: a queue holds at most 32
        // distinct kernels, and the cases together load more than that.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{
            ggml_backend_hsa_init(0), ggml_backend_free};
        const case_result r = run_case(backend.get(), c.M, c.N, c.K, c.nan_col);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 %4lldx%4lldx%4lld %-24s: %s\n", (long long)c.M, (long long)c.N,
               (long long)c.K, c.name, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
    }

    for (const bool b_in_hsa_buffer : {true, false}) {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{
            ggml_backend_hsa_init(0), ggml_backend_free};
        const case_result r = run_out_of_order_case(backend.get(), b_in_hsa_buffer);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("MUL_MAT f32 B placed after C, %-18s: %s\n",
               b_in_hsa_buffer ? "in an HSA buffer" : "without read slack", label);
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
