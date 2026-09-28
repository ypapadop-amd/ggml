// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Zero-copy sharing between the HIP and HSA backends. A HIP buffer imported with
// ggml_backend_hsa_buffer_import is the same memory seen from the NPU: a write through one backend
// is visible through the other, at a different address, with no copy in between.
//
// Needs the HIP and HSA backends on the same ROCR, and a warm kernel cache for SUB f32 [32]; see
// "Running HIP and HSA in the Same Process" in src/ggml-hsa/README.md.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

constexpr int64_t N = 32;

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what);
    failures += ok ? 0 : 1;
}

// no_alloc context with room for a few tensors and two small graphs
ggml_context * make_context() {
    const ggml_init_params params = {
        /*.mem_size   =*/16 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    return ggml_init(params);
}

std::vector<float> pattern(float start) {
    std::vector<float> v(N);
    std::iota(v.begin(), v.end(), start);
    return v;
}

// Address of @p t as seen through @p imported, the HSA buffer imported from t->buffer.
float * imported_address(ggml_backend_buffer_t imported, const ggml_tensor * t) {
    const auto offset = static_cast<const char *>(t->data) -
                        static_cast<const char *>(ggml_backend_buffer_get_base(t->buffer));
    return reinterpret_cast<float *>(static_cast<char *>(ggml_backend_buffer_get_base(imported)) +
                                     offset);
}

// Unallocated tensor in @p ctx with the type, shape and strides of @p src.
ggml_tensor * like(ggml_context * ctx, const ggml_tensor * src) {
    ggml_tensor * t = ggml_new_tensor(ctx, src->type, GGML_MAX_DIMS, src->ne);
    std::copy_n(src->nb, GGML_MAX_DIMS, t->nb);
    return t;
}

void test_import_rejects(ggml_backend_buffer_type_t hip_buft) {
    check(ggml_backend_hsa_buffer_import(0, nullptr) == nullptr, "import rejects a null buffer");

    ggml_backend_buffer_t cpu = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 256);
    check(ggml_backend_hsa_buffer_import(0, cpu) == nullptr, "import rejects a CPU buffer");
    ggml_backend_buffer_free(cpu);

    ggml_backend_buffer_t hsa =
        ggml_backend_buft_alloc_buffer(ggml_backend_hsa_buffer_type(0), 256);
    check(ggml_backend_hsa_buffer_import(0, hsa) == nullptr, "import rejects an HSA buffer");
    ggml_backend_buffer_free(hsa);

    ggml_backend_buffer_t empty = ggml_backend_buft_alloc_buffer(hip_buft, 0);
    check(ggml_backend_hsa_buffer_import(0, empty) == nullptr, "import rejects a zero-size buffer");
    ggml_backend_buffer_free(empty);

    ggml_backend_buffer_t hip = ggml_backend_buft_alloc_buffer(hip_buft, 256);
    check(ggml_backend_hsa_buffer_import(-1, hip) == nullptr, "import rejects a negative device");
    check(ggml_backend_hsa_buffer_import(ggml_backend_hsa_get_device_count(), hip) == nullptr,
          "import rejects an out-of-range device");
    ggml_backend_buffer_free(hip);
}

void test_import_aliases(ggml_backend_t hip) {
    ggml_context * ctx = make_context();
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_tensor * y = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(ctx, hip);

    // HIP places small buffers in a shared block, which the import maps as a whole; z is likely
    // (not guaranteed) such a neighbour, so the clear check below is meaningful only if it is
    ggml_context * neighbour_ctx = make_context();
    ggml_tensor * z = ggml_new_tensor_1d(neighbour_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t neighbour = ggml_backend_alloc_ctx_tensors(neighbour_ctx, hip);
    ggml_backend_tensor_set(z, pattern(500).data(), 0, ggml_nbytes(z));

    ggml_backend_buffer_t imported = ggml_backend_hsa_buffer_import(0, hip_buffer);
    check(imported != nullptr, "import a HIP buffer");
    if (imported != nullptr) {
        check(ggml_backend_buffer_get_size(imported) == ggml_backend_buffer_get_size(hip_buffer),
              "imported buffer has the HIP buffer's size");
        check(ggml_backend_buffer_get_base(imported) != ggml_backend_buffer_get_base(hip_buffer),
              "imported buffer is at a different address than the HIP buffer");

        std::vector<float> out(N);

        const auto in = pattern(1000);
        std::memcpy(imported_address(imported, x), in.data(), ggml_nbytes(x));
        ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
        check(out == in, "store through the imported address is read by HIP");

        const auto in2 = pattern(-50);
        ggml_backend_tensor_set(y, in2.data(), 0, ggml_nbytes(y));
        check(std::memcmp(imported_address(imported, y), in2.data(), ggml_nbytes(y)) == 0,
              "HIP write is seen through the imported address");

        ggml_backend_buffer_clear(imported, 0);
        ggml_backend_tensor_get(x, out.data(), 0, ggml_nbytes(x));
        check(out == std::vector<float>(N, 0.0f), "clearing the imported buffer clears HIP memory");
        ggml_backend_tensor_get(z, out.data(), 0, ggml_nbytes(z));
        check(out == pattern(500), "clearing the imported buffer leaves other HIP buffers alone");

        ggml_backend_buffer_free(imported);
    }

    ggml_backend_buffer_free(neighbour);
    ggml_free(neighbour_ctx);
    ggml_backend_buffer_free(hip_buffer);
    ggml_free(ctx);
}

void test_alias_rejects(ggml_backend_t hip) {
    ggml_context * ctx = make_context();
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(ctx, hip);
    ggml_context * other_ctx = make_context();
    ggml_tensor * w = ggml_new_tensor_1d(other_ctx, GGML_TYPE_F32, N);
    ggml_backend_buffer_t other_buffer = ggml_backend_alloc_ctx_tensors(other_ctx, hip);
    ggml_backend_buffer_t not_imported =
        ggml_backend_buft_alloc_buffer(ggml_backend_hsa_buffer_type(0), 256);

    ggml_backend_buffer_t imported = ggml_backend_hsa_buffer_import(0, hip_buffer);
    ggml_context * hsa_ctx = make_context();

    check(ggml_backend_hsa_tensor_alloc_alias(imported, like(hsa_ctx, w), w) != GGML_STATUS_SUCCESS,
          "alloc_alias rejects a tensor from a buffer that was not imported");
    check(ggml_backend_hsa_tensor_alloc_alias(nullptr, like(hsa_ctx, x), x) != GGML_STATUS_SUCCESS,
          "alloc_alias rejects a null imported buffer");
    check(ggml_backend_hsa_tensor_alloc_alias(not_imported, like(hsa_ctx, x), x) !=
              GGML_STATUS_SUCCESS,
          "alloc_alias rejects an HSA buffer that is not an import");

    ggml_tensor * wrong_shape = ggml_new_tensor_1d(hsa_ctx, GGML_TYPE_F32, N / 2);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, wrong_shape, x) != GGML_STATUS_SUCCESS,
          "alloc_alias rejects a different shape");
    ggml_tensor * wrong_type = ggml_new_tensor_1d(hsa_ctx, GGML_TYPE_I32, N);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, wrong_type, x) != GGML_STATUS_SUCCESS,
          "alloc_alias rejects a different type");
    ggml_tensor * placed = ggml_new_tensor_1d(hsa_ctx, GGML_TYPE_F32, N);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, placed, x) == GGML_STATUS_SUCCESS,
          "alloc_alias places a matching tensor");
    check(ggml_backend_hsa_tensor_alloc_alias(imported, placed, x) != GGML_STATUS_SUCCESS,
          "alloc_alias rejects a tensor that is already allocated");

    ggml_free(hsa_ctx);
    ggml_backend_buffer_free(imported);
    ggml_backend_buffer_free(not_imported);
    ggml_backend_buffer_free(other_buffer);
    ggml_free(other_ctx);
    ggml_backend_buffer_free(hip_buffer);
    ggml_free(ctx);
}

void test_alias_view(ggml_backend_t hip) {
    ggml_context * ctx = make_context();
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_tensor * view = ggml_view_1d(ctx, x, 8, 4 * sizeof(float)); // elements 4..11
    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(ctx, hip);
    const auto in = pattern(0);
    ggml_backend_tensor_set(x, in.data(), 0, ggml_nbytes(x));

    ggml_backend_buffer_t imported = ggml_backend_hsa_buffer_import(0, hip_buffer);
    ggml_context * hsa_ctx = make_context();
    ggml_tensor * alias = like(hsa_ctx, view);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, alias, view) == GGML_STATUS_SUCCESS &&
              std::memcmp(alias->data, in.data() + 4, 8 * sizeof(float)) == 0,
          "alias of a view starts at the view's offset");

    ggml_free(hsa_ctx);
    ggml_backend_buffer_free(imported);
    ggml_backend_buffer_free(hip_buffer);
    ggml_free(ctx);
}

// HIP computes s = a + b; the NPU reads s in place and writes s - c straight into the HIP tensor d.
void test_pipeline(ggml_backend_t hip, ggml_backend_t hsa) {
    constexpr int iterations = 5;

    ggml_context * hip_ctx = make_context();
    ggml_tensor * a = ggml_new_tensor_1d(hip_ctx, GGML_TYPE_F32, N);
    ggml_tensor * b = ggml_new_tensor_1d(hip_ctx, GGML_TYPE_F32, N);
    ggml_tensor * s = ggml_add(hip_ctx, a, b);
    ggml_tensor * d = ggml_new_tensor_1d(hip_ctx, GGML_TYPE_F32, N);
    // a HIP kernel consumes the NPU result; from the second iteration on it has read d before
    ggml_tensor * e = ggml_add(hip_ctx, d, a);
    ggml_backend_buffer_t hip_buffer = ggml_backend_alloc_ctx_tensors(hip_ctx, hip);
    ggml_cgraph * hip_graph = ggml_new_graph(hip_ctx);
    ggml_build_forward_expand(hip_graph, s);
    ggml_cgraph * hip_consumer_graph = ggml_new_graph(hip_ctx);
    ggml_build_forward_expand(hip_consumer_graph, e);

    ggml_backend_buffer_t imported = ggml_backend_hsa_buffer_import(0, hip_buffer);
    ggml_context * hsa_ctx = make_context();
    ggml_tensor * s_npu = like(hsa_ctx, s);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, s_npu, s) == GGML_STATUS_SUCCESS,
          "place an NPU input on a HIP tensor");
    ggml_tensor * c = ggml_new_tensor_1d(hsa_ctx, GGML_TYPE_F32, N);
    ggml_tensor * r = ggml_sub(hsa_ctx, s_npu, c);
    check(ggml_backend_hsa_tensor_alloc_alias(imported, r, d) == GGML_STATUS_SUCCESS,
          "place the NPU result on a HIP tensor");
    ggml_backend_buffer_t hsa_buffer = ggml_backend_alloc_ctx_tensors(hsa_ctx, hsa); // only c
    ggml_cgraph * hsa_graph = ggml_new_graph(hsa_ctx);
    ggml_build_forward_expand(hsa_graph, r);

    const auto va = pattern(11);
    const auto vc = pattern(10);
    ggml_backend_tensor_set(a, va.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(c, vc.data(), 0, ggml_nbytes(c));

    int bad_iterations = 0;
    int bad_consumer_iterations = 0;
    std::vector<float> out(N);
    for (int k = 1; k <= iterations; ++k) {
        const auto vb = pattern(100.0f * k);
        ggml_backend_tensor_set(b, vb.data(), 0, ggml_nbytes(b));
        const bool computed = ggml_backend_graph_compute(hip, hip_graph) == GGML_STATUS_SUCCESS &&
                              ggml_backend_graph_compute(hsa, hsa_graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(d, out.data(), 0, ggml_nbytes(d)); // read back by HIP
        bool ok = computed;
        for (int64_t i = 0; i < N; ++i) {
            ok = ok && out[i] == va[i] + vb[i] - vc[i];
        }
        bad_iterations += ok ? 0 : 1;

        bool consumer_ok =
            ggml_backend_graph_compute(hip, hip_consumer_graph) == GGML_STATUS_SUCCESS;
        ggml_backend_tensor_get(e, out.data(), 0, ggml_nbytes(e));
        for (int64_t i = 0; i < N; ++i) {
            consumer_ok = consumer_ok && out[i] == (va[i] + vb[i] - vc[i]) + va[i];
        }
        bad_consumer_iterations += consumer_ok ? 0 : 1;
    }
    check(bad_iterations == 0, "HIP -> NPU -> HIP on one import, 5 iterations");
    check(bad_consumer_iterations == 0, "a HIP op consumes the NPU result, 5 iterations");

    ggml_backend_buffer_free(hsa_buffer);
    ggml_free(hsa_ctx);
    ggml_backend_buffer_free(imported);
    ggml_backend_buffer_free(hip_buffer);
    ggml_free(hip_ctx);
}

} // namespace

int main() {
    ggml_backend_t hip = ggml_backend_cuda_init(0);
    ggml_backend_t hsa = ggml_backend_hsa_init(0);
    if (hip == nullptr || hsa == nullptr) {
        std::fprintf(stderr, "need both a HIP and an HSA device\n");
        return EXIT_FAILURE;
    }

    test_import_rejects(ggml_backend_get_default_buffer_type(hip));
    test_import_aliases(hip);
    test_alias_rejects(hip);
    test_alias_view(hip);
    test_pipeline(hip, hsa);

    ggml_backend_free(hsa);
    ggml_backend_free(hip);

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
