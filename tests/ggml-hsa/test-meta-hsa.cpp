// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Splits a MUL_MAT across the NPU and the GPU with the meta backend (ggml_backend_meta_device),
// which replaces the per-backend split buffer types: the meta device slices each weight across its
// devices, runs the graph on each, and combines the results. Three cases:
//   - A split by rows (ne[1]): each device computes a slice of C's rows, which the meta buffer
//     gathers on read.
//   - The same with an uneven split, so the slices differ in size.
//   - A and B split along K (ne[0]): each device computes a partial sum. The meta backend reduces
//     a partial result only when another node consumes it, so this case adds a residual D, as a
//     transformer does after such a MUL_MAT: C = A x B + D, with the generic AllReduce in between.
// Inputs are small integers, so every product and partial sum is exact and the result must match a
// host reference bit for bit.
//
// MUL_MAT always produces f32, so the cases above only reduce f32. The AllReduce hook the backend
// exports for the meta backend (ggml_backend_comm_allreduce_tensor) is also called directly, on one
// tensor per device, for f32, f16 and bf16, and with one device's node marked as not computed (an
// empty slice), which must count as zero.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

enum class case_result { pass, fail, skip };

// How the weights of a case are split; B is split the same way as A along K, and mirrored
// otherwise.
struct split_config {
    ggml_backend_meta_split_axis a_axis; // GGML_BACKEND_SPLIT_AXIS_1 (rows) or _0 (K)
    int64_t first;                       // elements of the split axis on the first device
    ggml_type type;                      // type of A and B; C is f32
};

// Returns the split state of the statically allocated tensors: "a" and, for a K split, "b" are
// split between the two devices, everything else ("d") is mirrored.
ggml_backend_meta_split_state get_split_state(const ggml_tensor * tensor, void * userdata) {
    const auto & config = *static_cast<const split_config *>(userdata);
    const bool is_a = std::strcmp(tensor->name, "a") == 0;
    const bool is_b = std::strcmp(tensor->name, "b") == 0;
    if (is_a || (is_b && config.a_axis == GGML_BACKEND_SPLIT_AXIS_0)) {
        ggml_backend_meta_split_state state{};
        state.axis = config.a_axis;
        state.ne[0] = config.first;
        state.ne[1] = tensor->ne[config.a_axis] - config.first;
        state.nr[0] = 1;
        state.n_segments = 1;
        return state;
    }
    ggml_backend_meta_split_state state{};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    state.nr[0] = 1;
    state.n_segments = 1;
    return state;
}

// Fills A [K, M], B [K, N] and D [M, N] with small integers. Different periods per operand keep neighbouring
// rows/columns distinct, so a slice placed at the wrong offset cannot coincide with the right value.
void fill_operands(std::vector<float> & fa,
                   std::vector<float> & fb,
                   std::vector<float> & fd,
                   int64_t M,
                   int64_t N,
                   int64_t K) {
    fa.resize(M * K);
    fb.resize(N * K);
    fd.resize(M * N);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            fa[i * K + k] = static_cast<float>((i * 7 + k * 3) % 5 - 2);
        }
    }
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < K; ++k) {
            fb[j * K + k] = static_cast<float>((j * 5 + k * 11) % 7 - 3);
        }
        for (int64_t i = 0; i < M; ++i) {
            fd[j * M + i] = static_cast<float>((i * 3 + j * 13) % 11 - 5);
        }
    }
}

// Checks C [M, N] against A x B (+ D if @p fd is not empty) bit for bit.
case_result check_result(const std::vector<float> & got,
                         const std::vector<float> & fa,
                         const std::vector<float> & fb,
                         const std::vector<float> & fd,
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
            if (!fd.empty()) {
                want += fd[j * M + i];
            }
            const float g = got[j * M + i];
            if (g != want) {
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

// Writes @p values into @p tensor, converted to its type (exact for the small integers used here).
void set_values(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        return;
    }
    std::vector<std::uint8_t> bytes(ggml_nbytes(tensor));
    ggml_get_type_traits(tensor->type)->from_float_ref(values.data(), bytes.data(),
                                                       static_cast<int64_t>(values.size()));
    ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
}

// Runs C = A x B for A [K, M] and B [K, N] (ggml layout, of type config.type) on a meta device
// over @p devs; for a K split, C = A x B + D for an f32 D [M, N].
case_result
run_case(ggml_backend_dev_t devs[2], split_config config, int64_t M, int64_t N, int64_t K) {
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs, 2, get_split_state, &config);
    if (meta_dev == nullptr) {
        printf("  could not create the meta device\n");
        return case_result::fail;
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{
        ggml_backend_dev_init(meta_dev, nullptr), ggml_backend_free};
    if (backend == nullptr) {
        printf("  could not create the meta backend\n");
        return case_result::fail;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta_dev);

    // weights: placed in a buffer of their own, so the meta device asks get_split_state for them
    const bool residual = config.a_axis == GGML_BACKEND_SPLIT_AXIS_0;
    const std::size_t weights_ctx_size = 3 * ggml_tensor_overhead();
    ggml_init_params weights_params{
        /*.mem_size   =*/weights_ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weights_ctx{ggml_init(weights_params),
                                                                    ggml_free};
    ggml_tensor * a = ggml_new_tensor_2d(weights_ctx.get(), config.type, K, M);
    ggml_set_name(a, "a");
    ggml_tensor * b = ggml_new_tensor_2d(weights_ctx.get(), config.type, K, N);
    ggml_set_name(b, "b");
    ggml_tensor * d = nullptr;
    if (residual) {
        d = ggml_new_tensor_2d(weights_ctx.get(), GGML_TYPE_F32, M, N);
        ggml_set_name(d, "d");
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weights_buf{
        ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(), buft),
        ggml_backend_buffer_free};
    if (weights_buf == nullptr) {
        printf("  weight allocation failed\n");
        return case_result::fail;
    }

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};
    ggml_tensor * ab = ggml_mul_mat(ctx.get(), a, b); // [M, N] f32
    if (!ggml_backend_supports_op(backend.get(), ab)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }
    ggml_tensor * c = residual ? ggml_add(ctx.get(), ab, d) : ab;
    ggml_set_name(c, "c");
    ggml_set_output(c);

    if (!ggml_backend_supports_op(backend.get(), c)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, c);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{ggml_gallocr_new(buft),
                                                                        ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<float> fa;
    std::vector<float> fb;
    std::vector<float> fd;
    fill_operands(fa, fb, fd, M, N, K);
    set_values(a, fa);
    set_values(b, fb);
    if (residual) {
        ggml_backend_tensor_set(d, fd.data(), 0, ggml_nbytes(d));
    } else {
        fd.clear();
    }

    if (ggml_backend_graph_compute(backend.get(), gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<float> got(M * N);
    ggml_backend_tensor_get(c, got.data(), 0, ggml_nbytes(c));

    return check_result(got, fa, fb, fd, M, N, K);
}

// Calls the backend's AllReduce hook on one tensor of @p type per device of @p devs, holding
// different small integers, and checks every tensor then holds their sum. With @p skip_second,
// the second tensor's node is marked as not computed, so the sum is the first tensor alone.
case_result run_allreduce_case(ggml_backend_dev_t devs[2], ggml_type type, bool skip_second) {
    ggml_backend_reg_t reg = ggml_backend_hsa_reg();
    auto comm_init = reinterpret_cast<ggml_backend_comm_init_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init"));
    auto comm_free = reinterpret_cast<ggml_backend_comm_free_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free"));
    auto comm_allreduce = reinterpret_cast<ggml_backend_comm_allreduce_tensor_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor"));
    if (comm_init == nullptr || comm_free == nullptr || comm_allreduce == nullptr) {
        printf("  the backend does not export the AllReduce hook\n");
        return case_result::fail;
    }

    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    constexpr int64_t n = 1000;
    std::vector<backend_ptr> backends;
    std::vector<context_ptr> ctxs;
    std::vector<buffer_ptr> bufs;
    ggml_backend_t raw_backends[2];
    ggml_tensor * tensors[2];
    std::vector<float> values[2];
    for (int d = 0; d < 2; ++d) {
        backends.emplace_back(ggml_backend_dev_init(devs[d], nullptr), ggml_backend_free);
        ggml_init_params params{
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ctxs.emplace_back(ggml_init(params), ggml_free);
        if (backends.back() == nullptr || ctxs.back() == nullptr) {
            printf("  could not create backend %d\n", d);
            return case_result::fail;
        }
        tensors[d] = ggml_new_tensor_1d(ctxs.back().get(), type, n);
        bufs.emplace_back(ggml_backend_alloc_ctx_tensors(ctxs.back().get(), backends.back().get()),
                          ggml_backend_buffer_free);
        if (bufs.back() == nullptr) {
            printf("  could not allocate on backend %d\n", d);
            return case_result::fail;
        }
        if (!(skip_second && d == 1)) {
            tensors[d]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        }
        values[d].resize(n);
        for (int64_t k = 0; k < n; ++k) {
            values[d][k] = static_cast<float>((k * (d + 3) + d) % 9 - 4);
        }
        set_values(tensors[d], values[d]);
        raw_backends[d] = backends.back().get();
    }

    std::unique_ptr<void, decltype(comm_free)> comm_ctx{comm_init(raw_backends, 2), comm_free};
    if (comm_ctx == nullptr) {
        printf("  comm_init refused two HSA backends\n");
        return case_result::fail;
    }
    if (!comm_allreduce(comm_ctx.get(), tensors)) {
        printf("  allreduce declined the tensors\n");
        return case_result::fail;
    }

    int64_t mismatches = 0;
    std::vector<std::uint8_t> bytes(ggml_nbytes(tensors[0]));
    std::vector<float> got(n);
    for (int d = 0; d < 2; ++d) {
        ggml_backend_tensor_get(tensors[d], bytes.data(), 0, bytes.size());
        if (type == GGML_TYPE_F32) {
            std::memcpy(got.data(), bytes.data(), bytes.size());
        } else {
            ggml_get_type_traits(type)->to_float(bytes.data(), got.data(), n);
        }
        for (int64_t k = 0; k < n; ++k) {
            const float want = values[0][k] + (skip_second ? 0.0f : values[1][k]);
            if (got[k] != want) {
                if (mismatches < 8) {
                    printf("  device %d element %lld: got %g want %g\n", d, (long long)k, got[k],
                           want);
                }
                ++mismatches;
            }
        }
    }
    if (mismatches != 0) {
        printf("  %lld / %lld elements mismatched\n", (long long)mismatches, (long long)(2 * n));
        return case_result::fail;
    }
    return case_result::pass;
}

// Returns the first HSA device of type @p type, or @c nullptr.
ggml_backend_dev_t find_device(ggml_backend_reg_t reg, enum ggml_backend_dev_type type) {
    for (std::size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

} // namespace

int main() {
    ggml_backend_reg_t reg = ggml_backend_hsa_reg();
    ggml_backend_dev_t npu = find_device(reg, GGML_BACKEND_DEVICE_TYPE_ACCEL);
    ggml_backend_dev_t gpu = find_device(reg, GGML_BACKEND_DEVICE_TYPE_GPU);
    if (npu == nullptr || gpu == nullptr) {
        printf("test-meta-hsa: needs an HSA NPU and an HSA GPU device (skipped)\n");
        return EXIT_SUCCESS;
    }
    printf("test-meta-hsa: %s + %s\n", ggml_backend_dev_description(npu),
           ggml_backend_dev_description(gpu));

    ggml_backend_dev_t devs[2] = {npu, gpu};

    struct test_case {
        const char * name;
        split_config config;
        int64_t M;
        int64_t N;
        int64_t K;
    };
    const test_case cases[] = {
        {"rows 32/32", {GGML_BACKEND_SPLIT_AXIS_1, 32, GGML_TYPE_F32}, 64, 32, 128},
        {"rows 48/16", {GGML_BACKEND_SPLIT_AXIS_1, 48, GGML_TYPE_F32}, 64, 32, 128},
        {"K 64/64 + residual", {GGML_BACKEND_SPLIT_AXIS_0, 64, GGML_TYPE_F32}, 64, 32, 128},
        {"bf16 K 64/64 + residual", {GGML_BACKEND_SPLIT_AXIS_0, 64, GGML_TYPE_BF16}, 64, 32, 128},
    };

    int failures = 0;
    int skips = 0;
    for (const test_case & tc : cases) {
        printf("case %s: M=%lld N=%lld K=%lld\n", tc.name, (long long)tc.M, (long long)tc.N,
               (long long)tc.K);
        switch (run_case(devs, tc.config, tc.M, tc.N, tc.K)) {
            case case_result::pass:
                printf("  OK\n");
                break;
            case case_result::fail:
                printf("  FAILED\n");
                ++failures;
                break;
            case case_result::skip:
                ++skips;
                break;
        }
    }

    struct allreduce_case {
        const char * name;
        ggml_type type;
        bool skip_second;
    };
    const allreduce_case allreduce_cases[] = {
        {"allreduce f32", GGML_TYPE_F32, false},
        {"allreduce f16", GGML_TYPE_F16, false},
        {"allreduce bf16", GGML_TYPE_BF16, false},
        {"allreduce bf16, second slice empty", GGML_TYPE_BF16, true},
    };
    for (const allreduce_case & tc : allreduce_cases) {
        printf("case %s\n", tc.name);
        if (run_allreduce_case(devs, tc.type, tc.skip_second) == case_result::pass) {
            printf("  OK\n");
        } else {
            printf("  FAILED\n");
            ++failures;
        }
    }

    printf("test-meta-hsa: %d failed, %d skipped, %zu total\n", failures, skips,
           std::size(cases) + std::size(allreduce_cases));
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
