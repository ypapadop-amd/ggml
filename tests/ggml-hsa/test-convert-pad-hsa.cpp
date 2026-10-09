// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the HSA-only ggml_hsa_convert_pad op: f32 [d0, d1] -> bf16 [d0pad, d1pad],
// converting the valid sub-block (round-to-nearest-even, matching the host reference), or
// f32 -> f32 / bf16 -> bf16, padding only; either way every padded element reads as zero. Builds a
// real single-node op graph and computes it on the device.
//
// The source holds raw bit patterns, not only tame values: rounding ties, values one below a
// carry, NaNs whose payload sits in the low 16 bits (rounding them as numbers would give an
// infinity), subnormals and random patterns, so every rounding and NaN branch is compared bit for
// bit.
//
// The kernel writes the whole destination, padding included, so the test fills the destination with
// a non-zero sentinel before computing: any padding element the kernel fails to zero shows up as a
// mismatch.

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

// Bit pattern of source element i: a mix of rounding ties, carries, NaNs, subnormals, random bits
// and small exact values.
uint32_t src_bits(int64_t i) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    const uint32_t sign = h & 0x80000000u;
    switch (i % 8) {
        case 0: // anything
            return h;
        case 1: // a tie, either parity
            return sign | 0x3f800000u | ((h & 0x7f) << 16) | 0x8000u;
        case 2: // a NaN with a payload in the low 16 bits only
            return sign | 0x7f800000u | (h & 0xffffu) | 1u;
        case 3: // a subnormal or zero
            return h & 0x807fffffu;
        case 4: // one below a tie, or one below a carry
            return sign | 0x40000000u | ((h & 0x7f) << 16) | ((h & 0x100) ? 0xffffu : 0x7fffu);
        case 5: // the largest binade: may round to infinity
            return sign | 0x7f7f0000u | (h & 0xffffu);
        default:
            break;
    }
    const float v = static_cast<float>(i % 97) * 0.5f - 13.0f;
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(v));
    return u;
}

bool run_case(ggml_backend_t backend,
              ggml_type src_type,
              ggml_type dst_type,
              int64_t d0,
              int64_t d1,
              int64_t d0pad,
              int64_t d1pad) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0, d1);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_convert_pad(ctx.get(), src, dst_type, d0pad, d1pad);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported\n");
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return false;
    }

    // fill the source with bit patterns (see src_bits); fill the (padded) destination with a
    // non-zero sentinel
    const std::size_t src_esize = ggml_type_size(src_type);
    std::vector<uint8_t> src_host(ggml_nbytes(src));
    for (int64_t i = 0; i < d0 * d1; ++i) {
        const uint32_t bits = src_bits(i);
        if (src_esize == 4) {
            std::memcpy(&src_host[i * 4], &bits, 4);
        } else {
            const uint16_t lo = static_cast<uint16_t>(bits);
            std::memcpy(&src_host[i * 2], &lo, 2);
        }
    }
    ggml_backend_tensor_set(src, src_host.data(), 0, ggml_nbytes(src));

    std::vector<uint8_t> dst_sentinel(ggml_nbytes(dst), 0xFF);
    ggml_backend_tensor_set(dst, dst_sentinel.data(), 0, ggml_nbytes(dst));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return false;
    }

    std::vector<uint8_t> dst_host(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_host.data(), 0, ggml_nbytes(dst));

    // compare bit patterns: f32 -> bf16 must round like the host, a same-type pad must copy exactly
    const std::size_t esize = ggml_type_size(dst_type);
    bool ok = true;
    for (int64_t i1 = 0; i1 < d1pad && ok; ++i1) {
        for (int64_t i0 = 0; i0 < d0pad && ok; ++i0) {
            uint32_t got = 0;
            std::memcpy(&got, &dst_host[(i1 * d0pad + i0) * esize], esize);
            uint32_t want = 0;
            if (i0 < d0 && i1 < d1) {
                std::memcpy(&want, &src_host[(i1 * d0 + i0) * src_esize], src_esize);
                if (src_type != dst_type) {
                    float v = 0.0f;
                    std::memcpy(&v, &want, sizeof(v));
                    want = ggml_fp32_to_bf16(v).bits;
                }
            }
            if (got != want) {
                printf("  mismatch at [%lld,%lld]: got 0x%08x want 0x%08x\n", (long long)i0,
                       (long long)i1, got, want);
                ok = false;
            }
        }
    }

    return ok;
}

} // namespace

int main() {
    struct {
        int64_t d0, d1, d0pad, d1pad;
        const char * name;
    } cases[] = {
        {32, 128, 32, 128, "exact (no pad)"},
        {40, 128, 64, 128, "d0 pad"},
        {32, 40, 32, 128, "d1 pad"},
        {40, 40, 64, 128, "d0+d1 pad"},
        // rows narrower than a vector, and partial last vectors whose padding is shorter than a
        // vector, a half vector or not a whole half (the kernel's store and tail paths)
        {2, 6, 8, 8, "2 -> 8"},
        {9, 6, 16, 8, "9 -> 16"},
        {15, 6, 16, 8, "15 -> 16"},
        {17, 6, 32, 8, "17 -> 32"},
        {33, 6, 40, 8, "33 -> 40"},
        {20, 6, 28, 8, "20 -> 28"},
        {26, 6, 28, 8, "26 -> 28"},
        {72, 6, 72, 8, "72 (vector + half)"},
        {100, 6, 104, 8, "100 -> 104"},
        // several rows per object (the design batches short rows): the narrow rows read past their
        // own end, forward or backward within the object
        {2, 48, 8, 48, "2 x 48 rows"},
        {3, 48, 8, 48, "3 x 48 rows"},
        {9, 64, 16, 64, "9 x 64 rows"},
        {72, 32, 72, 48, "72 x 32 rows"},
        // exact MNIST MUL_MAT operand shapes (parent f32 [K,M]/[K,N] -> padded bf16)
        {784, 500, 800, 512, "mnist A/B c1"},
        {500, 10, 512, 128, "mnist A c2"},
        {500, 500, 512, 512, "mnist B c2"},
        {784, 10, 800, 128, "mnist A c3"},
        // the im2col GEMM operands behind MNIST-CNN's two CONV_2D layers: the A operand's
        // d1 is the whole batch*OH*OW extent, orders of magnitude past every case above.
        {9, 392000, 16, 392000, "mnist-cnn conv1 A"},
        {9, 8, 16, 64, "mnist-cnn conv1 B"},
        {72, 98000, 72, 98048, "mnist-cnn conv2 A"},
        {72, 16, 72, 64, "mnist-cnn conv2 B"},
    };

    const struct {
        ggml_type src, dst;
    } types[] = {
        {GGML_TYPE_F32, GGML_TYPE_BF16},
        {GGML_TYPE_F32, GGML_TYPE_F32},
        {GGML_TYPE_BF16, GGML_TYPE_BF16},
    };

    bool all_ok = true;
    for (const auto & t : types) {
        // one backend per type pair: a single device context runs out of resources well before
        // it has loaded every kernel this test builds
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        if (backend == nullptr) {
            printf("HSA backend unavailable; skipping.\n");
            return 0;
        }
        for (const auto & c : cases) {
            // rows are streamed in 32-bit words: an odd number of bf16 source elements is not
            // a streamable row
            if ((c.d0 * ggml_type_size(t.src)) % 4 != 0) {
                continue;
            }
            bool ok = run_case(backend, t.src, t.dst, c.d0, c.d1, c.d0pad, c.d1pad);
            printf("HSA_CONVERT_PAD %4s->%-4s %-18s: %s\n", ggml_type_name(t.src),
                   ggml_type_name(t.dst), c.name, ok ? "PASSED" : "FAILED");
            all_ok = all_ok && ok;
        }
        ggml_backend_free(backend);
    }

    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
