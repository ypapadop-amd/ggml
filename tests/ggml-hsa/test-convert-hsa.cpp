// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for the HSA-only ggml_hsa_convert op: an element-wise dtype cast with no shape
// change (the on-device GGML_OP_CPY cast). Builds a real single-node op graph and computes it on the
// device. Covers f32->bf16 and bf16->f16 (round-to-nearest-even, bit-identical to the host
// reference), bf16->f32 (exact widening), the same-dtype plain copy, and f16->bf16 (exact widening
// then the same RNE). The two 16-bit-to-16-bit narrowing/widening pairs, f16->bf16 and bf16->f16,
// are additionally checked over every one of their 65536 input bit patterns.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"
#include "hsa-test-common.hpp"

namespace {

using hsa_test::cast_val;
using hsa_test::load_val;
using hsa_test::store_val;

enum class case_result { pass, fail, skip };

case_result run_case(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type, int64_t d0,
                     int64_t d1) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0, d1);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_convert(ctx.get(), src, dst_type);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        // Not every shape/dtype pair is streamable: a tensor whose byte size is not a whole
        // number of DMA words (a bf16 tensor with an odd element count) is left to the host copy
        // path rather than over-running the allocation. Treat as a skip, not a failure.
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    // varied, deterministic input pattern
    const int64_t n = d0 * d1;
    std::vector<uint8_t> src_bytes(ggml_nbytes(src));
    for (int64_t i = 0; i < n; ++i) {
        store_val(src_type, src_bytes.data(), i, static_cast<float>(i % 97) * 0.5f - 13.0f);
    }
    ggml_backend_tensor_set(src, src_bytes.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, ggml_nbytes(dst));

    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        // reference: read the source value (as stored in src_type), then cast to dst_type
        const float want = cast_val(dst_type, load_val(src_type, src_bytes.data(), i));
        const float got = load_val(dst_type, dst_bytes.data(), i);
        if (got != want) {
            printf("  mismatch at %lld: got %g want %g\n", (long long)i, got, want);
            ok = false;
        }
    }

    return ok ? case_result::pass : case_result::fail;
}

// Host reference for a 16-bit -> 16-bit conversion, on raw bit patterns.
uint16_t host_convert_bits(ggml_type src_type, ggml_type dst_type, uint16_t bits) {
    const float f = src_type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(bits)
                                              : ggml_bf16_to_fp32(ggml_bf16_t{bits});
    return dst_type == GGML_TYPE_F16 ? ggml_fp32_to_fp16(f) : ggml_fp32_to_bf16(f).bits;
}

// Exhaustive check for a 16-bit -> 16-bit direction: feeds all 65536 source bit patterns through
// the device kernel and compares the raw result bits against the host reference. The generic
// run_case above compares decoded floats, which cannot check NaN or distinguish signed zeros -- and
// both directions are hand-written integer arithmetic on the device (convert_f16_bits_to_f32,
// convert_f32_to_f16_scalar), so those are exactly the patterns worth pinning down. 65536 elements
// is even, so neither side trips the DMA-word check.
//
// The pattern set is repeated @p reps times. With @p in_place the destination aliases the source
// buffer, which is how the backend post-processes an f16 result: the bf16 result is narrowed over
// itself. Many repetitions span many streamed tiles, so a write landing before its read would show.
case_result run_exhaustive(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type,
                           int64_t reps = 1, bool in_place = false) {
    const int64_t n = (int64_t{1} << 16) * reps;

    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, 256, 256 * reps);
    ggml_set_name(src, "src");
    ggml_tensor * dst = ggml_hsa_convert(ctx.get(), src, dst_type);
    ggml_set_name(dst, "dst");

    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};
    if (in_place) {
        if (ggml_nbytes(src) != ggml_nbytes(dst)) {
            printf("  in-place needs equal byte sizes\n");
            return case_result::fail;
        }
        buffer.reset(ggml_backend_alloc_buffer(backend, ggml_nbytes(src)));
        if (buffer == nullptr) {
            printf("  buffer allocation failed\n");
            return case_result::fail;
        }
        void * base = ggml_backend_buffer_get_base(buffer.get());
        if (ggml_backend_tensor_alloc(buffer.get(), src, base) != GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(buffer.get(), dst, base) != GGML_STATUS_SUCCESS) {
            printf("  tensor allocation failed\n");
            return case_result::fail;
        }
    } else if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  graph allocation failed\n");
        return case_result::fail;
    }

    std::vector<uint16_t> src_bits(n);
    for (int64_t i = 0; i < n; ++i) {
        src_bits[i] = static_cast<uint16_t>(i);
    }
    ggml_backend_tensor_set(src, src_bits.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint16_t> dst_bits(n);
    ggml_backend_tensor_get(dst, dst_bits.data(), 0, ggml_nbytes(dst));

    int mismatches = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint16_t want = host_convert_bits(src_type, dst_type, src_bits[i]);
        if (dst_bits[i] != want) {
            if (mismatches < 8) {
                printf("  mismatch: src bits 0x%04x -> got 0x%04x want 0x%04x\n", src_bits[i],
                       dst_bits[i], want);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        printf("  %d / %lld patterns mismatched\n", mismatches, (long long)n);
        return case_result::fail;
    }

    return case_result::pass;
}

// Host reference on raw bit patterns, for any supported pair (16- or 32-bit on either side).
uint32_t host_convert_raw(ggml_type src_type, ggml_type dst_type, uint32_t bits) {
    if (src_type == dst_type) {
        return bits;
    }
    float f = 0.0f;
    if (src_type == GGML_TYPE_F32) {
        std::memcpy(&f, &bits, sizeof(f));
    } else if (src_type == GGML_TYPE_F16) {
        f = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(bits));
    } else {
        f = ggml_bf16_to_fp32(ggml_bf16_t{static_cast<uint16_t>(bits)});
    }
    if (dst_type == GGML_TYPE_F32) {
        uint32_t out = 0;
        std::memcpy(&out, &f, sizeof(out));
        return out;
    }
    return dst_type == GGML_TYPE_F16 ? ggml_fp32_to_fp16(f) : ggml_fp32_to_bf16(f).bits;
}

// Converts a [d0, d1] tensor whose element i holds the bit pattern pattern(i) and compares the raw
// result bits against the host. Unlike run_case this pins down rounding (ties of both parities),
// NaN payloads and signed zeros in every direction, at shapes chosen for their tiling: tiles of a
// half vector, no vector at all, and several workers with uneven shares of the tiles.
template <typename Pattern>
case_result run_bits(ggml_backend_t backend, ggml_type src_type, ggml_type dst_type, int64_t d0,
                     int64_t d1, Pattern pattern) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), src_type, d0, d1);
    ggml_tensor * dst = ggml_hsa_convert(ctx.get(), src, dst_type);
    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  op not supported (skipped)\n");
        return case_result::skip;
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
    const std::size_t src_size = ggml_type_size(src_type);
    const std::size_t dst_size = ggml_type_size(dst_type);
    std::vector<uint8_t> src_bytes(ggml_nbytes(src));
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t bits = pattern(i);
        std::memcpy(&src_bytes[i * src_size], &bits, src_size); // little-endian: the low bytes
    }
    ggml_backend_tensor_set(src, src_bytes.data(), 0, src_bytes.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  graph compute failed\n");
        return case_result::fail;
    }

    std::vector<uint8_t> dst_bytes(ggml_nbytes(dst));
    ggml_backend_tensor_get(dst, dst_bytes.data(), 0, dst_bytes.size());

    int64_t mismatches = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t in = 0;
        uint32_t got = 0;
        std::memcpy(&in, &src_bytes[i * src_size], src_size);
        std::memcpy(&got, &dst_bytes[i * dst_size], dst_size);
        const uint32_t want = host_convert_raw(src_type, dst_type, in);
        if (got != want) {
            if (mismatches < 8) {
                printf("  mismatch at %lld: src bits 0x%08x -> got 0x%08x want 0x%08x\n",
                       (long long)i, in, got, want);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        printf("  %lld / %lld elements mismatched\n", (long long)mismatches, (long long)n);
        return case_result::fail;
    }
    return case_result::pass;
}

} // namespace

int main() {
    {
        // A queue can hold at most 32 kernels; the 33rd dispatch fails with
        // HSA_STATUS_ERROR_OUT_OF_RESOURCES and suspends the queue, failing everything after it.
        // Every (direction, shape) pair below JITs its own kernel and there are more than 32 of
        // them, so each direction runs on a queue of its own (see the loops below) rather than
        // sharing one. This probe just reports an unavailable backend before any of that.
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        if (backend == nullptr) {
            printf("HSA backend unavailable; skipping.\n");
            return 0;
        }
        ggml_backend_free(backend);
    }

    struct {
        int64_t d0, d1;
        const char * name;
    } cases[] = {
        {32, 1, "single row"},
        {64, 8, "wide"},
        {128, 4, "wider"},
        {48, 16, "many rows"},
        {768, 4, "gpt2 n_embd"},
        {500, 500, "square"},
        {1, 64, "single col"},
        // Odd element count: a bf16 tensor is then not a whole number of DMA words, so those
        // variants must be declined rather than rounded up into the next allocation. The f32->f32
        // variant of the same shape still runs, which is what distinguishes "correctly declined"
        // from "the whole shape broke".
        {15, 1, "odd numel"},
        {33, 3, "odd numel rows"},
    };

    struct {
        ggml_type src_type, dst_type;
        const char * label;
    } variants[] = {
        {GGML_TYPE_F32, GGML_TYPE_BF16, "HSA_CONVERT f32->bf16"},
        {GGML_TYPE_BF16, GGML_TYPE_F32, "HSA_CONVERT bf16->f32"},
        {GGML_TYPE_F32, GGML_TYPE_F32, "HSA_CONVERT f32->f32"},
        {GGML_TYPE_F16, GGML_TYPE_BF16, "HSA_CONVERT f16->bf16"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, "HSA_CONVERT bf16->f16"},
    };

    bool any_fail = false;
    int passed = 0;
    int skipped = 0;
    for (const auto & v : variants) {
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        for (const auto & c : cases) {
            const case_result r = run_case(backend, v.src_type, v.dst_type, c.d0, c.d1);
            const char * label = r == case_result::pass   ? "PASSED"
                                 : r == case_result::skip ? "SKIPPED"
                                                          : "FAILED";
            printf("%s %-14s: %s\n", v.label, c.name, label);
            any_fail = any_fail || (r == case_result::fail);
            passed += (r == case_result::pass);
            skipped += (r == case_result::skip);
        }
        ggml_backend_free(backend);
    }

    struct {
        ggml_type src_type, dst_type;
        int64_t reps;
        bool in_place;
        const char * label;
    } exhaustive[] = {
        {GGML_TYPE_F16, GGML_TYPE_BF16, 1, false, "HSA_CONVERT f16->bf16"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 1, false, "HSA_CONVERT bf16->f16"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 1, true, "HSA_CONVERT bf16->f16 in place"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 256, true, "HSA_CONVERT bf16->f16 in place x256"},
    };
    for (const auto & e : exhaustive) {
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        const case_result r = run_exhaustive(backend, e.src_type, e.dst_type, e.reps, e.in_place);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("%s all 65536 bit patterns: %s\n", e.label, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
        ggml_backend_free(backend);
    }

    // Raw-bit cases. Each element's pattern: a 16-bit source cycles through all 65536 patterns; an
    // f32 source pairs every high half with low halves at and around the bf16 rounding tie.
    const auto all16 = [](int64_t i) { return static_cast<uint32_t>(i & 0xffff); };
    const auto f32_ties = [](int64_t i) {
        static const uint32_t lows[] = {0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff, 0x4000, 0xc000};
        return (static_cast<uint32_t>(i & 0xffff) << 16) | lows[(i >> 16) & 7];
    };
    // Every 16-bit pattern alone among normal values (1.0): one per 32 elements, so the vector
    // holding it has no other lane outside the vector path's range. With the patterns in order
    // (all16), a pattern at the edge of that range always shares its vector with one outside it,
    // and the scalar fix-up of that vector would hide a wrong edge.
    const auto isolated = [](ggml_type type) {
        const uint32_t one = type == GGML_TYPE_F16 ? 0x3c00 : 0x3f80;
        return [one](int64_t i) {
            return (i & 31) == 7 ? static_cast<uint32_t>((i >> 5) & 0xffff) : one;
        };
    };
    struct bits_case {
        ggml_type src_type, dst_type;
        int64_t d0, d1;
        bool f32_source_ties;
        const char * label;
    };
    const bits_case bits_cases[] = {
        // several workers, whole-vector tiles
        {GGML_TYPE_F32, GGML_TYPE_BF16, 2048, 256, true, "f32->bf16 ties x8, several workers"},
        // 10 x 500: a multiple of 8 but not 16, so tiles of half vectors
        {GGML_TYPE_F32, GGML_TYPE_BF16, 10, 500, true, "f32->bf16 half-vector tiles"},
        // 14 elements: no vector path
        {GGML_TYPE_F32, GGML_TYPE_BF16, 14, 1, true, "f32->bf16 scalar"},
        {GGML_TYPE_BF16, GGML_TYPE_F32, 256, 256, false, "bf16->f32 all patterns"},
        {GGML_TYPE_BF16, GGML_TYPE_F32, 500, 500, false, "bf16->f32 half-vector tiles"},
        {GGML_TYPE_BF16, GGML_TYPE_F32, 33, 2, false, "bf16->f32 scalar"},
        // 500 x 500 16-bit: a multiple of 16 but not 32
        {GGML_TYPE_F16, GGML_TYPE_BF16, 500, 500, false, "f16->bf16 half-vector tiles"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 500, 500, false, "bf16->f16 half-vector tiles"},
        {GGML_TYPE_F16, GGML_TYPE_BF16, 33, 2, false, "f16->bf16 scalar"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 33, 2, false, "bf16->f16 scalar"},
        // 4000 x 17: 17 tiles of 4000 over 4 workers on aie2 (5, 4, 4, 4), one on aie2p
        {GGML_TYPE_F16, GGML_TYPE_BF16, 4000, 17, false, "f16->bf16 uneven workers"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, 4000, 17, false, "bf16->f16 uneven workers"},
        // 4000 x 85: 85 tiles of 4000 over 2 workers on aie2p (43, 42), 4 on aie2; as f32, 125
        // tiles of 2720 (63, 62 on aie2p)
        {GGML_TYPE_F16, GGML_TYPE_BF16, 4000, 85, false, "f16->bf16 uneven large workers"},
        {GGML_TYPE_F32, GGML_TYPE_BF16, 4000, 85, true, "f32->bf16 uneven large workers"},
        {GGML_TYPE_F32, GGML_TYPE_F32, 2048, 256, true, "f32->f32 several workers"},
        {GGML_TYPE_BF16, GGML_TYPE_BF16, 500, 500, false, "bf16->bf16 half-vector tiles"},
    };
    struct {
        ggml_type src_type, dst_type;
        const char * label;
    } isolated_cases[] = {
        {GGML_TYPE_F16, GGML_TYPE_BF16, "f16->bf16 isolated patterns"},
        {GGML_TYPE_BF16, GGML_TYPE_F16, "bf16->f16 isolated patterns"},
    };
    for (const auto & c : isolated_cases) {
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        const case_result r =
            run_bits(backend, c.src_type, c.dst_type, 2048, 1024, isolated(c.src_type));
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("HSA_CONVERT %s (raw bits): %s\n", c.label, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
        ggml_backend_free(backend);
    }

    for (const auto & c : bits_cases) {
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        const case_result r =
            c.f32_source_ties ? run_bits(backend, c.src_type, c.dst_type, c.d0, c.d1, f32_ties)
                              : run_bits(backend, c.src_type, c.dst_type, c.d0, c.d1, all16);
        const char * label = r == case_result::pass   ? "PASSED"
                             : r == case_result::skip ? "SKIPPED"
                                                      : "FAILED";
        printf("HSA_CONVERT %s (raw bits): %s\n", c.label, label);
        any_fail = any_fail || (r == case_result::fail);
        passed += (r == case_result::pass);
        skipped += (r == case_result::skip);
        ggml_backend_free(backend);
    }

    if (any_fail) {
        printf("FAILURES\n");
        return 1;
    }
    printf("ALL PASSED (%d passed, %d skipped)\n", passed, skipped);
    return 0;
}
