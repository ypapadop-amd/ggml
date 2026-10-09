// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>
#include <cstdint>
#include <type_traits>

#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

// The row shape and the rows per object are fixed per JIT-compiled instance (convert_pad.py passes
// them): every loop bound below is a compile-time constant.
#if !defined(CONVERT_PAD_D0) || !defined(CONVERT_PAD_D0PAD) || !defined(CONVERT_PAD_ROWS)
#error "convert_pad.cc needs -DCONVERT_PAD_D0, -DCONVERT_PAD_D0PAD and -DCONVERT_PAD_ROWS"
#endif

namespace {

constexpr int32_t d0v = CONVERT_PAD_D0;
constexpr int32_t d0padv = CONVERT_PAD_D0PAD;
constexpr int32_t rows = CONVERT_PAD_ROWS;
static_assert(d0v >= 1 && d0padv >= d0v && rows >= 1,
              "convert_pad needs 1 <= d0 <= d0pad and at least one row");

#ifdef CONVERT_PAD_PAD_ONLY
static_assert(std::is_same_v<INPUT_DTYPE, OUTPUT_DTYPE>,
              "CONVERT_PAD_PAD_ONLY requires matching input and output types");
constexpr int V = 512 / (sizeof(OUTPUT_DTYPE) * 8);

inline aie::vector<OUTPUT_DTYPE, V> convert(const aie::vector<INPUT_DTYPE, V> & v) { return v; }
#else
static_assert(std::is_same_v<INPUT_DTYPE, f32> && std::is_same_v<OUTPUT_DTYPE, bf16>,
              "convert_pad converts f32 -> bf16 only");
constexpr int V = 512 / (sizeof(f32) * 8);

inline aie::vector<bf16, V> convert(const aie::vector<f32, V> & v) {
    return convert_f32_to_bf16_vector_hi16<V>(v);
}
#endif

// Widest aligned store the output rows allow. An object starts vector-aligned; with several rows
// per object, a later row starts at a multiple of the row's byte size, which convert_pad.py only
// allows when that is a multiple of 16.
constexpr int out_row_bytes = d0padv * static_cast<int>(sizeof(OUTPUT_DTYPE));
constexpr int out_row_align = out_row_bytes & -out_row_bytes;
constexpr int store_bytes = rows == 1 || out_row_align >= 64 ? 64 : out_row_align;
static_assert(store_bytes >= 16, "convert_pad output rows must be 16-byte aligned");

/// How a row's partial last vector reads its input.
enum class tail_load {
    from_end, ///< the V elements ending at the row's end, shifted down (they are in the object)
    forward,  ///< the V elements starting at the row, masked (they are in the object)
    bounce,   ///< the row's elements copied into a zeroed buffer (neither is in the object)
};

/**
 * @brief Converts the row @p in of D0 elements into the D0PAD-wide row @p out, zero-filling its
 * tail.
 *
 * Stores are whole vectors or aligned pieces of them (@ref store_v_prefix), since 16-bit unaligned
 * vector stores are broken in this aie_api; loads are unaligned. A partial last vector reads as
 * @p Load says, with its lanes past the row zeroed, so its store also starts the zero padding.
 */
template <int32_t D0, int32_t D0PAD, tail_load Load>
inline void pad_row(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out) {
    constexpr int32_t nblk = D0 / V;
    constexpr int32_t tail = D0 % V;
    constexpr int32_t vend = nblk * V;

    if constexpr (nblk > 0) {
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(nblk, nblk)
        for (int32_t b = 0; b < nblk; ++b) {
            store_v_prefix<V, store_bytes>(out + b * V,
                                           convert(aie::load_unaligned_v<V>(in + b * V)));
        }
    }

    constexpr int32_t tail_lanes = tail == 0 ? 0 : (D0PAD - vend < V ? D0PAD - vend : V);
    if constexpr (tail > 0) {
        const aie::vector<INPUT_DTYPE, V> zeros = aie::zeros<INPUT_DTYPE, V>();
        aie::vector<INPUT_DTYPE, V> v;
        if constexpr (Load == tail_load::from_end) {
            v = aie::shuffle_down_fill(aie::load_unaligned_v<V>(in + D0 - V), zeros, V - tail);
        } else if constexpr (Load == tail_load::forward) {
            static_assert(nblk == 0, "a forward tail load is for rows narrower than a vector");
            v = aie::select(zeros, aie::load_unaligned_v<V>(in),
                            aie::mask<V>::from_uint32((1u << tail) - 1u));
        } else {
            v = load_v_partial<V>(in + vend, tail);
        }
        store_v_prefix<tail_lanes, store_bytes>(out + vend, convert(v));
    }

    constexpr int32_t zstart = vend + tail_lanes;
    constexpr int32_t nzero = (D0PAD - zstart) / V;
    const aie::vector<OUTPUT_DTYPE, V> zeros = aie::zeros<OUTPUT_DTYPE, V>();
    if constexpr (nzero > 0) {
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_RANGE(nzero, nzero)
        for (int32_t b = 0; b < nzero; ++b) {
            store_v_prefix<V, store_bytes>(out + zstart + b * V, zeros);
        }
    }
    store_v_prefix<(D0PAD - zstart) % V, store_bytes>(out + zstart + nzero * V, zeros);
}

/**
 * @brief Converts the R rows of one object.
 *
 * A row at least a vector wide reads its partial last vector back from its own end. A narrower row
 * reads forward into the rows after it while a vector's worth remains in the object, and the rows
 * nearer the end read backward into the rows before them; an object too small for that uses a
 * bounce buffer.
 */
template <int32_t D0, int32_t D0PAD, int32_t R>
inline void pad_rows(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out) {
    // Row r can read forward if (R - r) * D0 >= V, backward if (r + 1) * D0 >= V.
    constexpr int32_t span = (V + D0 - 1) / D0;
    if constexpr (D0 >= V) {
        AIE_LOOP_RANGE(R, R)
        for (int32_t r = 0; r < R; ++r) {
            pad_row<D0, D0PAD, tail_load::from_end>(in + r * D0, out + r * D0PAD);
        }
    } else if constexpr (R >= 2 * span) {
        constexpr int32_t nfwd = R - span + 1;
        AIE_LOOP_RANGE(nfwd, nfwd)
        for (int32_t r = 0; r < nfwd; ++r) {
            pad_row<D0, D0PAD, tail_load::forward>(in + r * D0, out + r * D0PAD);
        }
        AIE_LOOP_RANGE(R - nfwd, R - nfwd)
        for (int32_t r = nfwd; r < R; ++r) {
            pad_row<D0, D0PAD, tail_load::from_end>(in + r * D0, out + r * D0PAD);
        }
    } else {
        AIE_LOOP_RANGE(R, R)
        for (int32_t r = 0; r < R; ++r) {
            pad_row<D0, D0PAD, tail_load::bounce>(in + r * D0, out + r * D0PAD);
        }
    }
}

} // namespace

extern "C" {

/**
 * @brief Widens CONVERT_PAD_ROWS rows from CONVERT_PAD_D0 to CONVERT_PAD_D0PAD elements,
 * zero-filling each row's tail.
 * @param[in]  in     CONVERT_PAD_ROWS input rows of CONVERT_PAD_D0 elements (INPUT_DTYPE).
 * @param[out] out    CONVERT_PAD_ROWS output rows of CONVERT_PAD_D0PAD elements (OUTPUT_DTYPE).
 * @param[in]  d0     Number of valid elements (unused: CONVERT_PAD_D0; kept for the ABI).
 * @param[in]  d0pad  Padded row width (unused: CONVERT_PAD_D0PAD; kept for the ABI).
 */
void ggml_hsa_convert_pad(const INPUT_DTYPE * __restrict in,
                          OUTPUT_DTYPE * __restrict out,
                          int32_t d0,
                          int32_t d0pad) {
    event0();
    (void)d0;
    (void)d0pad;
    pad_rows<d0v, d0padv, rows>(in, out);
    event1();
}

/**
 * @brief Zero-fills CONVERT_PAD_ROWS padding rows of CONVERT_PAD_D0PAD elements (rows past the
 * source's last row).
 * @param[out] out CONVERT_PAD_ROWS output rows of CONVERT_PAD_D0PAD elements (OUTPUT_DTYPE).
 */
void ggml_hsa_convert_pad_zero_row(OUTPUT_DTYPE * __restrict out) {
    event0();

    // The rows are contiguous: zero them as one block from the object's vector-aligned start.
    constexpr int32_t n = rows * d0padv;
    constexpr int32_t nblk = n / V;
    const aie::vector<OUTPUT_DTYPE, V> zeros = aie::zeros<OUTPUT_DTYPE, V>();

    AIE_PREPARE_FOR_PIPELINING
    for (int32_t b = 0; b < nblk; ++b) {
        aie::store_v(out + b * V, zeros);
    }
    store_v_prefix<n % V>(out + nblk * V, zeros);

    event1();
}

} // extern "C"
