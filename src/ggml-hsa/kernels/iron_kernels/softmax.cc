// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <stdint.h>

#include <bit>
#include <limits>
#include <type_traits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

/**
 * @brief Computes the ALiBi (Attention with Linear Biases) slope for a head.
 *
 * ALiBi applies position-dependent biases in attention using geometric slopes.
 * The slope for each head is computed based on the head index and total heads.
 *
 * For heads 0 to n_head_log2-1: slope = m0^(head_idx+1)
 * For heads >= n_head_log2:     slope = m1^(2*(head_idx - n_head_log2) + 1)
 *
 * where m0 = 2^(-max_bias/n_head_log2) and m1 = 2^(-max_bias/2/n_head_log2)
 *
 * @param[in] max_bias  Maximum bias value. If <= 0, returns 1.0.
 * @param[in] n_head    Total number of attention heads.
 * @param[in] head_idx  Index of the current head (0-based).
 *
 * @return The computed ALiBi slope for this head.
 */
inline float compute_alibi_slope(float max_bias, int32_t n_head, int32_t head_idx) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    const uint32_t n_head_log2 = 1u << floor_log2(static_cast<uint32_t>(n_head));
    const auto m0 = pow2(-max_bias / n_head_log2);
    const auto m1 = pow2(-(max_bias / 2.0f) / n_head_log2);

    float slope;
    if (static_cast<uint32_t>(head_idx) < n_head_log2) {
        // slope = m0^(head_idx+1) via repeated multiplication
        slope = m0;
        for (uint32_t j = 0; j < static_cast<uint32_t>(head_idx); ++j) {
            slope *= m0;
        }
    } else {
        // slope = m1^(2*(head_idx - n_head_log2) + 1)
        uint32_t exp = 2 * (static_cast<uint32_t>(head_idx) - n_head_log2) + 1;
        slope = m1;
        for (uint32_t j = 1; j < exp; ++j) {
            slope *= m1;
        }
    }

    return slope;
}

#ifdef GGML_OP_SOFT_MAX

namespace {

/// Softmax of one row of @p N elements: see ggml_op_soft_max.
inline void soft_max_row(const float * __restrict input,
                         float * __restrict output,
                         int32_t N,
                         float scale) {
#if __AIEARCH__ == 20 // aie2 only so far; aie2p keeps the scalar passes below, unmeasured there.
    // The scalar passes below call a runtime routine for every float multiply and compare (the
    // scalar unit has neither), so they run in vectors: whole vectors in place, and the last
    // N % V elements through a bounce buffer, so nothing past the row is read or written (the
    // row's neighbour may be the other half of a ping-pong ObjectFifo). Rows need not start
    // 512-bit aligned, so whole vectors use the unaligned accesses.
    constexpr int32_t V = 16;
    using vec = aie::vector<float, V>;

    const int32_t n_whole = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V - 1});
    const int32_t n_tail = N - n_whole;
    const auto in_tail = aie::mask<V>::from_uint32((uint32_t{1} << n_tail) - 1);
    const vec neg_inf = aie::broadcast<float, V>(-std::numeric_limits<float>::infinity());

    // x * 1.0f is x, and the emulated fp32 vector multiply is both slow and not exactly rounded,
    // so a unit scale (plain ggml_soft_max) skips it.
    const bool unit_scale = std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f);
    const auto scaled = [&](vec x) {
        return unit_scale ? x : aie::mul(x, scale).template to_vector<float>();
    };

    // The tail is read once and kept in registers through all three passes.
    const vec x_tail =
        n_tail > 0 ? load_v_partial<V>(input + n_whole, n_tail) : aie::zeros<float, V>();

    // Pass 1: max of the scaled row (see the scalar passes for why it precedes exp()). Rounding
    // scale*x is monotone in x, so max(scale*x) is scale*max(x) for scale >= 0 and scale*min(x)
    // for scale < 0, exactly: the row is reduced unscaled and scaled once.
    vec vmax = neg_inf;
    vec vmin = aie::neg(neg_inf);
    for (int32_t i = 0; i < n_whole; i += V) {
        const vec x = aie::load_unaligned_v<V>(input + i);
        vmax = aie::max(vmax, x);
        vmin = aie::min(vmin, x);
    }
    if (n_tail > 0) {
        vmax = aie::max(vmax, aie::select(neg_inf, x_tail, in_tail));
        vmin = aie::min(vmin, aie::select(aie::neg(neg_inf), x_tail, in_tail));
    }
    const vec vextreme = aie::broadcast<float, V>(
        std::bit_cast<int32_t>(scale) < 0 ? aie::reduce_min(vmin) : aie::reduce_max(vmax));
    const vec vrow_max = scaled(vextreme);

    // Pass 2: exp(scale*x - max) into out, and its sum (tail lanes past the row excluded). Each
    // exp() call takes kExpVectors vectors (see vec_exp_nonpositive); in the last call, lanes past
    // the row are computed and dropped.
    constexpr int32_t kExpVectors = 4;
    vec vsum = aie::zeros<float, V>();
    vec e_tail = aie::zeros<float, V>();
    for (int32_t i = 0; i < N; i += kExpVectors * V) {
        vec x[kExpVectors];
        AIE_LOOP_UNROLL_FULL
        for (int32_t k = 0; k < kExpVectors; ++k) {
            const int32_t j = i + k * V;
            x[k] = aie::sub(scaled(j < n_whole ? aie::load_unaligned_v<V>(input + j) : x_tail),
                            vrow_max);
        }
        const aie::vector<float, kExpVectors * V> e =
            vec_exp_nonpositive<kExpVectors * V>(aie::concat(x[0], x[1], x[2], x[3]));
        AIE_LOOP_UNROLL_FULL
        for (int32_t k = 0; k < kExpVectors; ++k) {
            const int32_t j = i + k * V;
            if (j < n_whole) {
                aie::store_unaligned_v(output + j, e.template extract<V>(k));
                vsum = aie::add(vsum, e.template extract<V>(k));
            } else if (j < N) {
                e_tail = aie::select(aie::zeros<float, V>(), e.template extract<V>(k), in_tail);
                vsum = aie::add(vsum, e_tail);
            }
        }
    }

    // Pass 3: normalize. aie::inv on fp32 ends in a Newton step of its own (see
    // soft_max_lane_rows); it replaces a runtime divide.
    const vec sum_inv = aie::inv(aie::broadcast<float, V>(aie::reduce_add(vsum)));
    for (int32_t i = 0; i < n_whole; i += V) {
        aie::store_unaligned_v(
            output + i,
            aie::mul(aie::load_unaligned_v<V>(output + i), sum_inv).template to_vector<float>());
    }
    if (n_tail > 0) {
        store_v_partial<V>(output + n_whole, aie::mul(e_tail, sum_inv).template to_vector<float>(),
                           n_tail);
    }
#else
    // Pass 1: max of the scaled row. Subtracting this below keeps every scalar_exp() argument
    // <= 0, so exp() can't overflow no matter how large the (scaled) logits are.
    float global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Pass 2: exp(scale*x - max), storing into out so pass 3 only needs to rescale it in place.
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Pass 3: normalize (multiply by the reciprocal instead of dividing per element).
    float sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; ++i) {
        output[i] *= sum_inv;
    }
#endif
}

#if __AIEARCH__ == 20 && defined(GGML_SOFTMAX_ROW_LEN) && GGML_SOFTMAX_ROW_LEN <= 32
#define GGML_SOFTMAX_LANE_ROWS 1

// Rows this short leave most lanes of a row's vectors idle, and a row is one serial chain of
// dependent vector steps (two reductions, exp(), a divide) that nothing else overlaps. So
// kLaneRows rows run together with one row per lane: column c of the group is one vector, each
// pass is lane-wise and needs no reduction, exp() runs on full vectors, and the group shares one
// reciprocal.
constexpr int32_t kLaneRows = 32;

/// Columns per exp() call: two 32-lane columns as one 64-lane vector, whose halves the scheduler
/// interleaves (see vec_exp_nonpositive).
constexpr int32_t kExpColumns = 2;

/// Columns, rounded up to whole exp() calls.
constexpr int32_t kLaneRowsColumns =
    (GGML_SOFTMAX_ROW_LEN + kExpColumns - 1) / kExpColumns * kExpColumns;

/// The group transposed: column c of its rows at [c * kLaneRows, (c + 1) * kLaneRows). A column
/// past the row (odd lengths) is computed and never read.
alignas(aie::vector_decl_align) float lane_rows_columns[kLaneRowsColumns * kLaneRows];

/// Softmax of each of @p n_rows rows of GGML_SOFTMAX_ROW_LEN elements: see ggml_op_soft_max.
inline void soft_max_lane_rows(const float * __restrict input,
                               float * __restrict output,
                               int32_t n_rows,
                               float scale) {
    constexpr int32_t N = GGML_SOFTMAX_ROW_LEN;
    constexpr int32_t L = kLaneRows;
    using vec = aie::vector<float, L>;
    float * __restrict cols = lane_rows_columns;

    // As in soft_max_row: a unit scale skips the emulated fp32 multiply, and the row max is the
    // scaled maximum (or, for a negative scale, minimum) of the unscaled row.
    const bool unit_scale = std::bit_cast<uint32_t>(scale) == std::bit_cast<uint32_t>(1.0f);
    const bool negative_scale = std::bit_cast<int32_t>(scale) < 0;
    const auto scaled = [&](vec x) {
        return unit_scale ? x : aie::mul(x, scale).template to_vector<float>();
    };

    for (int32_t r0 = 0; r0 < n_rows; r0 += L) {
        // In the last group, lanes past its rows hold the previous group's columns (or whatever
        // the buffer held): every pass is lane-wise, so they are computed and never stored.
        const int32_t rows = n_rows - r0 < L ? n_rows - r0 : L;

        // Transpose in, a column at a time, as 32-bit words: no float operation touches the data.
        uint32_t * __restrict cols_bits = reinterpret_cast<uint32_t *>(cols);
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict src = reinterpret_cast<const uint32_t *>(input + r0 * N) + c;
            uint32_t * __restrict col = cols_bits + c * L;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *col++ = *src;
                src += N;
            })
        }

        // Pass 1: the extreme of each row (lane), over its columns.
        vec extreme = aie::load_v<L>(cols);
        if (negative_scale) {
            for (int32_t c = 1; c < N; ++c) {
                extreme = aie::min(extreme, aie::load_v<L>(cols + c * L));
            }
        } else {
            for (int32_t c = 1; c < N; ++c) {
                extreme = aie::max(extreme, aie::load_v<L>(cols + c * L));
            }
        }
        const vec row_max = scaled(extreme);

        // Pass 2: exp(scale*x - max) in place, two columns at a time, and each row's sum.
        vec sum = aie::zeros<float, L>();
        for (int32_t c = 0; c < N; c += kExpColumns) {
            const vec x0 = aie::sub(scaled(aie::load_v<L>(cols + c * L)), row_max);
            const vec x1 = aie::sub(scaled(aie::load_v<L>(cols + (c + 1) * L)), row_max);
            const aie::vector<float, kExpColumns * L> e =
                vec_exp_nonpositive<kExpColumns * L>(aie::concat(x0, x1));
            aie::store_v(cols + c * L, e);
            sum = aie::add(sum, e.template extract<L>(0));
            if (c + 1 < N) {
                sum = aie::add(sum, e.template extract<L>(1));
            }
        }

        // Pass 3: normalize by 1/sum. aie::inv on fp32 ends in a Newton step of its own: a
        // further one moved the outputs' error vs a double reference by at most 2e-7.
        const vec sum_inv = aie::inv(sum);
        for (int32_t c = 0; c < N; ++c) {
            aie::store_v(cols + c * L,
                         aie::mul(aie::load_v<L>(cols + c * L), sum_inv).template to_vector<float>());
        }

        // Transpose out, a column at a time.
        AIE_LOOP_NO_UNROLL
        for (int32_t c = 0; c < N; ++c) {
            const uint32_t * __restrict col = cols_bits + c * L;
            uint32_t * __restrict dst = reinterpret_cast<uint32_t *>(output + r0 * N) + c;
            GGML_AIE_LOOP_MIN_TRIPS(8, r, rows, 1, {
                *dst = *col++;
                dst += N;
            })
        }
    }
}
#endif

} // namespace

#endif // GGML_OP_SOFT_MAX

extern "C" {

#ifdef GGML_OP_SOFT_MAX

/**
 * @brief Computes softmax without mask or sink tensors over a tile of rows.
 *
 * Implements the numerically stable softmax: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 * for each of the tile's @p n_rows rows of @p N elements.
 *
 * Algorithm, per row:
 * 1. Find global maximum of scaled inputs for numerical stability.
 * 2. Compute exp(scale*x - max) for each element and accumulate sum.
 * 3. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in       Input tile of n_rows * N elements.
 * @param[out] out      Output tile of n_rows * N elements.
 * @param[in]  N        Number of elements in a row.
 * @param[in]  n_rows   Number of rows in the tile.
 * @param[in]  scale    Scale factor applied to input before softmax.
 * @param[in]  max_bias Unused in this variant (kept for API consistency).
 */
void ggml_op_soft_max(const INPUT_DTYPE * __restrict in,
                      OUTPUT_DTYPE * __restrict out,
                      int32_t N,
                      int32_t n_rows,
                      float scale,
                      float max_bias) {
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const float * input = reinterpret_cast<const float *>(in);
    float * output = reinterpret_cast<float *>(out);

#ifdef GGML_SOFTMAX_LANE_ROWS
    soft_max_lane_rows(input, output, n_rows, scale);
#else
    for (int32_t r = 0; r < n_rows; ++r) {
        soft_max_row(input + r * N, output + r * N, N, scale);
    }
#endif

    event1();
}

#endif // GGML_OP_SOFT_MAX

#ifdef GGML_OP_SOFT_MAX_WITH_MASK

/**
 * @brief Computes softmax with a mask tensor and ALiBi position biases.
 *
 * Implements: softmax(scale*x + slope*mask) where slope is computed via ALiBi.
 *
 * Algorithm:
 * 1. Compute ALiBi slope for current head based on tile_idx and rows_per_head.
 * 2. Find global maximum of (scale*input + slope*mask) for numerical stability.
 * 3. Compute exp(scale*x + slope*mask - max) and accumulate sum.
 * 4. Normalize by dividing each exp value by the sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements (e.g., causal attention mask).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row.
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Maximum ALiBi bias value.
 * @param[in]  n_head        Total number of attention heads.
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 */
void ggml_op_soft_max_with_mask(const INPUT_DTYPE * __restrict in,
                                const MASK_DTYPE * __restrict mask,
                                OUTPUT_DTYPE * __restrict out,
                                int32_t N,
                                float scale,
                                float max_bias,
                                int32_t n_head,
                                int32_t tile_idx,
                                int32_t rows_per_head) {
    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // tile_idx enumerates rows in head-major order (rows_per_head consecutive rows share a head),
    // so this recovers which head the current row belongs to without a separate index tensor.
    const auto head_idx = tile_idx / rows_per_head;
    const auto slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Pass 1: max of the biased row (see ggml_op_soft_max for why this precedes exp()).
    float global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Pass 2: exp(scale*x + slope*mask - max), stored into out for pass 3 to rescale in place.
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Pass 3: normalize.
    float sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; ++i) {
        output[i] *= sum_inv;
    }

    event1();
}

#endif // GGML_OP_SOFT_MAX_WITH_MASK

#ifdef GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

/**
 * @brief Computes softmax with mask and sink (attention sink) tensors.
 *
 * This variant includes a "sink" value per attention head that participates
 * in the softmax normalization but is not stored in the output. Used for
 * streaming attention where early tokens act as attention sinks.
 *
 * Algorithm:
 * 1. Get sink value and ALiBi slope for current head based on tile_idx.
 * 2. Find global maximum including both (scale*input + slope*mask) and sink.
 * 3. Compute exp(scale*x + slope*mask - max) and accumulate sum.
 * 4. Add exp(sink - max) to sum for proper normalization.
 * 5. Normalize output by dividing each exp value by total sum.
 *
 * @param[in]  in            Input tensor of size N elements.
 * @param[in]  mask          Mask tensor of size N elements.
 * @param[in]  sinks         Per-head sink values array (indexed by head_idx).
 * @param[out] out           Output tensor of size N elements (can alias in).
 * @param[in]  N             Number of elements in the row.
 * @param[in]  tile_idx      Current tile index (used to determine head index).
 * @param[in]  rows_per_head Number of rows per attention head.
 * @param[in]  scale         Scale factor applied to input.
 * @param[in]  max_bias      Maximum ALiBi bias value.
 * @param[in]  n_head        Total number of attention heads.
 */
void ggml_op_soft_max_with_mask_and_sinks(const INPUT_DTYPE * __restrict in,
                                          const MASK_DTYPE * __restrict mask,
                                          const SINK_DTYPE * __restrict sinks,
                                          OUTPUT_DTYPE * __restrict out,
                                          int32_t N,
                                          int32_t tile_idx,
                                          int32_t rows_per_head,
                                          float scale,
                                          float max_bias,
                                          int32_t n_head) {

    static_assert(std::is_same<INPUT_DTYPE, float>::value, "INPUT_DTYPE must be float");
    static_assert(std::is_same<MASK_DTYPE, float>::value, "MASK_DTYPE must be float");
    static_assert(std::is_same<SINK_DTYPE, float>::value, "SINK_DTYPE must be float");
    static_assert(std::is_same<OUTPUT_DTYPE, float>::value, "OUTPUT_DTYPE must be float");

    event0();

    const auto * input = reinterpret_cast<const float *>(in);
    const auto * mask_input = reinterpret_cast<const float *>(mask);
    auto * output = reinterpret_cast<float *>(out);

    // Get sink value for this head
    const auto head_idx = tile_idx / rows_per_head;
    const auto sink_val = sinks[head_idx];

    // Compute ALiBi slope
    const auto slope = compute_alibi_slope(max_bias, n_head, head_idx);

    // Step 1: Find max for numerical stability (including sink)
    float global_max = sink_val;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope;
        if (val > global_max) {
            global_max = val;
        }
    }

    // Step 2: Compute exp(x - max) and sum
    float sum_total = 0.0f;
    for (int32_t i = 0; i < N; ++i) {
        float val = input[i] * scale + mask_input[i] * slope - global_max;
        float exp_val = scalar_exp(val);
        output[i] = exp_val;
        sum_total += exp_val;
    }

    // Add sink contribution to sum
    const auto sink_exp = scalar_exp(sink_val - global_max);
    sum_total += sink_exp;

    // Step 3: Normalize
    const auto sum_inv = 1.0f / sum_total;
    for (int32_t i = 0; i < N; i++) {
        output[i] *= sum_inv;
    }

    event1();
}

#endif // GGML_OP_SOFT_MAX_WITH_MASK_AND_SINKS

} // extern "C"
