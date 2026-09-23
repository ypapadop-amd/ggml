// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

// The three row loops below are all the same fixed trip count. AIE_LOOP_RANGE needs an integral
// constant expression, so the hint can only be attached on the compile-time-N build; the
// runtime-N fallback gets a bare loop.
#ifdef CROSS_ENTROPY_N
#    define CE_ROW_LOOP_HINT AIE_LOOP_RANGE(CROSS_ENTROPY_N, CROSS_ENTROPY_N)
#else
#    define CE_ROW_LOOP_HINT
#endif

extern "C" {

/**
 * @brief Computes cross-entropy loss using numerically stable log-softmax.
 *
 * Computes: loss = -sum(labels * log_softmax(logits))
 * where:   log_softmax(x_i) = (x_i - max) - log(sum(exp(x_j - max)))
 *
 * Three-pass algorithm for numerical stability:
 * 1. Find max(logits) to prevent overflow in exp().
 * 2. Compute sum_exp = sum(exp(logits - max)).
 * 3. Compute loss = -sum(labels * ((logits - max) - log(sum_exp))).
 *
 * @param[in]  logits   Input logits array of N elements (unnormalized scores).
 * @param[in]  labels   Target labels array of N elements (typically one-hot or probabilities).
 * @param[out] loss_out Single-element output array receiving the total loss.
 * @param[in]  N        Number of elements (arbitrary length).
 */
void ggml_op_cross_entropy_loss(const float * __restrict logits,
                                const float * __restrict labels,
                                float * __restrict loss_out,
                                int32_t N) {
    event0();

    // The design streams exactly one row per tile (see cross_entropy_loss.py), so the trip
    // count is the row length and is fixed per kernel instance -- each shape JITs its own .o.
    // Take it as a compile-time constant when the caller supplies one: it lets Peano bind the
    // loop hints below, fold the address arithmetic to immediate offsets, and drop the runtime
    // bound. The runtime-N path is kept for any caller that does not pass -DCROSS_ENTROPY_N.
#ifdef CROSS_ENTROPY_N
    constexpr int32_t Nv = CROSS_ENTROPY_N;
#else
    const int32_t Nv = N;
#endif

    // All three passes run a vector at a time when the row fits one vector.
    //
    // NOOP ablation put this kernel at the top of the MNIST graph, and within it the work is
    // all element-wise: exp was 87% of it before being vectorized, and the remaining scalar
    // max and dot product measured 2.36 us/image of a 34.67 us baseline (6.8%).
    //
    // The row (Nv elements) is shorter than the f32 vector and the ObjectFifo tile is exactly
    // Nv floats -- see the tile_size == row_length check in cross_entropy_loss.py -- so a
    // vector load straight off `logits`/`labels` would read past the end of the buffer. Stage
    // both rows into padded, vector-aligned scratch buffers and work from those.
    //
    // The padding lanes are chosen so that no pass has to mask them out:
    //   logits pad = logits[0] -- a value already in the row, so it cannot change the maximum,
    //                and stays finite through (x - max - log_sum_exp);
    //   labels pad = 0         -- contributes exactly 0*finite == 0 to the pass-3 dot product.
    // Pass 2 is the exception: exp() of a padding lane is a valid non-zero number, so its sum
    // still runs over the first Nv lanes only.
#if defined(CROSS_ENTROPY_N) && (CROSS_ENTROPY_N) <= 16
    constexpr int32_t V = 16;
    alignas(64) float lg[V];
    alignas(64) float lb[V];
    alignas(64) float es[V];

    AIE_LOOP_RANGE(Nv, Nv)
    for (int32_t i = 0; i < Nv; i++) {
        lg[i] = logits[i];
        lb[i] = labels[i];
    }
    for (int32_t i = Nv; i < V; i++) {
        lg[i] = logits[0];
        lb[i] = 0.0f;
    }

    const auto lgv = aie::load_v<V>(lg);
    const auto lbv = aie::load_v<V>(lb);

    // Pass 1: max(logits), so pass 2's exp() argument (logits - max) stays <= 0 and can't
    // overflow, no matter how large the logits are.
    const float global_max = aie::reduce_max(lgv);

    // Pass 2: sum(exp(logits - max)), the log-softmax denominator (unnormalized).
    const auto xv = aie::sub(lgv, aie::broadcast<float, V>(global_max));
    auto xv_exp = xv;
    aie::store_v(es, vec_exp<V>(xv_exp));

    float sum_exp = 0.0f;
    AIE_LOOP_RANGE(Nv, Nv)
    for (int32_t i = 0; i < Nv; i++) {
        sum_exp += es[i];
    }

    // log(sum_exp) computed once and reused for every element in pass 3, instead of computing
    // softmax probabilities per element (which would need an extra division and a second log).
    // Stays scalar: it is one value per row, with nothing to vectorize across.
    const float log_sum_exp = scalar_log(sum_exp);

    // Pass 3: log_softmax(x_i) = (x_i - max) - log_sum_exp; loss = -sum(labels * log_softmax).
    const auto lsv = aie::sub(xv, aie::broadcast<float, V>(log_sum_exp));
    const float total_loss = aie::reduce_add(aie::mul(lbv, lsv).template to_vector<float>());
#else
    // Runtime-N fallback: row length unknown at compile time, so the vector staging above
    // cannot be sized. Scalar, one element at a time.
    auto global_max = std::numeric_limits<float>::lowest();
    for (int32_t i = 0; i < Nv; i++) {
        if (logits[i] > global_max) {
            global_max = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int32_t i = 0; i < Nv; i++) {
        sum_exp += scalar_exp(logits[i] - global_max);
    }

    const float log_sum_exp = scalar_log(sum_exp);

    float total_loss = 0.0f;
    for (int32_t i = 0; i < Nv; i++) {
        total_loss += labels[i] * ((logits[i] - global_max) - log_sum_exp);
    }
#endif

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;

    event1();
}

} // extern "C"
