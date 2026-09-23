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

    // Pass 1: max(logits), so pass 2's exp() argument (logits - max) stays <= 0 and can't
    // overflow, no matter how large the logits are.
    auto global_max = std::numeric_limits<float>::lowest();
    CE_ROW_LOOP_HINT
    for (int32_t i = 0; i < Nv; i++) {
        if (logits[i] > global_max) {
            global_max = logits[i];
        }
    }

    // Pass 2: sum(exp(logits - max)), the log-softmax denominator (unnormalized).
    //
    // The exponentials dominate this kernel: stubbing scalar_exp/scalar_log out entirely moved
    // MNIST end-to-end from 48.55 to 32.22 us/image, i.e. 87% of the kernel's measured cost.
    // So evaluate them a vector at a time instead of one at a time.
    //
    // The row (Nv elements) is shorter than the f32 vector, and the ObjectFifo tile is exactly
    // Nv floats -- see the tile_size == row_length check in cross_entropy_loss.py -- so a
    // vector load straight off `logits` would read past the end of the buffer. Stage the row
    // into a padded, vector-aligned scratch buffer instead and load from that. Only the first
    // Nv lanes are summed, so the padding value cannot perturb the result.
    float sum_exp = 0.0f;
#if defined(CROSS_ENTROPY_N) && (CROSS_ENTROPY_N) <= 16
    {
        constexpr int32_t V = 16;
        alignas(64) float xs[V];
        alignas(64) float es[V];

        AIE_LOOP_RANGE(Nv, Nv)
        for (int32_t i = 0; i < Nv; i++) {
            xs[i] = logits[i] - global_max;
        }
        for (int32_t i = Nv; i < V; i++) {
            xs[i] = 0.0f;  // never summed; keeps the lane finite
        }

        auto xv = aie::load_v<V>(xs);
        auto ev = vec_exp<V>(xv);
        aie::store_v(es, ev);

        AIE_LOOP_RANGE(Nv, Nv)
        for (int32_t i = 0; i < Nv; i++) {
            sum_exp += es[i];
        }
    }
#else
    CE_ROW_LOOP_HINT
    for (int32_t i = 0; i < Nv; i++) {
        float x = logits[i] - global_max;
        sum_exp += scalar_exp(x);
    }
#endif

    // log(sum_exp) computed once and reused for every element in pass 3, instead of computing
    // softmax probabilities per element (which would need an extra division and a second log).
    const auto log_sum_exp = scalar_log(sum_exp);

    // Pass 3: log_softmax(x_i) = (x_i - max) - log_sum_exp; loss = -sum(labels * log_softmax).
    float total_loss = 0.0f;
    CE_ROW_LOOP_HINT
    for (int32_t i = 0; i < Nv; i++) {
        float log_softmax = (logits[i] - global_max) - log_sum_exp;
        total_loss += labels[i] * log_softmax;
    }

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;

    event1();
}

} // extern "C"
