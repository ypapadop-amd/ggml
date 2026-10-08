// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <cstdint>
#include <limits>

#include <aie_api/aie.hpp>

#include "aie_kernel_math.h"
#include "ggml-aie.hpp"

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

#if __AIEARCH__ == 20 // aie2 only so far; aie2p keeps the passes below, unmeasured there.
    // As below, but 16 lanes, and the last N % 16 elements in vectors too, through a bounce
    // buffer (nothing past the row is read), instead of scalar loops that call a runtime routine
    // for every float multiply and compare and for scalar_exp's arithmetic. One scalar_log per
    // row remains. Lanes past the row are excluded from the max and the exp sum, and contribute
    // label 0 * finite to the loss.
    {
        constexpr int32_t V16 = 16;
        using vec = aie::vector<float, V16>;
        const int32_t n_whole = static_cast<int32_t>(static_cast<uint32_t>(N) & ~uint32_t{V16 - 1});
        const int32_t n_tail = N - n_whole;
        const auto in_tail = aie::mask<V16>::from_uint32((uint32_t{1} << n_tail) - 1);
        const vec neg_inf = aie::broadcast<float, V16>(-std::numeric_limits<float>::infinity());
        const auto load_tail = [&](const float * p) {
            alignas(aie::vector_decl_align) float buf[V16] = {};
            for (int32_t j = 0; j < n_tail; ++j) {
                buf[j] = p[j];
            }
            return aie::load_v<V16>(buf);
        };
        const vec x_tail = n_tail > 0 ? load_tail(logits + n_whole) : aie::zeros<float, V16>();
        const vec l_tail = n_tail > 0 ? load_tail(labels + n_whole) : aie::zeros<float, V16>();

        // Pass 1: max(logits).
        vec vmax = neg_inf;
        for (int32_t i = 0; i < n_whole; i += V16) {
            vmax = aie::max(vmax, aie::load_unaligned_v<V16>(logits + i));
        }
        vmax = aie::max(vmax, aie::select(neg_inf, x_tail, in_tail));
        const vec vrow_max = aie::broadcast<float, V16>(aie::reduce_max(vmax));

        // Pass 2: sum(exp(logits - max)).
        vec vsum = aie::zeros<float, V16>();
        for (int32_t i = 0; i < n_whole; i += V16) {
            vec x = aie::sub(aie::load_unaligned_v<V16>(logits + i), vrow_max);
            vsum = aie::add(vsum, vec_exp<V16>(x));
        }
        if (n_tail > 0) {
            vec x = aie::sub(x_tail, vrow_max);
            vsum = aie::add(vsum, aie::select(aie::zeros<float, V16>(), vec_exp<V16>(x), in_tail));
        }
        const vec vlse = aie::broadcast<float, V16>(scalar_log(aie::reduce_add(vsum)));

        // Pass 3: loss = -sum(labels * ((logits - max) - log_sum_exp)).
        const auto row_term = [&](vec x, vec l) {
            return aie::mul(l, aie::sub(aie::sub(x, vrow_max), vlse)).template to_vector<float>();
        };
        vec vloss = aie::zeros<float, V16>();
        for (int32_t i = 0; i < n_whole; i += V16) {
            vloss = aie::add(vloss, row_term(aie::load_unaligned_v<V16>(logits + i),
                                             aie::load_unaligned_v<V16>(labels + i)));
        }
        if (n_tail > 0) {
            vloss = aie::add(vloss, row_term(x_tail, l_tail));
        }
        loss_out[0] = -aie::reduce_add(vloss);
    }
#else
    // The three passes below are vectorized over KERN_VEC lanes with a scalar tail. The win is
    // pass 2: an ablation that removed the transcendentals took this kernel from 4084 us to 594
    // us, so ~85% of its cost was the per-element scalar_exp. A row is only ~10 elements here
    // (one per class), which is under the 16-lane f32 vector, so the tail still runs a couple of
    // scalar iterations -- the vector body is what removes the bulk of the exp calls.
    constexpr int32_t V = KERN_VEC_SIZE;
    const int32_t vend = (N / V) * V;

    // Pass 1: max(logits), so pass 2's exp() argument (logits - max) stays <= 0 and can't
    // overflow, no matter how large the logits are.
    auto global_max = std::numeric_limits<float>::lowest();
    if (vend > 0) {
        aie::vector<float, V> vmax = aie::broadcast<float, V>(global_max);
        for (int32_t i = 0; i < vend; i += V) {
            vmax = aie::max(vmax, aie::load_unaligned_v<V>(logits + i));
        }
        global_max = aie::reduce_max(vmax);
    }
    for (int32_t i = vend; i < N; i++) {
        if (logits[i] > global_max) {
            global_max = logits[i];
        }
    }

    // Pass 2: sum(exp(logits - max)), the log-softmax denominator (unnormalized).
    float sum_exp = 0.0f;
    const aie::vector<float, V> vmax_b = aie::broadcast<float, V>(global_max);
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<float, V> x = aie::sub(aie::load_unaligned_v<V>(logits + i), vmax_b);
        sum_exp += aie::reduce_add(vec_exp<V>(x));
    }
    for (int32_t i = vend; i < N; i++) {
        sum_exp += scalar_exp(logits[i] - global_max);
    }

    // log(sum_exp) computed once and reused for every element in pass 3, instead of computing
    // softmax probabilities per element (which would need an extra division and a second log).
    const auto log_sum_exp = scalar_log(sum_exp);

    // Pass 3: log_softmax(x_i) = (x_i - max) - log_sum_exp; loss = -sum(labels * log_softmax).
    float total_loss = 0.0f;
    const aie::vector<float, V> vlse_b = aie::broadcast<float, V>(log_sum_exp);
    for (int32_t i = 0; i < vend; i += V) {
        aie::vector<float, V> ls =
            aie::sub(aie::sub(aie::load_unaligned_v<V>(logits + i), vmax_b), vlse_b);
        aie::vector<float, V> p =
            aie::mul(aie::load_unaligned_v<V>(labels + i), ls).template to_vector<float>();
        total_loss += aie::reduce_add(p);
    }
    for (int32_t i = vend; i < N; i++) {
        float log_softmax = (logits[i] - global_max) - log_sum_exp;
        total_loss += labels[i] * log_softmax;
    }

    // Store negated loss (cross entropy is -sum)
    loss_out[0] = -total_loss;
#endif

    event1();
}

} // extern "C"
