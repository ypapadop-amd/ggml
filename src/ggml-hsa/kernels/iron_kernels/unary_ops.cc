// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file unary_ops.cc
 * @brief Scalar unary operations for AIE kernels.
 */

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"

/**
 * @brief Applies a unary operation to each element of an input array.
 *
 * @tparam T       Element type of the input and output arrays.
 * @tparam Size    Integer type for the count parameter.
 * @tparam UnaryOp Callable type that takes a single element and returns the transformed value.
 *
 * @param[in]  in    Input array of count elements.
 * @param[in]  count Number of elements to process.
 * @param[out] out   Output array of count elements.
 * @param[in]  op    Unary operation to apply to each element.
 */
template <typename T, typename Size, typename UnaryOp>
void transform_n(const T * __restrict in, Size count, T * __restrict out, UnaryOp op) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in[i]);
    }
    event1();
}

/// Whether unary_ops.py selected the large L1-budgeted tile for this op, which it does for
/// exactly the ops with a vectorized body here. Templated so the static_assert below is
/// dependent and therefore checked per instantiation rather than at definition.
template <typename>
inline constexpr bool vectorized_tiling_v =
#ifdef GGML_VECTORIZED_TILING
    true;
#else
    false;
#endif

/**
 * @brief Applies a unary operation to N elements, vectorized when the operand types match.
 *
 * Vector counterpart of @c transform_n for the ops that have a direct aie:: equivalent.
 * The vector path is gated on TIn == TOut because it operates on the loaded vector without
 * a per-element cast; a widening or narrowing op keeps the scalar path unchanged.
 *
 * @tparam Aligned  Whether the tile base is 512-bit aligned, so aligned loads/stores are
 *                  safe. False (the default) is always correct; true is only valid when the
 *                  streamed tile is a whole number of vector registers, which holds for the
 *                  ops unary_ops.py gives the L1-budgeted tile (its tile is a multiple of V,
 *                  and the fallback tile below V leaves vend == 0 so no vector load runs).
 *                  Only RELU has been measured on the aligned path; the others are correct
 *                  either way and stay on the conservative default until measured.
 * @tparam TIn      Input element type.
 * @tparam TOut     Output element type.
 * @tparam VecOp    Callable applied to a vector of TIn.
 * @tparam ScalarOp Callable applied to one element, for the tail and the scalar path.
 *
 * @param[in]  in        Input array of N elements.
 * @param[out] out       Output array of N elements.
 * @param[in]  N         Number of elements to process.
 * @param[in]  vec_op    Vector operation to apply.
 * @param[in]  scalar_op Scalar operation to apply.
 */
template <bool Aligned = false, typename TIn, typename TOut, typename VecOp, typename ScalarOp>
void transform_vector_n(
    const TIn * __restrict in, TOut * __restrict out, int32_t N, VecOp vec_op, ScalarOp scalar_op) {
    static_assert(vectorized_tiling_v<TIn>,
                  "this op has a vectorized body but was compiled without "
                  "GGML_VECTORIZED_TILING, so it streams one vector register per object-fifo "
                  "round trip: add its op name to _VECTORIZED_OPS in unary_ops.py");

    event0();

    int32_t vend = 0;

    if constexpr (std::is_same_v<TIn, TOut>) {
        constexpr int32_t V = 512 / (sizeof(TOut) * 8);
        vend = (N / V) * V;

        // No AIE_LOOP_MIN_ITERATION_COUNT: a tensor narrower than V makes vend 0, and
        // promising >= 1 iteration would make the pipelined prologue run the body on too
        // few elements.
        AIE_PREPARE_FOR_PIPELINING
        for (int32_t i = 0; i < vend; i += V) {
            aie::vector<TIn, V> v;
            if constexpr (Aligned) {
                v = aie::load_v<V>(in + i);
            } else {
                v = aie::load_unaligned_v<V>(in + i);
            }

            const auto r = vec_op(v);

            if constexpr (Aligned) {
                aie::store_v(out + i, r);
            } else {
                aie::store_unaligned_v(out + i, r);
            }
        }
    }

    // Tail of the vector loop, or the whole range when there is no vector path.
    for (int32_t i = vend; i < N; ++i) {
        out[i] = scalar_op(in[i]);
    }

    event1();
}

extern "C" {

#ifdef GGML_OP_SQR

/**
 * @brief Computes the square of each element: out[i] = in[i]^2.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_sqr(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_vector_n(
        in, out, N, [](auto v) { return aie::mul(v, v).template to_vector<OUTPUT_DTYPE>(); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(v * v); });
}

#endif // GGML_OP_SQR

#ifdef GGML_OP_LOG

/**
 * @brief Computes the natural logarithm of each element: out[i] = log(in[i]).
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_log(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    static_assert(std::is_same_v<INPUT_DTYPE, float>, "Input type must be float32");
    static_assert(std::is_same_v<OUTPUT_DTYPE, float>, "Output type must be float32");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return scalar_log(v); });
}

#endif // GGML_OP_LOG

#ifdef GGML_OP_SQRT

/**
 * @brief Computes the square root of each element: out[i] = sqrt(in[i]).
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_sqrt(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out,
                [](auto v) -> OUTPUT_DTYPE { return static_cast<OUTPUT_DTYPE>(aie::sqrt(v)); });
}

#endif // GGML_OP_SQRT

#ifdef GGML_UNARY_OP_ABS

/**
 * @brief Computes the absolute value of each element: out[i] = |in[i]|.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_abs(const INPUT_DTYPE * __restrict in,
                       OUTPUT_DTYPE * __restrict out,
                       int32_t N) {
    transform_vector_n(
        in, out, N, [](auto v) { return vec_abs(v); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(scalar_abs(v)); });
}

#endif // GGML_UNARY_OP_ABS

#ifdef GGML_UNARY_OP_SGN

/**
 * @brief Computes the sign of each element: out[i] = sgn(in[i]).
 *
 * Returns 1 for positive values, -1 for negative values, and 0 for zero.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_sgn(const INPUT_DTYPE * __restrict in,
                       OUTPUT_DTYPE * __restrict out,
                       int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return (v > static_cast<INPUT_DTYPE>(0))
                   ? static_cast<OUTPUT_DTYPE>(1)
                   : ((v < static_cast<INPUT_DTYPE>(0)) ? static_cast<OUTPUT_DTYPE>(-1)
                                                        : static_cast<OUTPUT_DTYPE>(0));
    });
}

#endif // GGML_UNARY_OP_SGN

#ifdef GGML_UNARY_OP_NEG

/**
 * @brief Negates each element: out[i] = -in[i].
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_neg(const INPUT_DTYPE * __restrict in,
                       OUTPUT_DTYPE * __restrict out,
                       int32_t N) {
    transform_vector_n(
        in, out, N, [](auto v) { return vec_neg(v); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(-v); });
}

#endif // GGML_UNARY_OP_NEG

#ifdef GGML_UNARY_OP_STEP

/**
 * @brief Computes the Heaviside step function: out[i] = (in[i] > 0) ? 1 : 0.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_step(const INPUT_DTYPE * __restrict in,
                        OUTPUT_DTYPE * __restrict out,
                        int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(v > static_cast<INPUT_DTYPE>(0));
    });
}

#endif // GGML_UNARY_OP_STEP

#ifdef GGML_UNARY_OP_RELU

/**
 * @brief Applies ReLU activation: out[i] = max(0, in[i]).
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_relu(const INPUT_DTYPE * __restrict in,
                        OUTPUT_DTYPE * __restrict out,
                        int32_t N) {
    static_assert(std::is_same_v<INPUT_DTYPE, OUTPUT_DTYPE>,
                  "ReLU requires matching input and output types");
    transform_vector_n<true>(
        in, out, N, [](auto v) { return aie::max(v, static_cast<INPUT_DTYPE>(0)); },
        [](auto v) { return std::max<OUTPUT_DTYPE>(v, 0); });
}

#endif // GGML_UNARY_OP_RELU

#ifdef GGML_UNARY_OP_GELU

/**
 * @brief Applies the GELU activation (tanh approximation): out[i] = gelu(in[i]).
 *
 * Matches GGML's GGML_UNARY_OP_GELU:
 *   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
 *
 * tanh is evaluated via scalar_exp using the numerically stable identity
 * tanh(y) = sign(y) * (1 - e^-2|y|) / (1 + e^-2|y|), so large-magnitude arguments
 * saturate to +/-1 instead of overflowing.
 *
 * Accepts any floating-point element type: the polynomial and scalar_exp are evaluated in
 * fp32 regardless of the operand type, so a bf16 operand (what the backend substitutes for
 * f16, see @c substitute_fp16_bf16) is promoted on load and rounded once on store.
 *
 * @param[in]  in  Input array of N floating-point elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_gelu(const INPUT_DTYPE * __restrict in,
                        OUTPUT_DTYPE * __restrict out,
                        int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");

    constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kCoefA = 0.044715f;

    // Scalar fallback: the vector path below is f32-only (vec_exp evaluates in f32), so any other
    // element type, and the tail of a partial vector, goes through this.
    auto scalar_gelu = [](auto v) -> OUTPUT_DTYPE {
        const float x = static_cast<float>(v);
        const float y = kSqrt2OverPi * (x + kCoefA * x * x * x);

        const float ay = (y < 0.0f) ? -y : y;
        const float e = scalar_exp(-2.0f * ay);
        const float tanh_abs = (1.0f - e) / (1.0f + e);
        const float tanh_y = (y < 0.0f) ? -tanh_abs : tanh_abs;

        return static_cast<OUTPUT_DTYPE>(0.5f * x * (1.0f + tanh_y));
    };

    if constexpr (std::is_same_v<INPUT_DTYPE, f32> && std::is_same_v<OUTPUT_DTYPE, f32>) {
        transform_vector_n(in, out, N, [](auto v) { return vec_gelu(v); }, scalar_gelu);
    } else {
        transform_n(in, N, out, scalar_gelu);
    }
}

#endif // GGML_UNARY_OP_GELU

#ifdef GGML_UNARY_OP_HARDSIGMOID

/**
 * @brief Applies hard sigmoid activation: out[i] = clamp((in[i] + 3) / 6, 0, 1).
 *
 * A piecewise linear approximation of the sigmoid function.
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_hardsigmoid(const INPUT_DTYPE * __restrict in,
                               OUTPUT_DTYPE * __restrict out,
                               int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6)));
    });
}

#endif // GGML_UNARY_OP_HARDSIGMOID

#ifdef GGML_UNARY_OP_HARDSWISH

/**
 * @brief Applies hard swish activation: out[i] = in[i] * hardsigmoid(in[i]).
 *
 * Computes: x * clamp((x + 3) / 6, 0, 1)
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_hardswish(const INPUT_DTYPE * __restrict in,
                             OUTPUT_DTYPE * __restrict out,
                             int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            v * std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6)));
    });
}

#endif // GGML_UNARY_OP_HARDSWISH

#ifdef GGML_UNARY_OP_FLOOR

/**
 * @brief Computes the floor of each element: out[i] = floor(in[i]).
 *
 * Returns the largest integer less than or equal to the input.
 * Input type must be a floating-point type.
 *
 * @param[in]  in  Input array of N floating-point elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_floor(const INPUT_DTYPE * __restrict in,
                         OUTPUT_DTYPE * __restrict out,
                         int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
        }
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) : static_cast<int32>(v) - 1);
    });
}

#endif // GGML_UNARY_OP_FLOOR

#ifdef GGML_UNARY_OP_CEIL

/**
 * @brief Computes the ceiling of each element: out[i] = ceil(in[i]).
 *
 * Returns the smallest integer greater than or equal to the input.
 * Input type must be a floating-point type.
 *
 * @param[in]  in  Input array of N floating-point elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_ceil(const INPUT_DTYPE * __restrict in,
                        OUTPUT_DTYPE * __restrict out,
                        int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
        }
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) + 1 : static_cast<int32>(v));
    });
}

#endif // GGML_UNARY_OP_CEIL

#ifdef GGML_UNARY_OP_ROUND

/**
 * @brief Rounds each element to the nearest integer: out[i] = round(in[i]).
 *
 * Uses round-half-away-from-zero: 0.5 rounds to 1, -0.5 rounds to -1.
 * Input type must be a floating-point type.
 *
 * @param[in]  in  Input array of N floating-point elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_round(const INPUT_DTYPE * __restrict in,
                         OUTPUT_DTYPE * __restrict out,
                         int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0))
                ? static_cast<int32>(v + static_cast<INPUT_DTYPE>(.5))
                : static_cast<int32>(v - static_cast<INPUT_DTYPE>(.5)));
    });
}

#endif // GGML_UNARY_OP_ROUND

#ifdef GGML_UNARY_OP_TRUNC

/**
 * @brief Truncates each element toward zero: out[i] = trunc(in[i]).
 *
 * Returns the integer part by removing the fractional digits.
 * Input type must be a floating-point type.
 *
 * @param[in]  in  Input array of N floating-point elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_unary_op_trunc(const INPUT_DTYPE * __restrict in,
                         OUTPUT_DTYPE * __restrict out,
                         int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
    });
}

#endif // GGML_UNARY_OP_TRUNC

} // extern "C"
