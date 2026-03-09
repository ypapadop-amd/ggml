// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file unary_ops.cc
 * @brief Scalar unary operations for AIE kernels.
 *
 * This file implements various element-wise unary operations such as
 * sqr, sqrt, abs, sgn, neg, step, relu, hardsigmoid, hardswish,
 * floor, ceil, round, and trunc.
 */

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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return v * v; });
}

#endif // GGML_OP_SQR

#ifdef GGML_OP_SQRT

/**
 * @brief Computes the square root of each element: out[i] = sqrt(in[i]).
 *
 * @param[in]  in  Input array of N elements.
 * @param[out] out Output array of N elements.
 * @param[in]  N   Number of elements to process.
 */
void ggml_op_sqrt(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return aie::sqrt(v); });
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
    transform_n(in, N, out,
                [](auto v) -> OUTPUT_DTYPE { return v < static_cast<INPUT_DTYPE>(0) ? -v : v; });
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return -v; });
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return v > 0; });
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return std::max<INPUT_DTYPE>(v, 0); });
}

#endif // GGML_UNARY_OP_RELU

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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6));
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return v * std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6));
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<int32>(v);
        }
        return (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v)
                                                  : static_cast<int32>(v) - 1;
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<int32>(v);
        }
        return (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) + 1
                                                  : static_cast<int32>(v);
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return (v >= static_cast<INPUT_DTYPE>(0))
                   ? static_cast<int32>(v + static_cast<INPUT_DTYPE>(.5))
                   : static_cast<int32>(v - static_cast<INPUT_DTYPE>(.5));
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
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return static_cast<int32>(v); });
}

#endif // GGML_UNARY_OP_TRUNC

} // extern "C"
