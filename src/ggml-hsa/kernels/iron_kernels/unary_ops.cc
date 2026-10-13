// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

/**
 * @file unary_ops.cc
 * @brief Scalar unary operations for AIE kernels.
 */

#include "aie_kernel_math.h"
#include "aie_kernel_utils.h"
#include "ggml-aie.hpp"
#include "transform.h"

#if __AIE_ARCH__ != 20 // aie2 runs these ops scalar, on one-vector tiles.

/// Which integer @ref round_integral_n rounds to.
enum class round_integral { floor, ceil, round, trunc };

/**
 * @brief floorf, ceilf, roundf (half away from zero) or truncf of every lane, bit for bit, on the
 * bits of f32 values.
 *
 * On the magnitude a = |x|, from float adds and integer compares only (the f32 vector float to
 * int conversion rounds by the core's mode, and f32 compares are emulated): a + 2^23 - 2^23 is a
 * rounded to an integer whichever way the add rounds, and one less where that is above a, which
 * gives floor(a). The integer above it is taken where the mode and sign ask for it (round: where
 * a - floor(a), exact as both share an exponent, is at least 1/2), and x's sign put back, so
 * ceilf(-0.5) is -0 as on the CPU. |x| >= 2^23, infinities and NaN are already integral and kept.
 *
 * Takes and returns the bits as int32: LLVM turns the sign-bit mask of a float's bits into a
 * vector fabs, which aie2p runs one lane at a time.
 *
 * @tparam M Rounding.
 * @tparam V Vector width.
 * @param[in] bits The f32 inputs' bits.
 * @return The bits of every lane rounded to an integer.
 */
template <round_integral M, unsigned V>
aie::vector<int32_t, V> vec_round_integral(const aie::vector<int32_t, V> & bits) {
    using ivec = aie::vector<int32_t, V>;
    using fvec = aie::vector<float, V>;
    constexpr float kTwo23 = 8388608.0f;
    const ivec abs_bits = aie::bit_and(int32_t{0x7FFFFFFF}, bits);
    const fvec a = aie::vector_cast<float>(abs_bits);
    // Non-negative floats order as their bits do.
    fvec f = aie::sub(aie::add(a, kTwo23), kTwo23);
    f = aie::select(f, aie::sub(f, 1.0f), aie::gt(aie::vector_cast<int32_t>(f), abs_bits));
    if constexpr (M == round_integral::round) {
        const ivec frac = aie::vector_cast<int32_t>(aie::sub(a, f));
        f = aie::select(
            f, aie::add(f, 1.0f),
            aie::ge(frac, aie::broadcast<int32_t, V>(__builtin_bit_cast(int32_t, 0.5f))));
    } else if constexpr (M != round_integral::trunc) {
        const auto fractional = aie::neq(aie::vector_cast<int32_t>(f), abs_bits);
        const auto away = M == round_integral::floor ? aie::lt(bits, aie::zeros<int32_t, V>())
                                                     : aie::ge(bits, aie::zeros<int32_t, V>());
        f = aie::select(f, aie::add(f, 1.0f), fractional & away);
    }
    // f's bits are non-negative, so adding x's sign bit sets it.
    const ivec r = aie::add(aie::vector_cast<int32_t>(f), aie::bit_and(INT32_MIN, bits));
    return aie::select(
        r, bits,
        aie::ge(abs_bits, aie::broadcast<int32_t, V>(__builtin_bit_cast(int32_t, kTwo23))));
}

/// Scalar counterpart of @ref vec_round_integral, the same arithmetic on one element's bits.
template <round_integral M>
int32_t scalar_round_integral(int32_t bits) {
    constexpr float kTwo23 = 8388608.0f;
    const int32_t abs_bits = bits & 0x7FFFFFFF;
    if (abs_bits >= __builtin_bit_cast(int32_t, kTwo23)) {
        return bits;
    }
    const float a = __builtin_bit_cast(float, abs_bits);
    float f = (a + kTwo23) - kTwo23;
    if (__builtin_bit_cast(int32_t, f) > abs_bits) {
        f -= 1.0f;
    }
    if constexpr (M == round_integral::round) {
        if (__builtin_bit_cast(int32_t, a - f) >= __builtin_bit_cast(int32_t, 0.5f)) {
            f += 1.0f;
        }
    } else if constexpr (M != round_integral::trunc) {
        if (__builtin_bit_cast(int32_t, f) != abs_bits &&
            (M == round_integral::floor ? bits < 0 : bits >= 0)) {
            f += 1.0f;
        }
    }
    return __builtin_bit_cast(int32_t, f) + (bits & INT32_MIN);
}

/**
 * @brief out[i] = in[i] rounded to an integer by @p M: f32 walked as its bits in vectors of
 * @ref vec_round_integral, other types by @p scalar_op.
 */
template <round_integral M, typename TOut, typename TIn, typename ScalarOp>
void round_integral_n(TOut * __restrict out,
                      int32_t N,
                      ScalarOp scalar_op,
                      const TIn * __restrict in) {
    if constexpr (std::is_same_v<TIn, float> && std::is_same_v<TOut, float>) {
        transform_vector_walk_n<tile_vector_aligned_v<float>>(
            reinterpret_cast<int32_t *>(out), N, [](auto v) { return vec_round_integral<M>(v); },
            [](int32_t v) { return scalar_round_integral<M>(v); },
            reinterpret_cast<const int32_t *>(in));
    } else {
        transform_n(out, N, scalar_op, in);
    }
}

/**
 * @brief clamp((x + 3) / 6, 0, 1) of every lane, as ggml's CPU hardsigmoid (which gives 0 for
 * NaN).
 *
 * (x + 3) / 6 is a multiply by the f32 nearest 1/6, within one ulp of the division. The clamp
 * compares the result's bits as integers, since f32 compares are emulated: non-negative floats
 * order as their bits, NaN's above +inf's.
 *
 * @tparam V Vector width.
 * @param[in] x The input vector.
 * @return The element-wise hard sigmoid.
 */
template <unsigned V>
aie::vector<float, V> vec_hardsigmoid(const aie::vector<float, V> & x) {
    using ivec = aie::vector<int32_t, V>;
    constexpr int32_t kOneBits = __builtin_bit_cast(int32_t, 1.0f);
    constexpr int32_t kInfBits = 0x7F800000;
    const ivec t = aie::vector_cast<int32_t>(
        aie::mul(aie::add(x, 3.0f), 1.0f / 6.0f).template to_vector<float>());
    const ivec x_bits = aie::vector_cast<int32_t>(x);
    const auto pos_inf = aie::eq(x_bits, aie::broadcast<int32_t, V>(kInfBits));
    // Above 1, and +inf (whose emulated product is NaN), give 1; below 0 and NaN give 0.
    const ivec r = aie::select(t, aie::broadcast<int32_t, V>(kOneBits),
                               aie::gt(t, aie::broadcast<int32_t, V>(kOneBits)) | pos_inf);
    return aie::vector_cast<float>(
        aie::select(r, aie::zeros<int32_t, V>(),
                    aie::lt(t, aie::zeros<int32_t, V>()) |
                        (aie::gt(t, aie::broadcast<int32_t, V>(kInfBits)) & ~pos_inf)));
}

/**
 * @brief logf of every lane: @ref vec_log, with logf's special values and subnormals.
 *
 * A subnormal m 2^-149 is taken as m (the bits of 2^23 + m, less 2^23, both exact) and 149 ln 2
 * taken off its log. ln(+-0) is -inf, ln of a negative NaN, ln(+inf) +inf and ln(NaN) NaN.
 *
 * @tparam V Vector width.
 * @param[in] x The input vector.
 * @return The element-wise natural logarithm.
 */
template <unsigned V>
aie::vector<float, V> vec_logf(const aie::vector<float, V> & x) {
    using ivec = aie::vector<int32_t, V>;
    constexpr int32_t kMinNormalBits = 0x00800000;
    constexpr int32_t kInfBits = 0x7F800000;
    constexpr float kTwo23 = 8388608.0f;
    const ivec bits = aie::vector_cast<int32_t>(x);
    const auto subnormal = aie::lt(bits, aie::broadcast<int32_t, V>(kMinNormalBits));
    const aie::vector<float, V> m =
        aie::sub(aie::vector_cast<float>(aie::add(
                     bits, aie::broadcast<int32_t, V>(__builtin_bit_cast(int32_t, kTwo23)))),
                 kTwo23);
    aie::vector<float, V> y = vec_log<V>(aie::select(x, m, subnormal));
    y = aie::select(y, aie::sub(y, 103.27892990343185f), subnormal); // 149 ln 2
    ivec r = aie::select(aie::vector_cast<int32_t>(y), bits,
                         aie::ge(bits, aie::broadcast<int32_t, V>(kInfBits)));
    // A float compare: on the bits, LLVM folds "0 or INT32_MIN" into a vector fabs and then a
    // fneg, which aie2p cannot legalize.
    const auto zero = aie::eq(x, aie::zeros<float, V>());
    r = aie::select(r, aie::broadcast<int32_t, V>(static_cast<int32_t>(0xFF800000u)), zero);
    r = aie::select(r, aie::broadcast<int32_t, V>(0x7FC00000),
                    aie::lt(bits, aie::zeros<int32_t, V>()) & ~zero);
    return aie::vector_cast<float>(r);
}

/**
 * @brief out[i] = |in[i]|: floating point walked as its unsigned bits, the sign bit cleared;
 * other types as @ref vec_abs and @ref scalar_abs.
 *
 * On the bits because LLVM folds the mask of a float vector's bits into a vector fabs, which
 * aie2p runs one lane at a time.
 */
template <typename TOut, typename TIn>
void abs_n(TOut * __restrict out, int32_t N, const TIn * __restrict in) {
    if constexpr (is_floating_point_v<TOut> && std::is_same_v<TIn, TOut>) {
        using U = same_width_uint_t<TOut>;
        constexpr U kMask = static_cast<U>(~(U{1} << (sizeof(U) * 8 - 1)));
        transform_vector_walk_n<tile_vector_aligned_v<TOut>>(
            reinterpret_cast<U *>(out), N, [](auto v) { return aie::bit_and(kMask, v); },
            [](U v) { return static_cast<U>(v & kMask); }, reinterpret_cast<const U *>(in));
    } else {
        transform_vector_walk_n<tile_vector_aligned_v<TOut>>(
            out, N, [](auto v) { return vec_abs(v); },
            [](auto v) { return static_cast<TOut>(scalar_abs(v)); }, in);
    }
}

#endif // __AIE_ARCH__ != 20

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
    // aie2p: f32 32 lanes a step, as the emulated f32 multiply costs the same at 16.
    constexpr unsigned kVectorBits =
        __AIE_ARCH__ != 20 && std::is_same_v<OUTPUT_DTYPE, f32> ? 1024 : 512;
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, true, kVectorBits>(
        out, N, [](auto v) { return aie::mul(v, v).template to_vector<OUTPUT_DTYPE>(); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(v * v); }, in);
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE { return scalar_log(v); };
#if __AIE_ARCH__ != 20
    // 32 lanes a step: vec_log's emulated f32 multiplies cost the same at 16.
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, true, 1024>(
        out, N, [](auto v) { return vec_logf(v); }, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "Output type must be a floating point type");

    auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(scalar_sqrt(static_cast<float>(v)));
    };

    // vec_sqrt is f32-only; other element types are promoted per element by the scalar path.
    if constexpr (std::is_same_v<INPUT_DTYPE, f32> && std::is_same_v<OUTPUT_DTYPE, f32>) {
#if __AIE_ARCH__ != 20
        // 32 lanes a step, as vec_sqrt's emulated multiplies cost the same at 16, and walked as
        // int32: LLVM folds vec_sqrt's sign-bit mask of a float vector's bits into a vector fabs,
        // which aie2p runs one lane at a time.
        transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, true, 1024>(
            reinterpret_cast<int32_t *>(out), N,
            [](auto v) {
                // Inlined here: Peano otherwise calls it once per vector.
                [[clang::always_inline]] return aie::vector_cast<int32_t>(
                    vec_sqrt(aie::vector_cast<float>(v)));
            },
            [](int32_t v) {
                return __builtin_bit_cast(int32_t, scalar_sqrt(__builtin_bit_cast(float, v)));
            },
            reinterpret_cast<const int32_t *>(in));
#else
        transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
            out, N, [](auto v) { return vec_sqrt(v); }, scalar_op, in);
#endif
    } else {
        transform_n(out, N, scalar_op, in);
    }
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
#if __AIE_ARCH__ != 20
    abs_n(out, N, in);
#else
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
        out, N, [](auto v) { return vec_abs(v); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(scalar_abs(v)); }, in);
#endif
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
    // Two selects rather than a branch: aie::select(a, b, m) yields b where m holds. The
    // comparisons and the constants are exact, so the vector body matches the scalar tail bit
    // for bit.
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
        out, N,
        [](auto v) {
            using V = std::decay_t<decltype(v)>;
            const V zero = aie::zeros<OUTPUT_DTYPE, V::size()>();
            const V pos = aie::broadcast<OUTPUT_DTYPE, V::size()>(static_cast<OUTPUT_DTYPE>(1));
            const V neg = aie::broadcast<OUTPUT_DTYPE, V::size()>(static_cast<OUTPUT_DTYPE>(-1));
            return aie::select(aie::select(zero, neg, aie::lt(v, zero)), pos, aie::gt(v, zero));
        },
        [](auto v) -> OUTPUT_DTYPE {
            return (v > static_cast<INPUT_DTYPE>(0))
                       ? static_cast<OUTPUT_DTYPE>(1)
                       : ((v < static_cast<INPUT_DTYPE>(0)) ? static_cast<OUTPUT_DTYPE>(-1)
                                                            : static_cast<OUTPUT_DTYPE>(0));
        },
        in);
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
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
        out, N, [](auto v) { return vec_neg(v); },
        [](auto v) { return static_cast<OUTPUT_DTYPE>(-v); }, in);
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
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
        out, N,
        [](auto v) {
            using V = std::decay_t<decltype(v)>;
            const V zero = aie::zeros<OUTPUT_DTYPE, V::size()>();
            return aie::select(
                zero, aie::broadcast<OUTPUT_DTYPE, V::size()>(static_cast<OUTPUT_DTYPE>(1)),
                aie::gt(v, zero));
        },
        [](auto v) -> OUTPUT_DTYPE {
            return static_cast<OUTPUT_DTYPE>(v > static_cast<INPUT_DTYPE>(0));
        },
        in);
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
    // Aligned accesses only when the builder guaranteed the tile is a whole number of vector
    // registers. It usually is, but tiled_tile_size falls back below one vector for a count no
    // multiple of V divides (19 f32 elements gives tile 1), and hardcoding true there asks for
    // a guarantee that does not hold -- which transform_vector_n's static_assert rejects, so
    // the kernel fails to compile rather than running unaligned. Deriving it keeps both shapes
    // working and still leaves the assert to catch a hardcoded true elsewhere.
    //
    // ggml's (x > 0) ? x : 0, decided on the bits of x read as a signed integer: x > 0 exactly when
    // that integer is positive and not above +inf's (above it are the positive NaNs). Everything
    // else -- negatives, both zeros, -inf, every NaN -- gives +0.0, as on the CPU. One integer max
    // plus a compare and select per vector, where aie::max on f32 is emulated on aie2 (a subtract,
    // a bf16 compare and a select, ~3x the loop's II), and the scalar tail needs no float compare.
    static_assert(is_floating_point_v<OUTPUT_DTYPE>, "ReLU requires a floating point type");
    using I = std::make_signed_t<same_width_uint_t<OUTPUT_DTYPE>>;
    constexpr I kInfBits = static_cast<I>(I{0x7F80} << (8 * sizeof(I) - 16));
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>>(
        out, N,
        [](auto v) {
            using IV = aie::vector<I, decltype(v)::size()>;
            const IV bits = aie::vector_cast<I>(v);
            const IV kept = aie::max(bits, I{0});
            return aie::vector_cast<OUTPUT_DTYPE>(
                aie::select(kept, aie::zeros<I, IV::size()>(), aie::gt(bits, kInfBits)));
        },
        [](auto v) {
            // Bit casts, not memcpy: Peano emits a call to memcpy for that here.
            const I bits = __builtin_bit_cast(I, v);
            return __builtin_bit_cast(OUTPUT_DTYPE, (bits > 0 && bits <= kInfBits) ? bits : I{0});
        },
        in);
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
        // aie2p: 32 lanes a step, as vec_gelu's emulated multiplies cost the same at 16.
        constexpr unsigned kVectorBits = __AIE_ARCH__ != 20 ? 1024 : 512;
        transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, true, kVectorBits>(
            out, N, [](auto v) { return vec_gelu(v); }, scalar_gelu, in);
    } else {
        transform_n(out, N, scalar_gelu, in);
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6)));
    };
#if __AIE_ARCH__ != 20
    // f32 in vectors, 32 lanes a step: the emulated f32 multiply costs the same at 16.
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, std::is_same_v<INPUT_DTYPE, f32>,
                            1024>(out, N, [](auto v) { return vec_hardsigmoid(v); }, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            v * std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6)));
    };
#if __AIE_ARCH__ != 20
    // As HARDSIGMOID, then x where the hard sigmoid is 1: the emulated product would turn an
    // infinite x into NaN.
    transform_vector_walk_n<tile_vector_aligned_v<OUTPUT_DTYPE>, std::is_same_v<INPUT_DTYPE, f32>,
                            1024>(
        out, N,
        [](auto v) {
            using V = decltype(v);
            using IV = aie::vector<int32_t, V::size()>;
            const IV h = aie::vector_cast<int32_t>(vec_hardsigmoid(v));
            const IV x = aie::vector_cast<int32_t>(v);
            const V y = aie::select(
                aie::mul(v, aie::vector_cast<float>(h)).template to_vector<float>(), v,
                aie::eq(h, aie::broadcast<int32_t, V::size()>(__builtin_bit_cast(int32_t, 1.0f))));
            // A zero product of a finite negative x (bits below -inf's as int32) is -0; the
            // emulated product gives +0.
            return aie::select(
                y, aie::vector_cast<float>(aie::broadcast<int32_t, V::size()>(INT32_MIN)),
                aie::eq(aie::vector_cast<int32_t>(y), aie::zeros<int32_t, V::size()>()) &
                    aie::lt(x, aie::broadcast<int32_t, V::size()>(
                                   __builtin_bit_cast(int32_t, -INFINITY))));
        },
        scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
        }
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) : static_cast<int32>(v) - 1);
    };
#if __AIE_ARCH__ != 20
    round_integral_n<round_integral::floor>(out, N, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
        }
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) + 1 : static_cast<int32>(v));
    };
#if __AIE_ARCH__ != 20
    round_integral_n<round_integral::ceil>(out, N, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(
            (v >= static_cast<INPUT_DTYPE>(0))
                ? static_cast<int32>(v + static_cast<INPUT_DTYPE>(.5))
                : static_cast<int32>(v - static_cast<INPUT_DTYPE>(.5)));
    };
#if __AIE_ARCH__ != 20
    round_integral_n<round_integral::round>(out, N, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
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
    const auto scalar_op = [](auto v) -> OUTPUT_DTYPE {
        return static_cast<OUTPUT_DTYPE>(static_cast<int32>(v));
    };
#if __AIE_ARCH__ != 20
    round_integral_n<round_integral::trunc>(out, N, scalar_op, in);
#else
    transform_n(out, N, scalar_op, in);
#endif
}

#endif // GGML_UNARY_OP_TRUNC

} // extern "C"
