// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using bf16 = bfloat16;
using f32 = float;

template <typename T, typename Size, typename UnaryOp>
void transform_n(const T * __restrict in, Size count, T * __restrict out, UnaryOp op) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in[i]);
    }
    event1();
}

extern "C" {

#ifdef COMPILE_SQR

void ggml_op_sqr(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return v * v; });
}

#endif // COMPILE_SQR

#ifdef COMPILE_SQRT

void ggml_op_sqrt(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return aie::sqrt(v); });
}

#endif // COMPILE_SQRT

#ifdef COMPILE_ABS

void ggml_op_abs(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out,
                [](auto v) -> OUTPUT_DTYPE { return v < static_cast<INPUT_DTYPE>(0) ? -v : v; });
}

#endif // COMPILE_ABS

#ifdef COMPILE_SGN

void ggml_op_sgn(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return (v > static_cast<INPUT_DTYPE>(0))
                   ? static_cast<OUTPUT_DTYPE>(1)
                   : ((v < static_cast<INPUT_DTYPE>(0)) ? static_cast<OUTPUT_DTYPE>(-1)
                                                        : static_cast<OUTPUT_DTYPE>(0));
    });
}

#endif // COMPILE_SGN

#ifdef COMPILE_NEG

void ggml_op_neg(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return -v; });
}

#endif // COMPILE_NEG

#ifdef COMPILE_STEP

void ggml_op_step(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return v > 0; });
}

#endif // COMPILE_STEP

#ifdef COMPILE_RELU

void ggml_op_relu(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return std::max<INPUT_DTYPE>(v, 0); });
}

#endif // COMPILE_RELU

#ifdef COMPILE_HARDSIGMOID

void ggml_op_hardsigmoid(const INPUT_DTYPE * __restrict in,
                         OUTPUT_DTYPE * __restrict out,
                         int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6));
    });
}

#endif // COMPILE_HARDSIGMOID

#ifdef COMPILE_HARDSWISH

void ggml_op_hardswish(const INPUT_DTYPE * __restrict in,
                       OUTPUT_DTYPE * __restrict out,
                       int32_t N) {
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return v * std::min<INPUT_DTYPE>(1, std::max<INPUT_DTYPE>(0, (v + 3) / 6));
    });
}

#endif // COMPILE_HARDSWISH

} // extern "C"
