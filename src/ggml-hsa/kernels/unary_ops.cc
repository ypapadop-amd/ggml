// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

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

#ifdef COMPILE_FLOOR

void ggml_op_floor(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v)
                                                  : static_cast<int32>(v) - 1;
    });
}

#endif // COMPILE_FLOOR

#ifdef COMPILE_CEIL

void ggml_op_ceil(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        if (v == static_cast<int32>(v)) {
            return static_cast<int32>(v);
        }
        return (v >= static_cast<INPUT_DTYPE>(0)) ? static_cast<int32>(v) + 1
                                                  : static_cast<int32>(v);
    });
}

#endif // COMPILE_CEIL

#ifdef COMPILE_ROUND

void ggml_op_round(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE {
        return (v >= static_cast<INPUT_DTYPE>(0))
                   ? static_cast<int32>(v + static_cast<INPUT_DTYPE>(.5))
                   : static_cast<int32>(v - static_cast<INPUT_DTYPE>(.5));
    });
}

#endif // COMPILE_ROUND

#ifdef COMPILE_TRUNC

void ggml_op_trunc(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    static_assert(is_floating_point_v<INPUT_DTYPE>, "Input type must be a floating point type");
    transform_n(in, N, out, [](auto v) -> OUTPUT_DTYPE { return static_cast<int32>(v); });
}

#endif // COMPILE_TRUNC

} // extern "C"
