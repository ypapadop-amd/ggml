// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>

#include "ggml-aie.hpp"

template <typename T0, typename T1, typename TOut, typename Size, typename BinaryOp>
void transform_binary_n(const T0 * __restrict in0,
                        const T1 * __restrict in1,
                        Size count,
                        TOut * __restrict out,
                        BinaryOp op) {
    event0();

    // Use AIE vector operations to process multiple elements per iteration.
    constexpr int kVectorLength = 16;

    Size i = 0;

    // Vectorized main loop
    for (; i + kVectorLength <= count; i += kVectorLength) {
        aie::vector<T0, kVectorLength> v_in0 = aie::load_v<kVectorLength>(in0 + i);
        aie::vector<T1, kVectorLength> v_in1 = aie::load_v<kVectorLength>(in1 + i);
        aie::vector<TOut, kVectorLength> v_out;

        for (int lane = 0; lane < kVectorLength; ++lane) {
            v_out[lane] = op(v_in0[lane], v_in1[lane]);
        }

        aie::store_v(out + i, v_out);
    }

    // Scalar tail for remaining elements
    for (; i < count; ++i) {
        out[i] = op(in0[i], in1[i]);
    }
    event1();
}

extern "C" {

#ifdef COMPILE_ADD

void ggml_op_add(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a + b; });
}

#endif // COMPILE_ADD

#ifdef COMPILE_SUB

void ggml_op_sub(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a - b; });
}

#endif // COMPILE_SUB

#ifdef COMPILE_MUL

void ggml_op_mul(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a * b; });
}

#endif // COMPILE_MUL

#ifdef COMPILE_DIV

void ggml_op_div(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a / b; });
}

#endif // COMPILE_DIV

} // extern "C"
