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
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in0[i], in1[i]);
    }
    event1();
}

extern "C" {

#ifdef GGML_OP_ADD

void ggml_op_add(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a + b; });
}

#endif // GGML_OP_ADD

#ifdef GGML_OP_SUB

void ggml_op_sub(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a - b; });
}

#endif // GGML_OP_SUB

#ifdef GGML_OP_MUL

void ggml_op_mul(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a * b; });
}

#endif // GGML_OP_MUL

#ifdef GGML_OP_DIV

void ggml_op_div(const INPUT0_DTYPE * __restrict in0,
                 const INPUT1_DTYPE * __restrict in1,
                 OUTPUT_DTYPE * __restrict out,
                 int32_t N) {
    transform_binary_n(in0, in1, N, out, [](auto a, auto b) -> OUTPUT_DTYPE { return a / b; });
}

#endif // GGML_OP_DIV

} // extern "C"
