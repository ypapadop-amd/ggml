// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include <aie_api/aie.hpp>
#include "ggml-aie.hpp"

extern "C" {

#ifdef COMPILE_GGML_OP_SOFTMAX
// Softmax without mask or positional encoding
void ggml_op_softmax(
    const INPUT_DTYPE * __restrict in,
    OUTPUT_DTYPE * __restrict out,
    int32_t N,
    float scale,
    float max_bias)
{
}
#endif // COMPILE_GGML_OP_SOFTMAX

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MAX
// Softmax with mask tensor
void ggml_op_softmax_with_mask(
    const INPUT_DTYPE * __restrict in,
    const MASK_DTYPE * __restrict mask,
    OUTPUT_DTYPE * __restrict out,
    int32_t N,
    float scale,
    float max_bias)
{
}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MAX

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_POS
// Softmax with mask and positional encoding tensors
void ggml_op_softmax_with_mask_and_pos(
    const INPUT_DTYPE * __restrict in,
    const MASK_DTYPE * __restrict mask,
    const POS_DTYPE * __restrict pos,
    OUTPUT_DTYPE * __restrict out,
    int32_t N,
    float scale,
    float max_bias)
{
}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_POS

} // extern "C"