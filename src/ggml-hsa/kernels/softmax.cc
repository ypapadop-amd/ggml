// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-aie.hpp"
#include <aie_api/aie.hpp>

template <int VEC_SIZE>
inline aie::vector<float, VEC_SIZE> vec_exp(aie::vector<float, VEC_SIZE> x) {
    // Taylor series coefficients for exp(x)
    constexpr float C0 = 1.0f;
    constexpr float C1 = 1.0f;
    constexpr float C2 = 0.5f;                  // 1/2!
    constexpr float C3 = 0.1666666666666667f;   // 1/3!
    constexpr float C4 = 0.0416666666666667f;   // 1/4!
    constexpr float C5 = 0.0083333333333333f;   // 1/5!
    constexpr float C6 = 0.0013888888888889f;   // 1/6!
    constexpr float C7 = 0.0001984126984127f;   // 1/7!
    constexpr float C8 = 0.0000248015873016f;   // 1/8!
    constexpr float C9 = 0.0000027557319224f;   // 1/9!
    constexpr float C10 = 0.0000002755731922f;  // 1/10!
    constexpr float C11 = 0.0000000250521084f;  // 1/11!
    constexpr float C12 = 0.0000000020876757f;  // 1/12!
    constexpr float C13 = 0.0000000001605904f;  // 1/13!

    // clamp x to prevent overflow (exp(-88) ≈ 0, exp(88) ≈ inf)
    aie::vector<float, VEC_SIZE> v_clamp_min = aie::broadcast<float, VEC_SIZE>(-88.0f);
    aie::vector<float, VEC_SIZE> v_clamp_max = aie::broadcast<float, VEC_SIZE>(88.0f);
    x = aie::max(x, v_clamp_min);
    x = aie::min(x, v_clamp_max);

    // compute exp(x) using Horner's method

    aie::accum<accfloat, VEC_SIZE> tmp_accum;

    aie::vector<float, VEC_SIZE> poly = aie::broadcast<float, VEC_SIZE>(C13);

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C12));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C11));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C10));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C9));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C8));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C7));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C6));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C5));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C4));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C3));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C2));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();
    poly = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C1));

    tmp_accum = aie::mul(poly, x);
    poly = tmp_accum.template to_vector<float>();

    aie::vector<float, VEC_SIZE> result = aie::add(poly, aie::broadcast<float, VEC_SIZE>(C0));

    // clamp to positive (exp is always positive)
    aie::vector<float, VEC_SIZE> v_exp_min = aie::broadcast<float, VEC_SIZE>(1e-38f);
    result = aie::max(result, v_exp_min);

    return result;
}

extern "C" {
#ifdef COMPILE_GGML_OP_SOFTMAX
// Softmax without mask or sink
void ggml_op_softmax(const INPUT_DTYPE * __restrict in,
                     OUTPUT_DTYPE * __restrict out,
                     int32_t N,
                     float scale,
                     float max_bias) {
    event0();

    constexpr int VEC_SIZE = 16;
    const int num_iters = N >> 4; // N / 16

    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);

    // find max value for numerical stability

    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        v_max = aie::max(v_max, scaled_input);
    }

    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);

    // compute exp(x - max) and sum

    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;

        // apply scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();

        // subtract max for numerical stability
        aie::vector<float, VEC_SIZE> x = aie::sub(scaled_input, v_global_max);

        // compute exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);

        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);

        // store exp values
        *it_exp_out++ = exp_val;
    }

    // normalize by dividing by sum

    aie::vector<float, VEC_SIZE> v_sum_vec = v_sum_accum.to_vector<float>();
    float sum_total = aie::reduce_add(v_sum_vec);
    float sum_inv = aie::inv(sum_total);

    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> in_elems = *it_scale_in++;
        aie::accum<accfloat, VEC_SIZE> out_accum = aie::mul(in_elems, sum_inv);
        *it_soft_out++ = out_accum.to_vector<float>();
    }

    event1();
}

#endif // COMPILE_GGML_OP_SOFTMAX

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MASK
// Softmax with mask tensor
void ggml_op_softmax_with_mask(const INPUT_DTYPE * __restrict in,
                               const MASK_DTYPE * __restrict mask,
                               OUTPUT_DTYPE * __restrict out,
                               int32_t N,
                               float scale,
                               float max_bias) {
    event0();
    
    constexpr int VEC_SIZE = 16;
    const int num_iters = N >> 4;  // N / 16
    
    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);
    
    // compute max(scale * in + mask) across all elements

    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_max_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);
    
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_max_mask++;
        
        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        
        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);
        
        v_max = aie::max(v_max, masked_input);
    }
    
    float global_max = aie::reduce_max(v_max);
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);
    
    // compute exp(scale * in + mask - max) and accumulate sum
    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();
    
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_mask++;
        
        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        
        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);
        
        // x = masked_input - max (for numerical stability)
        aie::vector<float, VEC_SIZE> x = aie::sub(masked_input, v_global_max);
        
        // exp_val = exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);
        
        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);
        
        // store exp values for normalization pass
        *it_exp_out++ = exp_val;
    }
    
    // reduce sum across vector lanes
    aie::vector<float, VEC_SIZE> v_sum_vec = v_sum_accum.to_vector<float>();
    float sum_total = aie::reduce_add(v_sum_vec);
    float sum_inv = aie::inv(sum_total);
    
    // normalize by multiplying with 1/sum
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> in_elems = *it_scale_in++;
        aie::accum<accfloat, VEC_SIZE> out_accum = aie::mul(in_elems, sum_inv);
        *it_soft_out++ = out_accum.to_vector<float>();
    }
    
    event1();
}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MASK

#ifdef COMPILE_GGML_OP_SOFTMAX_WITH_MASK_AND_SINKS
// Softmax with mask and sink tensors
void ggml_op_softmax_with_mask_and_sinks(const INPUT_DTYPE * __restrict in,
                                         const MASK_DTYPE * __restrict mask,
                                         const SINK_DTYPE * __restrict sinks,
                                         OUTPUT_DTYPE * __restrict out,
                                         int32_t N,
                                         int32_t tile_idx,
                                         int32_t rows_per_head,
                                         float scale,
                                         float max_bias) {
    event0();
    
    constexpr int VEC_SIZE = 16;
    const int num_iters = N >> 4;  // N / 16
    
    // calculate which head this tile belongs to and get the sink value
    int32_t head_idx = tile_idx / rows_per_head;
    float sink_val = sinks[head_idx];
    
    auto it_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    auto it_exp_out = aie::begin_vector<VEC_SIZE>((float *)out);
    auto it_scale_in = aie::cbegin_restrict_vector<VEC_SIZE>((float *)out);
    auto it_soft_out = aie::begin_restrict_vector<VEC_SIZE>((float *)out);
    
    // find max value for numerical stability
    auto it_max_in = aie::cbegin_vector<VEC_SIZE>((float *)in);
    auto it_max_mask = aie::cbegin_vector<VEC_SIZE>((float *)mask);
    aie::vector<float, VEC_SIZE> v_max = aie::broadcast<float, VEC_SIZE>(-3.4028235e+38f);
    
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_max_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_max_mask++;
        
        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        
        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);
        
        v_max = aie::max(v_max, masked_input);
    }
    
    // reduce to scalar max, then include sink in max calculation
    float global_max = aie::reduce_max(v_max);
    global_max = (global_max > sink_val) ? global_max : sink_val;
    aie::vector<float, VEC_SIZE> v_global_max = aie::broadcast<float, VEC_SIZE>(global_max);
    
    // compute exp(scale * in + mask - max) and accumulate sum
    aie::accum<accfloat, VEC_SIZE> v_sum_accum = aie::zeros<accfloat, VEC_SIZE>();
    
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> input_vec = *it_in++;
        aie::vector<float, VEC_SIZE> mask_vec = *it_mask++;
        
        // scaled_input = in * scale
        aie::accum<accfloat, VEC_SIZE> scaled_accum = aie::mul(input_vec, scale);
        aie::vector<float, VEC_SIZE> scaled_input = scaled_accum.to_vector<float>();
        
        // masked_input = scaled_input + mask
        aie::vector<float, VEC_SIZE> masked_input = aie::add(scaled_input, mask_vec);
        
        // x = masked_input - max (for numerical stability)
        aie::vector<float, VEC_SIZE> x = aie::sub(masked_input, v_global_max);
        
        // exp_val = exp(x)
        aie::vector<float, VEC_SIZE> exp_val = vec_exp<VEC_SIZE>(x);
        
        // accumulate sum
        v_sum_accum = aie::add(v_sum_accum, exp_val);
        
        // store exp values for normalization pass
        *it_exp_out++ = exp_val;
    }
    
    // reduce sum across vector lanes
    aie::vector<float, VEC_SIZE> v_sum_vec = v_sum_accum.to_vector<float>();
    float sum_total = aie::reduce_add(v_sum_vec);
    
    // compute exp(sink - max) using vec_exp and add to sum
    float sink_shifted = sink_val - global_max;
    aie::vector<float, VEC_SIZE> sink_vec = aie::broadcast<float, VEC_SIZE>(sink_shifted);
    aie::vector<float, VEC_SIZE> sink_exp_vec = vec_exp<VEC_SIZE>(sink_vec);
    float sink_exp = sink_exp_vec.get(0);
    
    sum_total += sink_exp;
    float sum_inv = aie::inv(sum_total);
    
    // normalize by multiplying with 1/sum
    for (int i = 0; i < num_iters; i++) {
        aie::vector<float, VEC_SIZE> in_elems = *it_scale_in++;
        aie::accum<accfloat, VEC_SIZE> out_accum = aie::mul(in_elems, sum_inv);
        *it_soft_out++ = out_accum.to_vector<float>();
    }
    
    event1();
}
#endif // COMPILE_GGML_OP_SOFTMAX_WITH_MAX_AND_SINKS

} // extern "C"