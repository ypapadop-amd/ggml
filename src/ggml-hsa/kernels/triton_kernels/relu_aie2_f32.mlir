// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for ReLU (AIE2), f32 inputs.
// relu(x) = max(x, 0)
// Same strategy as relu_aie2.mlir but pads with an f32 zero instead of a bf16
// zero: @pad_and_promote_unary_bf16 hard-codes a bf16 padding value, which
// aircc rejects when the tensor element type is f32 ("expects a padding value
// of type 'f32', got 0.0 : bf16"). The pad body is inlined here with an f32
// padding value. The final @cast_bf16_only_ops is kept: AIE2 has no legal
// f32 vector max (aievec.max on vector<16xf32> fails to legalize), so the
// max is still computed in bf16.
// Uses shared library sequences from transform_library.mlir (auto-injected).
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    transform.include @canonicalize_with_fold_dims failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @fuse_elementwise_and_canonicalize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @flatten_tile_forall failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // f32 pad+promote (inlined @pad_and_promote_unary_bf16 with an f32 pad value).
    %op = transform.structured.match ops{["linalg.generic"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %padded_op, %pad_op, %__ = transform.structured.pad %op {
        padding_values=[0.0 : f32, 0.0 : f32],
        padding_dimensions=[0, 1],
        nofold_flags=[1, 1],
        copy_back_op="linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad_op
        : (!transform.any_op) -> !transform.any_op
    %padded_input = transform.get_producer_of_operand %padded_op[0]
        : (!transform.any_op) -> (!transform.any_op)
    %padded_input_buffer, %padded_input_new =
        transform.structured.bufferize_to_allocation %padded_input
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op
    %padded_result = transform.get_producer_of_operand %padded_op[1]
        : (!transform.any_op) -> (!transform.any_op)
    %padded_result_buffer, %padded_result_new =
        transform.structured.bufferize_to_allocation %padded_result
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    transform.include @vectorize_generics_at_16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op
    transform.include @cast_bf16_only_ops failures(propagate)
        (%vh) : (!transform.any_op) -> ()

    transform.yield
  }
}
