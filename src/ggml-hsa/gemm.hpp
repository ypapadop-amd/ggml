// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"

/**
 * @brief Prepares an F32 @c MUL_MAT node for the AIE whole-array GEMM kernel.
 *
 * The GEMM microkernel has no native f32 path and the whole-array tiling requires the matrix
 * dimensions to be multiples of the per-architecture tile factors. This rewrites the internal node
 * (and its sources) so the kernel sees bf16 operands zero-padded up to those tile multiples:
 *   - src0 A = [K, M], src1 B = [K, N], dst C = [M, N] (GGML MUL_MAT layout);
 *   - each dimension is padded to K->Kpad, M->Mpad, N->Npad and the dtype set to bf16;
 *   - the padded regions are zero-filled by whatever fills the operand (the @c CONVERT_PAD
 *     dispatch, or the host sub-block scatter if that kernel is unavailable), so the extra
 *     rows/cols and the interior K gap contribute nothing to the result.
 * The output node keeps f32 but needs its own padded temporary storage plus a de-pad copy back into
 * the (smaller) parent tensor, flagged via @c node_t::transform.
 *
 * Each of the three rewrites is skipped when the parent tensor already has the target dtype and
 * shape, which leaves that tensor pointing at the parent buffer (@c buffer_size stays 0) and elides
 * the corresponding convert/pad/de-pad dispatch -- the common case for GEMMs whose dimensions are
 * already tile multiples (see @ref ggml_hsa_pad_gemm_operand for the measured cost of not doing
 * so).
 *
 * On aie2 an f32 B is not converted at all: the GEMM streams it as f32 and converts each tile on
 * the core. Once B is at least one column group wide the GEMM also reads it unpadded and, when M is
 * at least one row block and the output is f32, writes C in place: it shifts its last column group
 * and row block back to end at N and M, and zeroes the K tail on the core. Such a GEMM has neither
 * a B pre-processing nor a de-pad dispatch; only A is still converted and padded.
 *
 * Only the contiguous, non-batched, non-permuted f32 x f32 case is handled (the shapes exercised by
 * MNIST). Returns @c false for anything else, leaving the node untouched so the caller falls back
 * to the generic path.
 *
 * @param[in] dev_info device information (selects the tile factors)
 * @param[in,out] node internal output node
 * @param[in,out] sources internal source nodes
 * @return @c true if the node was rewritten for the padded bf16 GEMM path
 */
bool ggml_hsa_prepare_mul_mat_f32(const ggml_hsa_device_info::device_info & dev_info,
                                  ggml_backend_hsa_tensor_extra::node_t & node,
                                  ggml_backend_hsa_tensor_extra::sources_t & sources);
