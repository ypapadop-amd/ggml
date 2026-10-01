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
 *   - the padded regions read as zero (pre-zeroed by @ref allocate_internal_storage), so the extra
 *     rows/cols and the interior K gap contribute nothing to the result.
 * The output node is left as the dense parent: the GEMM writes it directly.
 *
 * Each operand rewrite is skipped when the parent tensor already has the target dtype and
 * shape, which leaves that tensor pointing at the parent buffer (@c buffer_size stays 0) and elides
 * the corresponding convert/pad dispatch -- the common case for GEMMs whose dimensions are
 * already tile multiples (see @ref ggml_hsa_pad_gemm_operand for the measured cost of not doing
 * so).
 *
 * On aie2 an f32 B is not converted at all: the GEMM streams it as f32 and converts each tile on
 * the core. Once B is at least one column group wide the GEMM also reads it unpadded: it shifts its
 * last column group back to end at N and zeroes the K tail on the core, and, when M is at least one
 * row block, it shifts its last row block up to end at M as well. Such a GEMM has no B
 * pre-processing dispatch; only the constant A is still converted and padded, once.
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
