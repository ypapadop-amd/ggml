// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"

#include "ggml.h"

/**
 * @brief Copy the contents of tensor @p src to tensor @p dst.
 *
 * @warning This function does not synchronize with any device.
 *
 * @param[in] src tensor to copy from
 * @param[in] dst tensor to copy to
 */
ggml_status ggml_hsa_copy_tensor(const ggml_tensor * src, ggml_tensor * dst);

/**
 * @brief Copy the overlapping sub-block of @p src into @p dst, converting the datatype if needed.
 *
 * Unlike @ref ggml_hsa_copy_tensor, @p src and @p dst may have different shapes. Only the
 * per-dimension overlap (@c min(src->ne[i], dst->ne[i])) is copied; both tensors are indexed
 * through their own strides, so a smaller logical tensor can be scattered into a larger,
 * zero-padded destination (or a padded tensor gathered back into a smaller one). Padding gaps in
 * @p dst are left untouched and must be pre-zeroed by the caller.
 *
 * @warning This function does not synchronize with any device.
 *
 * @param[in] src tensor to copy from
 * @param[in] dst tensor to copy to
 */
ggml_status ggml_hsa_copy_subblock(const ggml_tensor * src, ggml_tensor * dst);

/**
 * @brief Duplicate tensor @c t->src[0] to @p t without changing the shape or the datatype of the
 * tensor.
 *
 * @note @p t may be a view of @c t->src[0], in which case the operation is a no-op.
 */
ggml_status ggml_hsa_compute_dup(ggml_backend_hsa_context & ctx, ggml_tensor * t);

/**
 * @brief Copy tensor @c t->src[0] to @c t->src[1] without changing the layout of the tensor.
 *
 * @note This operation may change the datatype between @c t->src[0] and @c t->src[1] (e.g.,
 * @ref ggml_cast is a special case of this operation).
 */
ggml_status ggml_hsa_compute_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * t);

/**
 * @brief Store a copy of @c t->src[0] to @p t, where @p t has contiguous storage.
 *
 * @note This operation may change the layout of @c t->src[0] but it does not change the datatype.
 */
ggml_status ggml_hsa_compute_cont(ggml_backend_hsa_context & ctx, ggml_tensor * t);

/**
 * @brief Gather rows of @c t->src[0] selected by the int32 index tensor @c t->src[1] into @p t
 * (GGML_OP_GET_ROWS), converting the datatype if needed.
 *
 * For each of the @c nr = ggml_nelements(t->src[1]) output rows, reads the int32 row index and
 * copies the @c ne00 elements of the selected @c t->src[0] row into the corresponding @p t row,
 * matching @c ggml_compute_forward_get_rows.
 */
ggml_status ggml_hsa_compute_get_rows(ggml_backend_hsa_context & ctx, ggml_tensor * t);
