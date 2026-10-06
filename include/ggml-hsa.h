#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_HSA_NAME "HSA"
#define GGML_HSA_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hsa_init(int32_t device);

GGML_BACKEND_API bool ggml_backend_is_hsa(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_hsa_buffer_type(int32_t device);

/**
 * @brief Maps another ROCm backend's buffer (e.g., HIP) for the NPU, without copying.
 *
 * The caller synchronizes between the backends and frees the result before @p buffer.
 *
 * @param[in] device HSA device
 * @param[in] buffer buffer to import
 * @return HSA buffer over the same memory at the same offsets, or @c NULL on failure
 */
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_hsa_buffer_import(int32_t device, ggml_backend_buffer_t buffer);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_hsa_split_buffer_type(int32_t main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and HSA agent
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_hsa_host_buffer_type(void);

GGML_BACKEND_API int32_t ggml_backend_hsa_get_device_count(void);
GGML_BACKEND_API void ggml_backend_hsa_get_device_description(int32_t device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_hsa_get_device_memory(int32_t device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_hsa_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_hsa_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hsa_reg(void);

/**
 * @defgroup ggml_hsa_ops HSA-only graph operators
 *
 * These build a single-node result whose op is one of the HSA-only operators (see @c ggml_hsa_op
 * in the backend). They are the internal MUL_MAT convert/pad pre-amble and de-pad post-amble, plus
 * the element-wise dtype cast, exposed as ordinary ggml ops so they can be driven through
 * @c ggml_build_forward_expand + @c ggml_backend_graph_compute like any other op.
 *
 * @note These operators are only supported by the HSA backend.
 *
 * @warning The returned node carries an op value above @c GGML_OP_COUNT, which core ggml does not
 * expect. @c ggml_op_name indexes @c GGML_OP_NAME, an array of exactly @c GGML_OP_COUNT entries,
 * with no bounds check, and generic consumers reach it through @c ggml_op_desc -- notably
 * @c ggml_backend_sched's node dump, which runs when scheduler debug output is enabled
 * (@c GGML_SCHED_DEBUG > 1). The HSA backend itself is safe (it routes every such site through
 * @c ggml_hsa_op_name), but these nodes must not be handed to a scheduler with debug output on, or
 * to any other generic ggml diagnostic, until core grows a supported extension range.
 * @{
 */

/**
 * @brief Converts @p a to @p type and widens it into the given (larger or equal) 2D shape.
 *
 * Supported: f32 -> bf16 (convert and pad), and bf16 -> bf16 or f32 -> f32 (pad only).
 *
 * The kernel writes the whole destination: the first @c a->ne[1] rows with their tail columns
 * <tt>[a->ne[0], ne0)</tt> zero-filled, and the trailing rows <tt>[a->ne[1], ne1)</tt> zero-filled,
 * so the destination need not be pre-zeroed.
 *
 * @param[in] ctx  context to allocate the result in
 * @param[in] a    source tensor; must be 2D (@c ne[2] == @c ne[3] == 1)
 * @param[in] type datatype of the result
 * @param[in] ne0  padded row width, >= @c a->ne[0]
 * @param[in] ne1  padded row count, >= @c a->ne[1]
 * @return the result tensor, of shape <tt>[ne0, ne1, 1, 1]</tt>
 */
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_convert_pad(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type, int64_t ne0,
    int64_t ne1);

/**
 * @brief Strips the zero-padding from @p a, gathering the top-left sub-block into the given
 * (smaller or equal) 2D shape and converting it to @p type.
 *
 * @param[in] ctx  context to allocate the result in
 * @param[in] a    padded source tensor; must be 2D (@c ne[2] == @c ne[3] == 1)
 * @param[in] type datatype of the result
 * @param[in] ne0  unpadded row width, <= @c a->ne[0]
 * @param[in] ne1  unpadded row count, <= @c a->ne[1]
 * @return the result tensor, of shape <tt>[ne0, ne1, 1, 1]</tt>
 */
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_depad(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type, int64_t ne0,
    int64_t ne1);

/**
 * @brief Element-wise datatype cast of @p a to @p type, keeping the shape.
 *
 * @param[in] ctx  context to allocate the result in
 * @param[in] a    source tensor; must be dense and contiguous
 * @param[in] type datatype of the result
 * @return the result tensor, with the same shape as @p a
 */
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_convert(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);

/** @} */

/**
 * @brief Places @p tensor on the memory of @p src, so the HSA agent reads or writes @p src in place.
 *
 * For an op result, call before the graph is allocated.
 *
 * @param[in] imported result of @ref ggml_backend_hsa_buffer_import for @p src->buffer
 * @param[in,out] tensor unallocated tensor with the type, shape and strides of @p src
 * @param[in] src tensor to alias
 * @return @c GGML_STATUS_SUCCESS, or @c GGML_STATUS_FAILED if the arguments do not match
 */
GGML_BACKEND_API enum ggml_status ggml_backend_hsa_tensor_alloc_alias(ggml_backend_buffer_t imported, ggml_tensor * tensor, const ggml_tensor * src);

#ifdef  __cplusplus
}
#endif