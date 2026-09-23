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
 * @{
 */

/**
 * @brief Converts @p a to @p type and widens it into the given (larger or equal) 2D shape.
 *
 * The kernel writes the first @c a->ne[1] rows, zero-filling each one's tail columns
 * <tt>[a->ne[0], ne0)</tt>.
 *
 * @warning The trailing rows <tt>[a->ne[1], ne1)</tt> are NOT written. The backend does not zero
 * buffers at allocation, so when @p ne1 is greater than @c a->ne[1] the caller must pre-zero the
 * destination (e.g. with @c ggml_backend_tensor_memset) or those rows hold whatever was already in
 * the buffer.
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

#ifdef  __cplusplus
}
#endif