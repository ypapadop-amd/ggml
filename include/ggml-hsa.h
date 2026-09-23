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

// HSA-only graph operators.
//
// These build a single-node result whose op is one of the HSA-only operators (see enum ggml_hsa_op
// in the backend). They are the internal MUL_MAT convert/pad pre-amble and de-pad post-amble, plus
// the element-wise dtype cast, exposed as ordinary ggml ops so they can be driven through
// ggml_build_forward_expand + ggml_backend_graph_compute like any other op. They are only supported
// by the HSA backend.

// dtype-convert `a` to `type` and zero-pad it into the given (larger or equal) 2D shape.
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_convert_pad(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type, int64_t ne0,
    int64_t ne1);

// strip the zero-padding from `a`, gathering the top-left sub-block into the given (smaller or
// equal) 2D shape and converting it to `type`.
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_depad(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type, int64_t ne0,
    int64_t ne1);

// element-wise dtype cast of `a` to `type` (same shape).
GGML_BACKEND_API struct ggml_tensor * ggml_hsa_convert(
    struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type);

#ifdef  __cplusplus
}
#endif