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

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_hsa_host_buffer_type(void);

GGML_BACKEND_API int32_t ggml_backend_hsa_get_device_count(void);
GGML_BACKEND_API void ggml_backend_hsa_get_device_description(int32_t device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_hsa_get_device_memory(int32_t device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_hsa_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_hsa_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hsa_reg(void);

#ifdef  __cplusplus
}
#endif