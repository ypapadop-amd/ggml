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

// Builds and dispatches an internal single-input transform kernel on the HSA backend, then waits
// for completion. Compiles (or fetches from cache) the kernel named op_name for the (src, dst)
// shape/dtype pair, dispatches it with src as the sole source and dst as the destination. src and
// dst must already be allocated on backend (device-resident data pointers). This is the same
// builder used by graph_compute, so the test path matches production. Intended for tests that drive
// individual internal kernels (e.g. the MUL_MAT convert/pad pre-amble "HSA_CONVERT_PAD" and de-pad
// post-amble "HSA_DEPAD") which are not reachable through ggml_backend_graph_compute.
GGML_BACKEND_API enum ggml_status ggml_hsa_test_dispatch_transform(
    ggml_backend_t backend, const char * op_name, const struct ggml_tensor * src, struct ggml_tensor * dst);

#ifdef  __cplusplus
}
#endif