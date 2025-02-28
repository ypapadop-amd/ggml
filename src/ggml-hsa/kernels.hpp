#pragma once

#include <cstddef>

#include "ggml.h"
#include "ggml-hsa/common.hpp"

/**
 * @brief PDI buffer.
 */
struct ggml_hsa_pdi_buffer {
    std::uint64_t * data{};
    std::size_t size{};
};

/**
 * @brief Instructions buffer.
 */
struct ggml_hsa_insts_buffer {
    std::uint32_t * data{};
    std::size_t size{};
};

/**
 * @brief NPU kernel.
 */
struct ggml_hsa_npu_kernel {
    ggml_hsa_pdi_buffer pdi_buffer;
    ggml_hsa_insts_buffer insts_buffer;
};

/**
 * @brief Returns if the kernel exists.
 *
 * @param tensor tensor to load a kernel for
 */
bool ggml_hsa_kernel_exists(const ggml_tensor * tensor);

/**
 * @brief Creates a kernel for the tensor's operation.
 *
 * @param ctx backend context
 * @param tensor tensor to create a kernel for
 * @param kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_create_kernel(ggml_backend_hsa_context & ctx, const ggml_tensor * tensor, ggml_hsa_npu_kernel & kernel);

/**
 * @brief Unloads the kernel.
 *
 * @param ctx backend context
 * @param kernel kernel to destroy
 */
void ggml_hsa_destroy_kernel(ggml_backend_hsa_context & ctx, ggml_hsa_npu_kernel & kernel);

bool ggml_hsa_supports_add(const ggml_tensor * tensor);
ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_cpy(const ggml_tensor * tensor);
ggml_status ggml_hsa_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_mul_mat(const ggml_tensor * tensor);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
