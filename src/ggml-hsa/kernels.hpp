#pragma once

#include <cstddef>

#include "ggml.h"
#include "ggml-hsa/common.hpp"

/**
 * @brief Returns if the kernel exists.
 *
 * @param tensor tensor to load a kernel for
 */
bool ggml_hsa_kernel_exists(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor);

/**
 * @brief Finds the AIE agent kernel for the tensor's operation.
 *
 * This function will attempt to load the kernel if not found in @ref ggml_backend_hsa_context::aie_kernels.
 *
 * @param ctx backend context
 * @param tensor tensor to find the kernel for
 * @param kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_find_aie_kernel(ggml_backend_hsa_context & ctx, const ggml_tensor * tensor, ggml_hsa_aie_kernel & kernel);

/**
 * @brief Destroys the kernel.
 *
 * @param ctx backend context
 * @param kernel kernel to destroy
 */
void ggml_hsa_destroy_aie_kernel(ggml_backend_hsa_context & ctx, ggml_hsa_aie_kernel & kernel);

bool ggml_hsa_supports_add(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor);
ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_cpy(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor);
ggml_status ggml_hsa_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_mul_mat(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
