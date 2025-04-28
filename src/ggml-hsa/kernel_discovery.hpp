// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

/**
 * @brief Returns if the kernel exists for the device and tensor.
 *
 * @param dev_info[in] device information
 * @param tensor[in] tensor to load a kernel for
 */
bool ggml_hsa_kernel_exists(const ggml_hsa_device_info::device_info & dev_info,
                            const ggml_tensor * tensor);

/**
 * @brief Creates the AIE kernel for the tensor's operation.
 *
 * This function will try the following until one succeeds:
 *   -# attempt to load the kernel from the @ref ctx kernel cache in @ref
 *      ggml_backend_hsa_context::aie_kernels,
 *   -# load the kernel from disk from the system kernel directory,
 *   -# load the kernel from disk from the user kernel directory,
 *   -# JIT compile the kernel and store it to the user kernel directory.
 * If nothing works, an error message will be returned.
 *
 * @param[in] ctx backend context
 * @param[in] tensor tensor to find the kernel for
 * @param[out] kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_create_aie_kernel(ggml_backend_hsa_context & ctx,
                                       const ggml_tensor * tensor,
                                       ggml_hsa_aie_kernel & kernel);

/**
 * @brief Destroys the kernel.
 *
 * @param[in] ctx backend context
 * @param[in] kernel kernel to destroy
 */
void ggml_hsa_destroy_aie_kernel(ggml_backend_hsa_context & ctx, ggml_hsa_aie_kernel & kernel);
