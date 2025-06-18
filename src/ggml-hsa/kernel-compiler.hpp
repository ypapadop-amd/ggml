// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <filesystem>

#include "ggml-hsa/common.hpp"
#include "ggml.h"

/**
 * @brief Returns if the kennel for the operation in @p tensor can be compiled for the device.
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to compile a kernel for
 */
bool ggml_hsa_can_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                 const ggml_tensor * tensor);

/**
 * @brief Compiles the kernel for the operation in @p tensor.
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to compile a kernel for
 * @param[in] kernel_name kernel name
 * @param[in] output_path directory to write kernel to
 */
ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & kernel_name,
                                    const std::filesystem::path & output_path);
