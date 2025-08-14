// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <string>

/**
 * @brief Creates the AIE kernel for the tensor's operation.
 *
 * This function will try the following until one succeeds:
 *   -# load the kernel from the precompiled kernel directory,
 *   -# load the kernel from the cached kernel directory,
 *   -# JIT compile the kernel and store it to the cached kernel directory.
 * If none of the above succeeds, an error message will be returned.
 *
 * @param[in] dev_info device information
 * @param[in] kernel_name kernel name
 * @param[in] tensor tensor to find the kernel for
 * @param[out] kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_create_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                       const std::string & kernel_name,
                                       const ggml_tensor & tensor,
                                       ggml_hsa_aie_kernel & kernel);
