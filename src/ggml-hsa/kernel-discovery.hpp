// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <optional>
#include <string>

/**
 * @brief Creates the kernel for the tensor's operation.
 *
 * This function may try different approaches until one succeeds:
 *   -# load the kernel from a precompiled kernel directory,
 *   -# load the kernel from a cached kernel directory,
 *   -# compile the kernel, store it to the cached kernel directory, and load it.
 * If none of the above succeeds, an error message will be returned.
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to find the kernel for
 * @param[in] op_name operation name; if provided, it overrides the default op name derived from the
 * tensor's operation type (used for internal kernels such as the MUL_MAT convert/pad pre/post-amble
 * whose carrier tensor has no GGML op of its own)
 * @param[in] kernel_name kernel name
 * @param[out] kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const ggml_tensor & tensor,
                                   std::optional<std::string> op_name,
                                   const std::string & kernel_name,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel);
