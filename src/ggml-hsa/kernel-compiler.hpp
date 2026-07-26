// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "ggml-hsa/common.hpp"
#include "ggml.h"

/**
 * @brief Compiles a kernel for the operation in @p tensor.
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to compile a kernel for
 * @param[in] op_name operation name; if provided, it overrides the default op name derived from the
 * tensor's operation type
 * @param[in] kernel_name kernel name
 * @param[in] output_path directory to write kernel to
 */
ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor & tensor,
                                    std::optional<std::string> op_name,
                                    const std::string & kernel_name,
                                    const std::filesystem::path & output_path);
