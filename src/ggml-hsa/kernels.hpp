#pragma once

#include <cstddef>

#include "ggml-hsa/common.hpp"
#include "ggml.h"

bool ggml_hsa_supports_add(const ggml_hsa_device_info::device_info & dev_info,
                           const ggml_tensor * tensor);
ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_mul_mat(const ggml_hsa_device_info::device_info & dev_info,
                               const ggml_tensor * tensor);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
