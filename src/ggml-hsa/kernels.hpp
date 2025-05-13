// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <cstddef>

#include "ggml-hsa/common.hpp"
#include "ggml.h"

ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
ggml_status ggml_hsa_sub(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
ggml_status ggml_hsa_mul(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
ggml_status ggml_hsa_div(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
