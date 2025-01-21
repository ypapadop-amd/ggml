#pragma once

#include "ggml.h"
#include "ggml-hsa/common.hpp"

bool ggml_hsa_supports_cpy(const ggml_tensor * tensor);
ggml_status ggml_hsa_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_mul_mat(const ggml_tensor * tensor);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
