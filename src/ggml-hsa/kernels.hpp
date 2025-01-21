#pragma once

#include "ggml.h"
#include "ggml-hsa/common.hpp"

ggml_status ggml_hsa_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * node);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * node);
