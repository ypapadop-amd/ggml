#pragma once

#include <cstddef>

#include "ggml-hsa/common.hpp"
#include "ggml.h"

ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
