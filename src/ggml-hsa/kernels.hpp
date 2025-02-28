#pragma once

#include <cstddef>

#include "ggml.h"
#include "ggml-hsa/common.hpp"

/**
 * @brief PDI buffer.
 */
struct ggml_hsa_pdi_buffer {
  std::uint64_t * data{};
  std::size_t size{};
};

/**
 * @brief Instructions buffer.
 */
struct ggml_hsa_instr_buffer {
  std::uint32_t * data{};
  std::size_t size{};
};

/**
 * @brief Returns if the kernel exists.
 *
 * @param tensor tensor to load a kernel for
 */
bool ggml_hsa_kernel_exists(const ggml_tensor * tensor);

/**
 * @brief Loads the kernel for the tensor's operation.
 *
 * @param ctx backend context
 * @param tensor tensor to load a kernel for
 * @param pdi_buf PDI buffer
 * @param instr_buf instructions buffer
 */
ggml_status ggml_hsa_load_kernel(ggml_backend_hsa_context & ctx, const ggml_tensor * tensor, ggml_hsa_pdi_buffer & pdi_buf, ggml_hsa_instr_buffer & instr_buf);

bool ggml_hsa_supports_add(const ggml_tensor * tensor);
ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_cpy(const ggml_tensor * tensor);
ggml_status ggml_hsa_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

bool ggml_hsa_supports_mul_mat(const ggml_tensor * tensor);
ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);
