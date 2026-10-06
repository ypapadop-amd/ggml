// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/kernel.hpp"

#include <cstddef>

/**
 * @brief Kernel for AIE agents, dispatched with an AIE agent dispatch packet.
 */
class ggml_hsa_aie_kernel final : public ggml_hsa_kernel {
  public:
    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
