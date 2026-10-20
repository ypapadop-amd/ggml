// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/kernel.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

/**
 * @brief Kernel for AIE agents, dispatched with an AIE agent dispatch packet.
 */
class ggml_hsa_aie_kernel final : public ggml_hsa_kernel {
  public:
    /**
     * @brief Loads the kernel in the hsaco at @p path on @p agent.
     *
     * The hsaco must hold exactly one kernel. It is found by kind, not by name, since a kernel
     * packed from a full ELF keeps the ELF's name.
     *
     * @param[in] agent agent to load the kernel on
     * @param[in] path hsaco path
     * @param[in] kernel_name ggml's name for the kernel, used in error messages
     */
    ggml_status
    load(hsa_agent_t agent, const std::filesystem::path & path, const std::string & kernel_name);

    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
