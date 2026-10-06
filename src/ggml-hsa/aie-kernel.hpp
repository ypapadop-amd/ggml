// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

/**
 * @brief Kernel for AIE agents.
 *
 * The kernel is loaded from an hsaco (HSA code object) holding an AIE section. The kernel owns the
 * executable it was loaded into, which keeps its kernel object valid.
 */
class ggml_hsa_aie_kernel : public ggml_hsa_kernel {
    hsa_code_object_reader_t m_reader{};
    hsa_executable_t m_executable{};
    std::uint64_t m_kernel_object{};
    std::uint32_t m_kernarg_size{};

  public:
    ggml_hsa_aie_kernel() = default;
    ggml_hsa_aie_kernel(const ggml_hsa_aie_kernel &) = delete;
    ggml_hsa_aie_kernel & operator=(const ggml_hsa_aie_kernel &) = delete;
    ~ggml_hsa_aie_kernel() override;

    /**
     * @brief Loads kernel @p kernel_name from the hsaco at @p path on @p agent.
     *
     * @param[in] agent AIE agent to load the kernel on
     * @param[in] path hsaco path
     * @param[in] kernel_name kernel symbol name
     * @param[out] kernel loaded kernel
     */
    static ggml_status load(hsa_agent_t agent,
                            const std::filesystem::path & path,
                            const std::string & kernel_name,
                            std::shared_ptr<ggml_hsa_aie_kernel> & kernel);

    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
