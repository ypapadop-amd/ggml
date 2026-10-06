// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

/**
 * @brief Base class for kernels loaded from an hsaco (HSA code object).
 *
 * The kernel owns the executable it was loaded into, which keeps its kernel object valid. Loading
 * is the same for every agent type; dispatch is specific to the agent type and is implemented by
 * the derived classes.
 */
class ggml_hsa_kernel {
    hsa_code_object_reader_t m_reader{};
    hsa_executable_t m_executable{};
    std::uint64_t m_kernel_object{};
    std::uint32_t m_kernarg_size{};

  protected:
    /// @brief Returns the kernel object handle.
    std::uint64_t kernel_object() const { return m_kernel_object; }

    /// @brief Returns the kernel argument segment size in bytes.
    std::uint32_t kernarg_size() const { return m_kernarg_size; }

  public:
    ggml_hsa_kernel() = default;
    ggml_hsa_kernel(const ggml_hsa_kernel &) = delete;
    ggml_hsa_kernel & operator=(const ggml_hsa_kernel &) = delete;
    virtual ~ggml_hsa_kernel();

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

    /**
     * @brief Dispatches the kernel.
     *
     * @param[in] ctx backend context
     * @param[in] src_tensors source tensors
     * @param[in] num_src_tensors number of source tensors
     * @param[out] dst_tensor destination tensor
     */
    virtual ggml_status dispatch(ggml_backend_hsa_context & ctx,
                                 ggml_tensor * src_tensors[],
                                 std::size_t num_src_tensors,
                                 ggml_tensor & dst_tensor) const = 0;
};
