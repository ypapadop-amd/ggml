// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief Base class for kernels loaded from an hsaco (HSA code object).
 *
 * The kernel owns the executable it was loaded into, which keeps its kernel object valid.
 * Loading the executable (see @ref load_executable) is the same for every agent type. Which
 * kernel symbol inside it to dispatch, and what else to read from it, is agent-type-specific and
 * is left to each derived class: @ref ggml_hsa_aie_kernel finds it by kind (a kernel packed from
 * a full ELF keeps the ELF's own name), @ref ggml_hsa_gpu_kernel by name, read from the code
 * object's comgr metadata. Dispatch is also agent-type-specific.
 */
class ggml_hsa_kernel {
    hsa_executable_t m_executable{};
    std::uint64_t m_kernel_object{};
    std::uint32_t m_kernarg_size{};

  public:
    ggml_hsa_kernel() = default;
    ggml_hsa_kernel(const ggml_hsa_kernel &) = delete;
    ggml_hsa_kernel & operator=(const ggml_hsa_kernel &) = delete;
    virtual ~ggml_hsa_kernel();

    /// @brief Returns the executable loaded by @ref load_executable, or a zero handle before then.
    hsa_executable_t executable() const { return m_executable; }

    /// @brief Returns the kernel object handle, once a derived class has resolved it.
    std::uint64_t kernel_object() const { return m_kernel_object; }

    /// @brief Returns the kernel argument segment size in bytes, once a derived class has
    /// resolved it.
    std::uint32_t kernarg_size() const { return m_kernarg_size; }

    /// @brief Sets the kernel object handle a derived class resolved from the loaded executable.
    void set_kernel_object(std::uint64_t kernel_object) { m_kernel_object = kernel_object; }

    /// @brief Sets the kernel argument segment size a derived class resolved.
    void set_kernarg_size(std::uint32_t kernarg_size) { m_kernarg_size = kernarg_size; }

    /**
     * @brief Loads the hsaco at @p path into an executable on @p agent.
     *
     * Reads the file into @p blob, creates a code object reader from it, and creates, loads and
     * freezes an executable from that reader. The reader is needed only for that call: once the
     * executable is frozen, it has ingested whatever it needs from the code object, so the
     * reader is destroyed before this returns and is not kept as kernel state.
     *
     * Single-use: fails if this kernel already has an executable loaded.
     *
     * @param[in] agent agent to load the executable on
     * @param[in] path hsaco path
     * @param[in] kernel_name ggml's name for the kernel, used in error messages
     * @param[out] blob the hsaco's raw bytes, for a derived class that still needs to read the
     * file itself (e.g. GPU's comgr metadata parsing); unused bytes if the derived class does not
     */
    ggml_status load_executable(hsa_agent_t agent,
                                const std::filesystem::path & path,
                                const std::string & kernel_name,
                                std::vector<char> & blob);

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
