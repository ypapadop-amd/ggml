// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hsa/hsa.h>

/**
 * @brief Description of a single kernel argument, read from the code object metadata.
 *
 * Offsets and sizes come from the compiled kernel itself (via comgr) rather than a
 * hand-maintained struct, so the kernarg layout is robust to ABI/architecture changes.
 */
struct ggml_hsa_gpu_kernel_arg {
    std::string name;        ///< Argument name (explicit args only; empty for hidden args).
    std::string value_kind;  ///< AMDGPU value kind, e.g. "global_buffer", "hidden_block_count_x".
    std::uint32_t offset{};  ///< Byte offset of the argument within the kernarg segment.
    std::uint32_t size{};    ///< Argument size in bytes.
};

/**
 * @brief Kernel for GPU agents (ROCm/HIP code objects loaded from @c .hsaco files).
 *
 * The kernel is dispatched using a standard HSA AQL kernel dispatch packet. The
 * code object is loaded into an @c hsa_executable_t at creation time; the kernel
 * descriptor handle, segment sizes, and argument layout are queried from the
 * resolved symbol and the code object metadata.
 */
class ggml_hsa_gpu_kernel : public ggml_hsa_kernel {
  public:
    hsa_executable_t executable{};        ///< Loaded executable owning the code object.
    std::uint64_t kernel_object{};        ///< HSA kernel descriptor handle.
    std::uint32_t private_segment_size{}; ///< Kernel private segment size in bytes.
    std::uint32_t group_segment_size{};   ///< Kernel group (LDS) segment size in bytes.
    std::uint32_t kernarg_size{};         ///< Kernel argument segment size in bytes.
    std::uint32_t kernarg_align{};        ///< Kernel argument segment alignment in bytes.
    std::uint32_t work_group_size{64};    ///< Default workgroup size when launch geometry is unset.
    std::uint32_t grid_size_x{};          ///< Fixed AQL grid size (total work-items); 0 = derive from N.
    std::uint32_t workgroup_size_x{};     ///< Fixed workgroup size (work-items/group); 0 = use default.
    std::vector<ggml_hsa_gpu_kernel_arg> args; ///< Argument layout from code object metadata.

    ggml_hsa_gpu_kernel() = default;
    ~ggml_hsa_gpu_kernel() override;

    ggml_hsa_gpu_kernel(const ggml_hsa_gpu_kernel &) = delete;
    ggml_hsa_gpu_kernel & operator=(const ggml_hsa_gpu_kernel &) = delete;

    /**
     * @brief Dispatches the GPU kernel.
     *
     * @param[in] ctx backend context
     * @param[in] src_tensors source tensors
     * @param[in] num_src_tensors number of source tensors
     * @param[out] dst_tensor destination tensor
     */
    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
