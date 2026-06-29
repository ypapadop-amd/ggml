// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

#include <hsa/hsa.h>

/**
 * @brief Kernel for GPU agents (ROCm/HIP code objects loaded from @c .hsaco files).
 *
 * The kernel is dispatched using a standard HSA AQL kernel dispatch packet. The
 * code object is loaded into an @c hsa_executable_t at creation time; the kernel
 * descriptor handle and segment sizes are queried from the resolved symbol.
 */
class ggml_hsa_gpu_kernel : public ggml_hsa_kernel {
  public:
    hsa_executable_t executable{};        ///< Loaded executable owning the code object.
    std::uint64_t kernel_object{};        ///< HSA kernel descriptor handle.
    std::uint32_t private_segment_size{}; ///< Kernel private segment size in bytes.
    std::uint32_t group_segment_size{};   ///< Kernel group (LDS) segment size in bytes.
    std::uint32_t kernarg_size{};         ///< Kernel argument segment size in bytes.
    std::uint32_t kernarg_align{};        ///< Kernel argument segment alignment in bytes.
    std::uint32_t work_group_size{64};    ///< Workgroup size (work-items per group, dim x).

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
