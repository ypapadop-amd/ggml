// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/kernel.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <hsa/hsa.h>

/// Size of a GPU kernarg slot in bytes: the explicit arguments plus the HIP hidden arguments.
inline constexpr std::size_t ggml_hsa_gpu_kernarg_slot_size = 1024;

/// Alignment of a GPU kernarg slot in bytes; covers the kernarg alignment of HIP kernels.
inline constexpr std::size_t ggml_hsa_gpu_kernarg_alignment = 64;

/**
 * @brief Description of a single kernel argument, read from the code object metadata.
 *
 * Offsets and sizes come from the compiled kernel itself (via comgr) rather than a
 * hand-maintained struct, so the kernarg layout is robust to ABI/architecture changes.
 */
struct ggml_hsa_gpu_kernel_arg {
    std::string name;       ///< Argument name (explicit args only; empty for hidden args).
    std::string value_kind; ///< AMDGPU value kind, e.g. "global_buffer", "hidden_block_count_x".
    std::uint32_t offset{}; ///< Byte offset of the argument within the kernarg segment.
    std::uint32_t size{};   ///< Argument size in bytes.
};

/// @name Types of the arguments @ref ggml_hsa_gpu_kernel::dispatch fills from fixed-size values
/// @{
using ggml_hsa_gpu_pointer_arg_t = std::uint64_t;       ///< @c global_buffer
using ggml_hsa_gpu_element_count_arg_t = std::uint64_t; ///< the first @c by_value argument
using ggml_hsa_gpu_block_count_arg_t = std::uint32_t;   ///< @c hidden_block_count_{x,y,z}
/// @c hidden_group_size_{x,y,z}, @c hidden_remainder_x and @c hidden_grid_dims
using ggml_hsa_gpu_group_dim_arg_t = std::uint16_t;
/// @}

/**
 * @brief Returns the size @ref ggml_hsa_gpu_kernel::dispatch writes for an argument, or 0 if it
 * does not fill the argument from a fixed-size value.
 *
 * The single source of the sizes both the load-time check and the dispatch use.
 *
 * @param[in] value_kind     argument kind from the code object metadata
 * @param[in] by_value_index position of the argument among the @c by_value arguments; only the
 *                           first, the element count, has a fixed size
 */
inline std::uint32_t ggml_hsa_gpu_fixed_arg_size(std::string_view value_kind,
                                                 std::size_t by_value_index) {
    if (value_kind == "global_buffer") {
        return sizeof(ggml_hsa_gpu_pointer_arg_t);
    }
    if (value_kind == "by_value") {
        return by_value_index == 0 ? sizeof(ggml_hsa_gpu_element_count_arg_t) : 0;
    }
    if (value_kind == "hidden_block_count_x" || value_kind == "hidden_block_count_y" ||
        value_kind == "hidden_block_count_z") {
        return sizeof(ggml_hsa_gpu_block_count_arg_t);
    }
    if (value_kind == "hidden_group_size_x" || value_kind == "hidden_group_size_y" ||
        value_kind == "hidden_group_size_z" || value_kind == "hidden_remainder_x" ||
        value_kind == "hidden_grid_dims") {
        return sizeof(ggml_hsa_gpu_group_dim_arg_t);
    }
    return 0;
}

/**
 * @brief Kernel for GPU agents (ROCm/HIP code objects loaded from @c .hsaco files).
 *
 * The kernel is dispatched using a standard HSA AQL kernel dispatch packet. The code object is
 * loaded into an executable with @ref ggml_hsa_kernel::load_executable at creation time; the
 * kernel descriptor handle and kernarg segment size are the base class's (resolved by name, from
 * the code object's comgr metadata, rather than by kind as @ref ggml_hsa_aie_kernel does); the
 * remaining segment sizes and the argument layout below are GPU-specific and have no base-class
 * equivalent.
 */
class ggml_hsa_gpu_kernel : public ggml_hsa_kernel {
  public:
    std::uint32_t private_segment_size{}; ///< Kernel private segment size in bytes.
    std::uint32_t group_segment_size{};   ///< Kernel group (LDS) segment size in bytes.
    std::uint32_t kernarg_align{};        ///< Kernel argument segment alignment in bytes.
    std::uint32_t work_group_size{64};    ///< Default workgroup size when launch geometry is unset.
    std::uint32_t grid_size_x{}; ///< Fixed AQL grid size (total work-items); 0 = derive from N.
    std::uint32_t workgroup_size_x{}; ///< Fixed workgroup size (work-items/group); 0 = use default.
    std::uint32_t dynamic_group_segment_size{}; ///< Dynamic LDS in bytes, from the sidecar.
    std::vector<ggml_hsa_gpu_kernel_arg> args; ///< Argument layout from code object metadata.

    ggml_hsa_gpu_kernel() = default;

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
