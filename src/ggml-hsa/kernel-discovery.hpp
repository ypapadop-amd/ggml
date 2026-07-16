// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef GGML_HSA_GPU
#include <hsa/hsa.h>

/**
 * @brief A GPU kernel resolved from a loaded HSA code object.
 *
 * Holds the descriptor handle and segment sizes queried from a code-object symbol. It is the
 * shared result type of the low-level loader helpers below, reused by both the native single-kernel
 * loader (@ref ggml_hsa_create_kernel) and the HIP host ABI (hip-abi.cpp), which resolves many
 * kernels from a single fatbin.
 */
struct ggml_hsa_loaded_kernel {
    hsa_executable_t executable{};        ///< Executable owning the code object (0 if not owned here).
    std::uint64_t    kernel_object{};     ///< HSA kernel descriptor handle.
    std::uint32_t    private_segment_size{}; ///< Private (scratch) segment size in bytes.
    std::uint32_t    group_segment_size{};   ///< Group (LDS) segment size in bytes.
    std::uint32_t    kernarg_size{};      ///< Kernel argument segment size in bytes.
    std::uint32_t    kernarg_align{16};   ///< Kernel argument segment alignment (floored to 16).
};

/**
 * @brief Loads and freezes an in-memory AMDGPU code object into a new HSA executable.
 *
 * Wraps the native loader sequence (@c hsa_code_object_reader_create_from_memory ->
 * @c hsa_executable_create_alt -> @c hsa_executable_load_agent_code_object ->
 * @c hsa_executable_freeze). The caller owns @p executable and must destroy it.
 *
 * @param[in]  dev_info    device whose agent the code object is loaded for
 * @param[in]  code_object pointer to the code-object (ELF) bytes
 * @param[in]  size        size of the code object in bytes
 * @param[out] executable  frozen executable on success
 * @return @c true on success
 */
bool ggml_hsa_load_executable(const ggml_hsa_device_info::device_info & dev_info,
                              const void * code_object, std::size_t size,
                              hsa_executable_t & executable);

/**
 * @brief Resolves a kernel descriptor symbol in a frozen executable and queries its layout.
 *
 * @param[in]  dev_info   device whose agent the symbol is resolved for
 * @param[in]  executable frozen executable (from @ref ggml_hsa_load_executable)
 * @param[in]  symbol     kernel descriptor symbol (e.g. "vecadd.kd")
 * @param[out] out        resolved kernel object + segment sizes (executable field left untouched)
 * @return @c true on success
 */
bool ggml_hsa_resolve_kernel_symbol(const ggml_hsa_device_info::device_info & dev_info,
                                    hsa_executable_t executable, const std::string & symbol,
                                    ggml_hsa_loaded_kernel & out);

/**
 * @brief Loads a code object and resolves a single kernel symbol in one call.
 *
 * Convenience wrapper over @ref ggml_hsa_load_executable + @ref ggml_hsa_resolve_kernel_symbol used
 * by the native single-kernel loader. On success @p out.executable owns the loaded code object.
 *
 * @param[in]  dev_info    device whose agent the code object is loaded for
 * @param[in]  code_object pointer to the code-object (ELF) bytes
 * @param[in]  size        size of the code object in bytes
 * @param[in]  symbol      kernel descriptor symbol to resolve
 * @param[out] out         loaded executable + resolved kernel
 * @return @c true on success
 */
bool ggml_hsa_load_code_object(const ggml_hsa_device_info::device_info & dev_info,
                               const void * code_object, std::size_t size,
                               const std::string & symbol, ggml_hsa_loaded_kernel & out);
#endif // GGML_HSA_GPU

/**
 * @brief Creates the kernel for the tensor's operation.
 *
 * This function may try different approaches until one succeeds:
 *   -# load the kernel from a precompiled kernel directory,
 *   -# load the kernel from a cached kernel directory,
 *   -# compile the kernel, store it to the cached kernel directory, and load it.
 * If none of the above succeeds, an error message will be returned.
 *
 * @param[in] dev_info device information
 * @param[in] kernel_name kernel name
 * @param[in] tensor tensor to find the kernel for
 * @param[out] kernel kernel for the operation of @p tensor
 */
ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const std::string & kernel_name,
                                   const ggml_tensor & tensor,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel);
