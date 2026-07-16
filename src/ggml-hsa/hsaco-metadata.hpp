// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Neutral AMDGPU code-object metadata parser (comgr only, no ggml/HSA-context
// dependency). Shared by the ggml-hsa native GPU kernel loader
// (kernel-discovery.cpp) and the HIP shim (hip-shim/hip_shim.cpp) so the comgr
// "amdhsa.kernels" walk lives in exactly one place.
//
// It is compiled into BOTH the ggml-hsa and ggml-hip backend libraries: those
// libraries cannot depend on each other (ggml-hsa links ggml-hip for the
// ggml-cuda symbols), so a single shared source — rather than a cross-library
// call — is the de-duplication that avoids a circular dependency.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief A single kernel argument descriptor parsed from code-object metadata.
 */
struct ggml_hsaco_arg {
    std::string   name;         ///< Argument name (explicit args only; empty for hidden args).
    std::string   value_kind;   ///< AMDGPU value kind, e.g. "global_buffer", "hidden_block_count_x".
    std::uint32_t offset = 0;   ///< Byte offset of the argument within the kernarg segment.
    std::uint32_t size   = 0;   ///< Argument size in bytes.
};

/**
 * @brief A single kernel's metadata parsed from a code object.
 */
struct ggml_hsaco_kernel {
    std::string                 symbol; ///< Kernel descriptor symbol as emitted (e.g. "name.kd").
    std::vector<ggml_hsaco_arg> args;   ///< Argument layout (explicit + hidden).
};

/**
 * @brief Parses every kernel in an in-memory AMDGPU code object via comgr.
 *
 * A hipcc fatbin typically contains many kernels (template instantiations); a
 * Triton @c .hsaco typically contains one. Both are handled uniformly.
 *
 * @param[in]  code_object pointer to the code-object (ELF) bytes
 * @param[in]  size        size of the code object in bytes
 * @param[out] out         kernels keyed by mangled name (the ".kd" suffix is stripped from the
 *                         key so it matches @c __hipRegisterFunction's deviceName); each value's
 *                         @c symbol retains the raw ".kd" symbol for @c hsa_executable_get_symbol
 * @return @c true if at least one kernel was parsed, @c false otherwise
 */
bool ggml_hsaco_parse_kernels(const void * code_object, std::size_t size,
                              std::unordered_map<std::string, ggml_hsaco_kernel> & out);
