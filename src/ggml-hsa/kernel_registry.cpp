#include "kernels.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <iostream>

#include "ggml-impl.h"

static const std::filesystem::path kernel_base_path = [] {
    // retrieve the kernel directory as a relative path from this shared library
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&ggml_hsa_find_aie_kernel), &info) == 0) {
        GGML_ABORT("Could not retrieve kernel base directory\n");
    }
    auto library_path = std::filesystem::path{info.dli_fname}.parent_path() / "kernels";
    if (!std::filesystem::is_directory(library_path)) {
        GGML_ABORT("Directory %s is not a valid path.\n", library_path.c_str());
    }
    return library_path;
}();
static const std::string_view pdi_file_suffix = ".pdi";
static const std::string_view inst_file_suffix = "_insts.txt";

/**
 * @brief Creates a kernel name for the operation in tensor @p tensor.
 */
static ggml_status ggml_hsa_create_kernel_name(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor, std::string & kernel_name) {
    if ((tensor->op < GGML_OP_NONE) || (tensor->op >= GGML_OP_COUNT)) {
        GGML_LOG_ERROR("%s: Tensor operation index out of bounds (%d >= GGML_OP_COUNT)\n", __func__, tensor->op);
        return GGML_STATUS_FAILED;
    }

    std::ostringstream oss;
    std::string_view op_name = ggml_op_name(tensor->op);
    std::transform(op_name.begin(), op_name.end(), std::ostreambuf_iterator(oss), [&](char c) { return std::tolower(c); });
    oss << '-' << dev_info.name;
    oss << '-' << ggml_type_name(tensor->type);
    oss << '-' << ggml_nelements(tensor->src[0]);
    kernel_name = oss.str();

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Returns if @p p is a file.
 */
static bool ggml_hsa_is_file(const std::filesystem::path & p) {
    return std::filesystem::is_regular_file(p) || std::filesystem::is_symlink(p);
}

/**
 * @brief Returns the paths for PDI and insts for the kernel of @p tensor.
 */
static ggml_status ggml_hsa_create_kernel_paths(const std::string & kernel_name, std::filesystem::path & pdi_path, std::filesystem::path & instr_path) {
    const auto partial_path = kernel_base_path / kernel_name;

    pdi_path = partial_path;
    pdi_path += pdi_file_suffix;
    if (!ggml_hsa_is_file(pdi_path)) {
        GGML_LOG_WARN("%s: No PDI file found for kernel %s in %s\n", __func__, kernel_name.c_str(), pdi_path.c_str());
        return GGML_STATUS_FAILED;
    }

    instr_path = partial_path;
    instr_path += inst_file_suffix;
    if (!ggml_hsa_is_file(instr_path)) {
        GGML_LOG_WARN("%s: No instr file found for kernel %s in %s\n", __func__, kernel_name.c_str(), instr_path.c_str());
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Reads a PDI file from @p p and returns its contents and size in bytes in @p buffer and @p buffer_size respectively.
 */
static ggml_status ggml_hsa_load_pdi(hsa_amd_memory_pool_t pool, const std::filesystem::path & p, ggml_hsa_pdi_buffer & buffer) {
    std::ifstream is(p.string(), std::ios::binary | std::ios::ate | std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, p.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    GGML_ASSERT(size > 0);
    if (!is.seekg(0, std::ios::beg)) {
        GGML_LOG_ERROR("%s: I/O error, could not get file size for %s\n", __func__, p.c_str());
        return GGML_STATUS_FAILED;
    }
    if (auto status = hsa_amd_memory_pool_allocate(pool, size, 0, reinterpret_cast<void **>(&buffer.data));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, size);
        return GGML_STATUS_FAILED;
    }

    is.read(reinterpret_cast<char *>(buffer.data), size);
    buffer.size = size;

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Reads an instruction file from @p p and returns its contents and number of instructions size in @p buffer
 *        and @p instr_count respectively.
 */
static ggml_status ggml_hsa_load_insts(hsa_amd_memory_pool_t pool, const std::filesystem::path & p, ggml_hsa_insts_buffer & buffer) {
    std::ifstream is(p.string(), std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, p.c_str());
        return GGML_STATUS_FAILED;
    }

    std::string line;
    std::vector<std::uint32_t> instr_v;
    while (std::getline(is, line)) {
        std::istringstream iss(line);
        std::uint32_t a;
        if (!(iss >> std::hex >> a)) {
            GGML_LOG_ERROR("%s: I/O error, could not read file %s\n", __func__, p.c_str());
            return GGML_STATUS_FAILED;
        }
        instr_v.push_back(a);
    }
    GGML_ASSERT(instr_v.empty() == false);

    const std::size_t required_memory_size = instr_v.size() * sizeof(std::uint32_t);
    if (auto status = hsa_amd_memory_pool_allocate(pool, required_memory_size, 0, reinterpret_cast<void **>(&buffer.data));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, required_memory_size);
        return GGML_STATUS_FAILED;
    }

    std::copy(instr_v.begin(), instr_v.end(), buffer.data);
    buffer.size = instr_v.size();

    return GGML_STATUS_SUCCESS;
}

bool ggml_hsa_kernel_exists(const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor) {
    std::string kernel_name;
    if (auto status = ggml_hsa_create_kernel_name(dev_info, tensor, kernel_name); status != GGML_STATUS_SUCCESS) {
        return status;
    }

    std::filesystem::path pdi_path;
    std::filesystem::path instr_path;
    return ggml_hsa_create_kernel_paths(kernel_name, pdi_path, instr_path) == GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_find_aie_kernel(ggml_backend_hsa_context & ctx, const ggml_tensor * tensor, ggml_hsa_aie_kernel & kernel) {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[ctx.device];

    std::string kernel_name;
    if (auto status = ggml_hsa_create_kernel_name(dev_info, tensor, kernel_name); status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // find kernel in already loaded kernels
    auto it = ctx.aie_kernels.find(kernel_name);
    if (it != ctx.aie_kernels.end()) {
        kernel = it->second;
        return GGML_STATUS_SUCCESS;
    }

    // kernel not found, locate it and load it
    std::filesystem::path pdi_path;
    std::filesystem::path instr_path;
    if (auto status = ggml_hsa_create_kernel_paths(kernel_name, pdi_path, instr_path); status != GGML_STATUS_SUCCESS) {
        return status;
    }

    ggml_hsa_aie_kernel tmp_kernel;
    if (auto status = ggml_hsa_load_pdi(dev_info.dev_memory.memory_pool, pdi_path, tmp_kernel.pdi_buffer);
        status != GGML_STATUS_SUCCESS) {
      return status;
    }

    if (auto status = ggml_hsa_load_insts(dev_info.dev_memory.memory_pool, instr_path, tmp_kernel.insts_buffer);
        status != GGML_STATUS_SUCCESS) {
      return status;
    }

    ctx.aie_kernels.emplace(std::move(kernel_name), tmp_kernel);
    kernel = tmp_kernel;

    return GGML_STATUS_SUCCESS;
}

void ggml_hsa_destroy_aie_kernel(ggml_backend_hsa_context & /*ctx*/, ggml_hsa_aie_kernel & kernel) {
    if (auto status = hsa_amd_memory_pool_free(kernel.pdi_buffer.data); status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_amd_memory_pool_free error (%d)\n", __func__, status);
    }
    if (auto status = hsa_amd_memory_pool_free(kernel.insts_buffer.data); status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_amd_memory_pool_free error (%d)\n", __func__, status);
    }
    kernel = {};
}
