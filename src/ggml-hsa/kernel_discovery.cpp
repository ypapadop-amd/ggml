#include "kernel_discovery.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include "ggml-impl.h"

namespace fs = std::filesystem;

// PDI file suffix.
static const std::string_view pdi_file_suffix = ".pdi";

// Binary instructions file suffix.
static const std::string_view inst_file_suffix = "_insts.bin";

// System (i.e., precompiled and installed) kernel directory.
static const fs::path system_kernel_dir = [] {
    // retrieve the kernel directory as a relative path from this shared library
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&ggml_hsa_kernel_exists), &info) == 0) {
        GGML_ABORT("Could not retrieve kernel base directory\n");
    }
    auto library_path = fs::path{info.dli_fname}.parent_path() / "iron-kernels";
    if (!fs::is_directory(library_path)) {
        GGML_ABORT("Directory %s is not a valid path.\n", library_path.c_str());
    }
    return library_path;
}();

// User (i.e., out-of-tree and JIT compiled) kernel base path.
static const fs::path user_kernel_dir = [] {
    // user compiled and JIT kernels are stored in XDG_CACHE_HOME if defined or $HOME/.cache if not
    fs::path dir;
    if (const char * cache_dir = std::getenv("XDG_CACHE_HOME"); cache_dir != nullptr) {
        dir = fs::path(cache_dir) / "ggml";
    } else {
        const char * home_dir = std::getenv("HOME");
        if (home_dir == nullptr) {
            home_dir = "/tmp";
        }
        dir = fs::path(home_dir) / ".cache/ggml";
    }
    GGML_LOG_INFO("ggml_hsa_backend: User kernels in %s\n", dir.c_str());
    return dir;
}();

/**
 * @brief Creates a kernel name for the operation in tensor @p tensor.
 */
static ggml_status ggml_hsa_create_kernel_name(const ggml_tensor * tensor,
                                               std::string & kernel_name) {
    if ((tensor->op < GGML_OP_NONE) || (tensor->op >= GGML_OP_COUNT)) {
        GGML_LOG_ERROR("%s: Tensor operation index out of bounds (%d >= GGML_OP_COUNT)\n", __func__,
                       tensor->op);
        return GGML_STATUS_FAILED;
    }

    std::ostringstream oss;
    std::string_view op_name = ggml_op_name(tensor->op);
    std::transform(op_name.begin(), op_name.end(), std::ostreambuf_iterator(oss),
                   [&](char c) { return std::tolower(c); });
    oss << '-';
    ggml_hsa_output_tensor(tensor, oss);
    oss << '-';
    ggml_hsa_output_tensor(tensor->src[0], oss);
    for (int i = 1; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] == nullptr) {
            break;
        }
        oss << '-';
        ggml_hsa_output_tensor(tensor->src[i], oss);
    }
    kernel_name = oss.str();

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Returns if @p p is a file.
 */
static bool ggml_hsa_is_file(const fs::path & p) {
    return fs::is_regular_file(p) || fs::is_symlink(p);
}

/**
 * @brief Searches all directories for the kernel.
 */
static bool ggml_hsa_find_kernel(const std::string & device_name,
                                 const std::string & kernel_name,
                                 fs::path & pdi_path,
                                 fs::path & insts_path) {
    const auto partial_path = fs::path(device_name).append(kernel_name);
    const auto partial_pdi_path = fs::path(partial_path).concat(pdi_file_suffix);
    const auto partial_insts_path = fs::path(partial_path).concat(inst_file_suffix);

    auto tmp_pdi_path = system_kernel_dir / partial_pdi_path;
    auto tmp_insts_path = system_kernel_dir / partial_insts_path;
    if (ggml_hsa_is_file(tmp_pdi_path) && ggml_hsa_is_file(tmp_insts_path)) {
        pdi_path = std::move(tmp_pdi_path);
        insts_path = std::move(tmp_insts_path);
        return true;
    }

    tmp_pdi_path = user_kernel_dir / partial_pdi_path;
    tmp_insts_path = user_kernel_dir / partial_insts_path;
    if (ggml_hsa_is_file(tmp_pdi_path) && ggml_hsa_is_file(tmp_insts_path)) {
        pdi_path = std::move(tmp_pdi_path);
        insts_path = std::move(tmp_insts_path);
        return true;
    }

    return false;
}

/**
 * @brief Reads a PDI file from @p path and returns its contents and size in bytes in @p buffer.
 */
static ggml_status
ggml_hsa_load_pdi(hsa_amd_memory_pool_t pool, const fs::path & path, ggml_hsa_pdi_buffer & buffer) {
    std::ifstream is(path, std::ios::binary | std::ios::ate | std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    if (!is.seekg(0, std::ios::beg) || (size == 0)) {
        GGML_LOG_ERROR("%s: I/O error, could not get file size for %s\n", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    if (auto status =
            hsa_amd_memory_pool_allocate(pool, size, 0, reinterpret_cast<void **>(&buffer.data));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, size);
        return GGML_STATUS_FAILED;
    }

    is.read(reinterpret_cast<char *>(buffer.data), size);
    buffer.size = size;

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Reads an instruction file from @p path and returns its contents and number of instructions
 *        in @p buffer.
 */
static ggml_status ggml_hsa_load_insts(hsa_amd_memory_pool_t pool,
                                       const fs::path & path,
                                       ggml_hsa_insts_buffer & buffer) {
    std::ifstream is(path, std::ios::binary | std::ios::ate | std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    if (!is.seekg(0, std::ios::beg) || (size == 0)) {
        GGML_LOG_ERROR("%s: I/O error, could not get file size for %s\n", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    if (auto status =
            hsa_amd_memory_pool_allocate(pool, size, 0, reinterpret_cast<void **>(&buffer.data));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, size);
        return GGML_STATUS_FAILED;
    }

    if (size % sizeof(std::uint32_t) != 0) {
        GGML_LOG_ERROR("%s: File size is not a multiple of %zu bytes\n", __func__,
                       sizeof(std::uint32_t));
        return GGML_STATUS_FAILED;
    }

    is.read(reinterpret_cast<char *>(buffer.data), size);
    buffer.size = size / sizeof(std::uint32_t);

    return GGML_STATUS_SUCCESS;
}

bool ggml_hsa_kernel_exists(const ggml_hsa_device_info::device_info & dev_info,
                            const ggml_tensor * tensor) {
    // generate kernel name
    std::string kernel_name;
    if (ggml_hsa_create_kernel_name(tensor, kernel_name) != GGML_STATUS_SUCCESS) {
        return false;
    }

    // check if the kernel exists as a file
    fs::path pdi_path;
    fs::path insts_path;
    return ggml_hsa_find_kernel(dev_info.name, kernel_name, pdi_path, insts_path);
}

ggml_status ggml_hsa_find_aie_kernel(ggml_backend_hsa_context & ctx,
                                     const ggml_tensor * tensor,
                                     ggml_hsa_aie_kernel & kernel) {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[ctx.device];

    // generate kernel name
    std::string kernel_name;
    if (auto status = ggml_hsa_create_kernel_name(tensor, kernel_name);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // find kernel in already loaded kernels
    auto it = ctx.aie_kernels.find(kernel_name);
    if (it != ctx.aie_kernels.end()) {
        kernel = it->second;
        return GGML_STATUS_SUCCESS;
    }

    // kernel not found, search the kernel directories
    fs::path pdi_path;
    fs::path insts_path;
    if (!ggml_hsa_find_kernel(dev_info.name, kernel_name, pdi_path, insts_path)) {
        return GGML_STATUS_FAILED;
    }

    // load PDI and instructions
    ggml_hsa_aie_kernel tmp_kernel;
    if (auto status = ggml_hsa_load_pdi(dev_info.dev_memory.memory_pool, pdi_path, tmp_kernel.pdi);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    if (auto status =
            ggml_hsa_load_insts(dev_info.dev_memory.memory_pool, insts_path, tmp_kernel.insts);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    ctx.aie_kernels.emplace(std::move(kernel_name), tmp_kernel);
    kernel = tmp_kernel;

    return GGML_STATUS_SUCCESS;
}

void ggml_hsa_destroy_aie_kernel(ggml_backend_hsa_context & /*ctx*/, ggml_hsa_aie_kernel & kernel) {
    if (auto status = hsa_amd_memory_pool_free(kernel.pdi.data); status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_amd_memory_pool_free error (%d)\n", __func__, status);
    }
    if (auto status = hsa_amd_memory_pool_free(kernel.insts.data); status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_amd_memory_pool_free error (%d)\n", __func__, status);
    }
    kernel = {};
}
