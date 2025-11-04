// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-discovery.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "ggml-hsa/aie-kernel.hpp"
#include "ggml-impl.h"
#ifdef GGML_HSA_JIT_COMPILE
#include "ggml-hsa/kernel-compiler.hpp"
#endif

namespace fs = std::filesystem;

/**
 * @brief Returns the precompiled kernel directory.
 */
static fs::path ggml_hsa_precompiled_kernel_dir() {
    if (const char * kernel_dir = std::getenv("GGML_HSA_KERNEL_DIR"); kernel_dir != nullptr) {
        auto dir = fs::path(kernel_dir);
        if (!fs::is_directory(dir)) {
            GGML_ABORT("%s: GGML_HSA_KERNEL_DIR (%s) is not a valid directory.\n", __func__,
                       dir.c_str());
        }
        return dir;
    }
    GGML_HSA_LOG_INFO("%s: no pregenerated kernel directory defined.", __func__);
    return fs::path{};
}

/// Precompiled kernel directory.
static const fs::path kernel_dir = ggml_hsa_precompiled_kernel_dir();

/**
 * @brief Returns the cached kernel directory and clears it if requested.
 *
 * Cached kernels are stored in the following directories:
 * 1. GGML_HSA_KERNEL_CACHE_DIR if defined, or
 * 2. $XDG_CACHE_HOME/ggml if XDG_CACHE_HOME is defined, or,
 * 3. $HOME/.cache/ggml if HOME is defined, or
 * 4. /tmp/ggml/ggml-hsa otherwise.
 */
static fs::path ggml_hsa_cached_kernel_dir() {
    fs::path cache_dir;
    if (const char * base_dir = std::getenv("GGML_HSA_KERNEL_CACHE_DIR"); base_dir != nullptr) {
        cache_dir = fs::path(base_dir);
    } else if (const char * base_dir = std::getenv("XDG_CACHE_HOME"); base_dir != nullptr) {
        cache_dir = fs::path(base_dir) / "ggml";
    } else if (const char * base_dir = std::getenv("HOME"); base_dir != nullptr) {
        cache_dir = fs::path(base_dir) / ".cache/ggml";
    } else {
        cache_dir = fs::path("/tmp/ggml/ggml-hsa");
    }
    GGML_HSA_LOG_INFO("%s: cached kernels in %s", __func__, cache_dir.c_str());

    if (const char * clear_cache = std::getenv("GGML_HSA_KERNEL_CACHE_CLEAR");
        clear_cache != nullptr && ggml_hsa_string_to_bool(clear_cache)) {
        GGML_HSA_LOG_INFO("%s: clearing kernel cache in %s", __func__, cache_dir.c_str());
        fs::remove_all(cache_dir);
    }

    return cache_dir;
}

/// Cached (i.e., JIT compiled) kernel directory.
static const fs::path cached_kernel_dir = ggml_hsa_cached_kernel_dir();

/// PDI file suffix.
static constexpr std::string_view pdi_file_suffix = ".pdi";

/// Binary instructions file suffix.
static constexpr std::string_view inst_file_suffix = "_insts.bin";

/**
 * @brief Returns if @p p is a file.
 */
static bool ggml_hsa_is_file(const fs::path & p) {
    return fs::is_regular_file(p) || fs::is_symlink(p);
}

/**
 * @brief Returns if the files for a @ref ggml_hsa_aie_kernel exists in any of the directories.
 */
static bool ggml_hsa_find_aie_kernel_files(const std::string & device_name,
                                           const std::string & kernel_name,
                                           fs::path & pdi_path,
                                           fs::path & insts_path) {
    const auto partial_path = fs::path(device_name).append(kernel_name);
    const auto partial_pdi_path = fs::path(partial_path).concat(pdi_file_suffix);
    const auto partial_insts_path = fs::path(partial_path).concat(inst_file_suffix);

    if (!kernel_dir.empty()) {
        // find kernel in pregenerated kernel directory
        auto tmp_pdi_path = kernel_dir / partial_pdi_path;
        auto tmp_insts_path = kernel_dir / partial_insts_path;
        if (ggml_hsa_is_file(tmp_pdi_path) && ggml_hsa_is_file(tmp_insts_path)) {
            pdi_path = std::move(tmp_pdi_path);
            insts_path = std::move(tmp_insts_path);
            return true;
        }
    }

    // find kernel in cached kernel directory
    auto tmp_pdi_path = cached_kernel_dir / partial_pdi_path;
    auto tmp_insts_path = cached_kernel_dir / partial_insts_path;
    if (ggml_hsa_is_file(tmp_pdi_path) && ggml_hsa_is_file(tmp_insts_path)) {
        pdi_path = std::move(tmp_pdi_path);
        insts_path = std::move(tmp_insts_path);
        return true;
    }

    // kernel not found
    return false;
}

/**
 * @brief Reads a PDI file from @p path and returns its contents and size in bytes in @p buffer.
 */
static ggml_status
ggml_hsa_load_pdi(hsa_amd_memory_pool_t pool, const fs::path & path, ggml_hsa_pdi_buffer & buffer) {
    std::ifstream is(path, std::ios::binary | std::ios::ate | std::ios::in);
    if (is.fail()) {
        GGML_HSA_LOG_ERROR("%s: could not open file %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    if (!is.seekg(0, std::ios::beg) || (size == 0)) {
        GGML_HSA_LOG_ERROR("%s: could not get file size for %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    void * ptr = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate %zu bytes (%s)", __func__, size,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }

    buffer = ggml_hsa_pdi_buffer{reinterpret_cast<std::uint64_t *>(ptr)};

    is.read(reinterpret_cast<char *>(buffer.data()), size);

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
        GGML_HSA_LOG_ERROR("%s: could not open file %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    if (!is.seekg(0, std::ios::beg) || (size == 0)) {
        GGML_HSA_LOG_ERROR("%s: could not get file size for %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    if (size % sizeof(std::uint32_t) != 0) {
        GGML_HSA_LOG_ERROR("%s: file size %zu bytes is not a multiple of %zu bytes", __func__, size,
                           sizeof(std::uint32_t));
        return GGML_STATUS_FAILED;
    }

    void * ptr = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate %zu bytes (%s)", __func__, size,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }

    buffer = ggml_hsa_insts_buffer{reinterpret_cast<std::uint32_t *>(ptr),
                                   (size / sizeof(std::uint32_t))};

    is.read(reinterpret_cast<char *>(buffer.data()), size);

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_hsa_create_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                              const std::string & kernel_name,
                                              const ggml_tensor & tensor,
                                              std::shared_ptr<ggml_hsa_kernel> & kernel) {
    fs::path pdi_path;
    fs::path insts_path;

    // search for kernel files
    if (!ggml_hsa_find_aie_kernel_files(dev_info.name, kernel_name, pdi_path, insts_path)) {
#ifdef GGML_HSA_JIT_COMPILE
        // kernel files not found, compile kernel
        if (auto status =
                ggml_hsa_compile_aie_kernel(dev_info, tensor, kernel_name, cached_kernel_dir);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }

        // search for kernel files after compilation
        if (!ggml_hsa_find_aie_kernel_files(dev_info.name, kernel_name, pdi_path, insts_path)) {
            return GGML_STATUS_FAILED;
        }
#else
        GGML_HSA_LOG_INFO("%s: JIT compilation is disabled, kernel cannot be compiled", __func__);
        return GGML_STATUS_FAILED;
#endif
    }

    auto aie_kernel = std::make_shared<ggml_hsa_aie_kernel>();

    // load PDI and instructions
    if (auto status = ggml_hsa_load_pdi(dev_info.dev_memory.memory_pool, pdi_path, aie_kernel->pdi);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    if (auto status =
            ggml_hsa_load_insts(dev_info.dev_memory.memory_pool, insts_path, aie_kernel->insts);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    kernel = std::move(aie_kernel);

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const std::string & kernel_name,
                                   const ggml_tensor & tensor,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel) {
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_AIE:
            return ggml_hsa_create_aie_kernel(dev_info, kernel_name, tensor, kernel);

        // unsupported device types
        default:
            GGML_HSA_LOG_ERROR("%s: unsupported device %s", __func__, dev_info.name.c_str());
            return GGML_STATUS_FAILED;
    }
}
