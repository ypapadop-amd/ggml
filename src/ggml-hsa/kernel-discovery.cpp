// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-discovery.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

/// Code object file suffix.
static constexpr std::string_view hsaco_file_suffix = ".hsaco";

/**
 * @brief Returns if @p p is a file.
 */
static bool ggml_hsa_is_file(const fs::path & p) {
    return fs::is_regular_file(p) || fs::is_symlink(p);
}

/**
 * @brief Returns if the code object for a @ref ggml_hsa_aie_kernel exists in any of the
 * directories.
 */
static bool ggml_hsa_find_aie_kernel_file(const std::string & device_name,
                                          const std::string & kernel_name,
                                          fs::path & hsaco_path) {
    const auto partial_hsaco_path =
        fs::path(device_name).append(kernel_name).concat(hsaco_file_suffix);

    if (!kernel_dir.empty()) {
        // find kernel in pregenerated kernel directory
        auto tmp_hsaco_path = kernel_dir / partial_hsaco_path;
        if (ggml_hsa_is_file(tmp_hsaco_path)) {
            hsaco_path = std::move(tmp_hsaco_path);
            return true;
        }
    }

    // find kernel in cached kernel directory
    auto tmp_hsaco_path = cached_kernel_dir / partial_hsaco_path;
    if (ggml_hsa_is_file(tmp_hsaco_path)) {
        hsaco_path = std::move(tmp_hsaco_path);
        return true;
    }

    // kernel not found
    return false;
}

/**
 * @brief Creates the kernel for the tensor's operation.
 *
 * This function will try the following until one succeeds in order of priority:
 *   -# load the kernel from a precompiled kernel directory,
 *   -# load the kernel from a cached kernel directory,
 *   -# compile the kernel, store it to the cached kernel directory, and load it.
 * If none of the above succeeds, an error message will be returned.
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to find the kernel for
 * @param[in] op_name operation name; if provided, it overrides the default op name derived from the
 * tensor's operation type
 * @param[in] kernel_name kernel name
 * @param[out] kernel kernel for the operation of @p tensor
 */
static ggml_status ggml_hsa_create_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                              const ggml_tensor & tensor,
                                              std::optional<std::string> op_name,
                                              const std::string & kernel_name,
                                              std::shared_ptr<ggml_hsa_kernel> & kernel) {
    fs::path hsaco_path;

    // search for kernel file
    if (!ggml_hsa_find_aie_kernel_file(dev_info.name, kernel_name, hsaco_path)) {
#ifdef GGML_HSA_JIT_COMPILE
        // kernel file not found, compile kernel
        if (auto status =
                ggml_hsa_compile_kernel(dev_info, tensor, op_name, kernel_name, cached_kernel_dir);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }

        // search for kernel file after compilation
        if (!ggml_hsa_find_aie_kernel_file(dev_info.name, kernel_name, hsaco_path)) {
            return GGML_STATUS_FAILED;
        }
#else
        GGML_HSA_LOG_INFO("%s: JIT compilation is disabled, kernel cannot be compiled", __func__);
        return GGML_STATUS_FAILED;
#endif
    }

    std::shared_ptr<ggml_hsa_aie_kernel> aie_kernel;
    if (auto status =
            ggml_hsa_aie_kernel::load(dev_info.agent, hsaco_path, kernel_name, aie_kernel);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    kernel = std::move(aie_kernel);

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const ggml_tensor & tensor,
                                   std::optional<std::string> op_name,
                                   const std::string & kernel_name,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel) {
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_AIE:
            return ggml_hsa_create_aie_kernel(dev_info, tensor, op_name, kernel_name, kernel);

        // unsupported device types
        default:
            GGML_HSA_LOG_ERROR("%s: unsupported device %s", __func__, dev_info.name.c_str());
            return GGML_STATUS_FAILED;
    }
}
