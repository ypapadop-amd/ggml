// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-discovery.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include "ggml-impl.h"
#ifdef GGML_HSA_JIT_COMPILE
#include "ggml-hsa/kernel-compiler.hpp"
#endif

namespace fs = std::filesystem;

// PDI file suffix.
static const std::string_view pdi_file_suffix = ".pdi";

// Binary instructions file suffix.
static const std::string_view inst_file_suffix = "_insts.bin";

// Precompiled kernel directory.
static const fs::path kernel_dir = [] {
    if (const char * kernel_dir = std::getenv("GGML_HSA_KERNEL_DIR"); kernel_dir != nullptr) {
        auto dir = fs::path(kernel_dir);
        if (!fs::is_directory(dir)) {
            GGML_ABORT(
                "ggml_hsa_backend: value of GGML_HSA_KERNEL_DIR (%s) is not a valid directory.\n",
                dir.c_str());
        }
        return dir;
    }
    GGML_LOG_INFO("ggml_hsa_backend: no pregenerated kernel directory defined.\n");
    return fs::path{};
}();

// Cached (i.e., JIT compiled) kernel directory.
static const fs::path cached_kernel_dir = [] {
    // Cached kernels are stored in the following directories:
    // 1. GGML_HSA_KERNEL_CACHE_DIR if defined, or
    // 2. $XDG_CACHE_HOME/ggml if XDG_CACHE_HOME is defined, or
    // 3. $HOME/.cache/ggml if HOME is defined, or
    // 4. /tmp/ggml/ggml-hsa otherwise.

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
    GGML_LOG_INFO("ggml_hsa_backend: Cached kernels in %s\n", cache_dir.c_str());

    if (const char * clear_cache = std::getenv("GGML_HSA_JIT_CLEAR_CACHE");
        clear_cache != nullptr && ggml_hsa_string_to_bool(clear_cache)) {
        GGML_LOG_INFO("ggml_hsa_backend: Clearing kernel cache in %s\n", cache_dir.c_str());
        fs::remove_all(cache_dir);
    }

    return cache_dir;
}();

/**
 * @brief Returns if the tensor can be flattened.
 *
 * A tensor can be flattened if it participates in an operation that is independent of the tensor's
 * dimensions.
 */
constexpr bool ggml_hsa_tensor_can_flatten(const ggml_tensor * tensor) {
    return (tensor->op == GGML_OP_UNARY) || (tensor->op == GGML_OP_SQR) ||
           (tensor->op == GGML_OP_SQRT) || (tensor->op == GGML_OP_LOG) ||
           (tensor->op == GGML_OP_SIN) || (tensor->op == GGML_OP_COS) ||
           (tensor->op == GGML_OP_SILU_BACK) || (tensor->op == GGML_OP_LEAKY_RELU);
}

/**
 * @brief Creates a kernel name for the operation in tensor @p tensor.
 */
static ggml_status ggml_hsa_create_kernel_name(const ggml_tensor * tensor,
                                               std::string & kernel_name) {
    if ((tensor->op < GGML_OP_NONE) || (tensor->op >= GGML_OP_COUNT)) {
        GGML_LOG_ERROR("%s: Tensor \"%s\" operation index out of bounds (%d >= GGML_OP_COUNT)\n",
                       __func__, ggml_get_name(tensor), tensor->op);
        return GGML_STATUS_FAILED;
    }

    const bool flatten = ggml_hsa_tensor_can_flatten(tensor);

    std::ostringstream oss;
    std::string_view op_name = ggml_op_desc(tensor);
    std::transform(op_name.begin(), op_name.end(), std::ostreambuf_iterator(oss),
                   [&](char c) { return std::tolower(c); });
    oss << '-';
    ggml_hsa_output_tensor(tensor, oss, flatten);
    for (std::int32_t i = 0; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] == nullptr) {
            break;
        }
        oss << '-';
        ggml_hsa_output_tensor(tensor->src[i], oss, flatten);
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
        return GGML_STATUS_ALLOC_FAILED;
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
        return GGML_STATUS_ALLOC_FAILED;
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

/**
 * @brief Finds and if not found, compiles the kernel.
 */
static ggml_status
ggml_hsa_find_or_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                const ggml_tensor * tensor,
                                const std::string & kernel_name,
                                fs::path & pdi_path,
                                fs::path & insts_path) {
    // search for kernel files
    if (ggml_hsa_find_kernel(dev_info.name, kernel_name, pdi_path, insts_path)) {
        return GGML_STATUS_SUCCESS;
    }

#ifdef GGML_HSA_JIT_COMPILE
    // kernel files not found, compile kernel
    if (auto status = ggml_hsa_compile_kernel(dev_info, tensor, kernel_name, cached_kernel_dir);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // search for kernel files after compilation
    if (ggml_hsa_find_kernel(dev_info.name, kernel_name, pdi_path, insts_path)) {
        return GGML_STATUS_SUCCESS;
    }
#else
    GGML_UNUSED(tensor);
#endif

    return GGML_STATUS_ABORTED;
}

bool ggml_hsa_kernel_exists(const ggml_hsa_device_info::device_info & dev_info,
                            const ggml_tensor * tensor) {
    // generate kernel name
    std::string kernel_name;
    if (ggml_hsa_create_kernel_name(tensor, kernel_name) != GGML_STATUS_SUCCESS) {
        return false;
    }

    // check if the kernel exists as a file; it will generate the kernel if it can JIT compiled
    fs::path pdi_path;
    fs::path insts_path;
    return ggml_hsa_find_or_compile_kernel(dev_info, tensor, kernel_name, pdi_path, insts_path) ==
           GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_create_aie_kernel(ggml_backend_hsa_context & ctx,
                                       const ggml_tensor * tensor,
                                       ggml_hsa_aie_kernel & kernel) {
    // generate kernel name
    std::string kernel_name;
    if (auto status = ggml_hsa_create_kernel_name(tensor, kernel_name);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // check if kernel is blocked
    if (ctx.blocked_aie_kernels.find(kernel_name) != ctx.blocked_aie_kernels.end()) {
        // kernel is blocked from being loaded
        GGML_LOG_WARN("%s: kernel \"%s\" is blocked\n", __func__, kernel_name.c_str());
        return GGML_STATUS_ABORTED;
    }

    // find kernel in already loaded kernels
    if (auto it = ctx.aie_kernels.find(kernel_name); it != ctx.aie_kernels.end()) {
        kernel = it->second;
        return GGML_STATUS_SUCCESS;
    }

    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[ctx.device];

    // kernel not found, search the kernel directories
    fs::path pdi_path;
    fs::path insts_path;
    if (auto status =
            ggml_hsa_find_or_compile_kernel(dev_info, tensor, kernel_name, pdi_path, insts_path);
        status != GGML_STATUS_SUCCESS) {
        // kernel not found and could not be compiled; block to avoid further compilation attempts
        ctx.blocked_aie_kernels.insert(kernel_name);
        return status;
    }

    // load PDI and instructions
    ggml_hsa_aie_kernel tmp_kernel;
    if (auto status = ggml_hsa_load_pdi(dev_info.dev_memory.memory_pool, pdi_path, tmp_kernel.pdi);
        status != GGML_STATUS_SUCCESS) {
        ctx.blocked_aie_kernels.insert(kernel_name);
        return status;
    }

    if (auto status =
            ggml_hsa_load_insts(dev_info.dev_memory.memory_pool, insts_path, tmp_kernel.insts);
        status != GGML_STATUS_SUCCESS) {
        ctx.blocked_aie_kernels.insert(kernel_name);
        return status;
    }

    tmp_kernel.num_src_tensors = ggml_hsa_nsrcs(tensor);
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
