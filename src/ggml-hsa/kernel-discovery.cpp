// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-discovery.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "ggml-impl.h"

#ifdef GGML_HSA_AIE
#include "ggml-hsa/aie-kernel.hpp"
#endif

#ifdef GGML_HSA_GPU
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-hsa/gpu-kernel.hpp"
#include "ggml-hsa/hsaco-metadata.hpp"
#endif

// The kernel compiler bridge (Python JIT) is shared by all device backends.
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

/**
 * @brief Returns if @p p is a file.
 */
static bool ggml_hsa_is_file(const fs::path & p) {
    return fs::is_regular_file(p) || fs::is_symlink(p);
}

#ifdef GGML_HSA_AIE

/// PDI file suffix.
static constexpr std::string_view pdi_file_suffix = ".pdi";

/// Binary instructions file suffix.
static constexpr std::string_view inst_file_suffix = "_insts.bin";

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
 * @brief Reads a binary file from @p path and returns its contents in @p buffer.
 */
static ggml_status ggml_hsa_load_file(hsa_amd_memory_pool_t pool,
                                      const fs::path & path,
                                      ggml_hsa_aie_buffer & buffer) {
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (is.fail()) {
        GGML_HSA_LOG_ERROR("%s: could not open file %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::streamoff file_size = is.tellg();
    if ((file_size <= 0) || !is.seekg(0, std::ios::beg)) {
        GGML_HSA_LOG_ERROR("%s: could not get file size for %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    const auto size = static_cast<std::size_t>(file_size);

    void * ptr = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(pool, size, 0, &ptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate %zu bytes (%s)", __func__, size,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }

    buffer = ggml_hsa_aie_buffer{static_cast<std::byte *>(ptr), size};
    is.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));
    if (!is || is.gcount() != static_cast<std::streamsize>(size)) {
        GGML_HSA_LOG_ERROR("%s: failed to read %zu bytes from %s", __func__, size, path.c_str());
        buffer = ggml_hsa_aie_buffer{};
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
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
 * @param[in] kernel_name kernel name
 * @param[in] tensor tensor to find the kernel for
 * @param[out] kernel kernel for the operation of @p tensor
 */
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
                ggml_hsa_compile_kernel(dev_info, tensor, kernel_name, cached_kernel_dir);
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
    if (auto status =
            ggml_hsa_load_file(dev_info.dev_memory.memory_pool, pdi_path, aie_kernel->pdi);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    if (auto status =
            ggml_hsa_load_file(dev_info.dev_memory.memory_pool, insts_path, aie_kernel->insts);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    kernel = std::move(aie_kernel);

    return GGML_STATUS_SUCCESS;
}

#endif // GGML_HSA_AIE

#ifdef GGML_HSA_GPU

/// GPU code object file suffix.
static constexpr std::string_view hsaco_file_suffix = ".hsaco";

/**
 * @brief Sanitizes a kernel name into a valid C identifier for use as a kernel symbol.
 *
 * The generic kernel name may contain characters (e.g. '-') that are not valid in a C
 * identifier. The HIP-compiled kernels export their entry point using the sanitized name, so
 * both the @c .hsaco file name and the kernel descriptor symbol are derived from it.
 */
static std::string ggml_hsa_sanitize_kernel_name(const std::string & kernel_name) {
    std::string sanitized = kernel_name;
    for (char & c : sanitized) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
            c = '_';
        }
    }
    return sanitized;
}

/**
 * @brief Returns if the @c .hsaco file for a @ref ggml_hsa_gpu_kernel exists in any directory.
 */
static bool ggml_hsa_find_gpu_kernel_files(const std::string & device_name,
                                           const std::string & symbol_name,
                                           fs::path & hsaco_path) {
    const auto partial_path =
        fs::path(device_name).append(symbol_name).concat(hsaco_file_suffix);

    if (!kernel_dir.empty()) {
        // find kernel in pregenerated kernel directory
        auto tmp = kernel_dir / partial_path;
        if (ggml_hsa_is_file(tmp)) {
            hsaco_path = std::move(tmp);
            return true;
        }
    }

    // find kernel in cached kernel directory
    auto tmp = cached_kernel_dir / partial_path;
    if (ggml_hsa_is_file(tmp)) {
        hsaco_path = std::move(tmp);
        return true;
    }

    return false;
}

/**
 * @brief Parses the kernel symbol and argument layout from a single-kernel code object.
 *
 * Thin wrapper over the shared multi-kernel parser @ref ggml_hsaco_parse_kernels (hsaco-metadata),
 * which walks the AMDGPU "amdhsa.kernels" metadata via comgr. A native (Triton/hipcc single-op)
 * @c .hsaco contains exactly one kernel, so the first parsed entry is used. Reading the symbol and
 * layout from the code object itself keeps the loader independent of the producer's naming and
 * avoids hardcoding the kernarg ABI. The same parser resolves the many kernels of a hipcc fatbin in
 * the HIP host ABI (hip-abi.cpp).
 *
 * @param[in] blob in-memory code object bytes
 * @param[out] out_symbol kernel descriptor symbol (e.g. "add_8f32_8f32_8f32.kd" or "vecadd.kd")
 * @param[out] out_args parsed argument descriptors
 */
static bool ggml_hsa_parse_kernel_metadata(const std::vector<char> & blob, std::string & out_symbol,
                                           std::vector<ggml_hsa_gpu_kernel_arg> & out_args) {
    std::unordered_map<std::string, ggml_hsaco_kernel> kernels;
    if (!ggml_hsaco_parse_kernels(blob.data(), blob.size(), kernels)) {
        return false;
    }
    // Each native code object contains exactly one kernel; use the first.
    const auto & kernel = kernels.begin()->second;
    out_symbol = kernel.symbol;
    out_args.clear();
    out_args.reserve(kernel.args.size());
    for (const auto & a : kernel.args) {
        ggml_hsa_gpu_kernel_arg info;
        info.name = a.name;
        info.value_kind = a.value_kind;
        info.offset = a.offset;
        info.size = a.size;
        out_args.push_back(std::move(info));
    }
    return !out_args.empty() && !out_symbol.empty();
}

/**
 * @brief Shared low-level loader helpers (see kernel-discovery.hpp).
 *
 * Factored out of @ref ggml_hsa_create_gpu_kernel so the HIP host ABI (hip-abi.cpp) reuses the exact
 * native loader sequence rather than duplicating it: @ref ggml_hsa_load_executable performs the
 * code-object load/freeze once per fatbin, and @ref ggml_hsa_resolve_kernel_symbol resolves each of
 * the (potentially many) kernels a fatbin contains.
 */
bool ggml_hsa_load_executable(const ggml_hsa_device_info::device_info & dev_info,
                              const void * code_object, std::size_t size,
                              hsa_executable_t & executable) {
    hsa_code_object_reader_t reader{};
    auto status = hsa_code_object_reader_create_from_memory(code_object, size, &reader);
    if (status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to read code object (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return false;
    }

    if (status = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                           HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                           &executable);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to create executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return false;
    }

    if (status = hsa_executable_load_agent_code_object(executable, dev_info.agent, reader, nullptr,
                                                       nullptr);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to load code object (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return false;
    }

    if (status = hsa_executable_freeze(executable, nullptr); status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to freeze executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return false;
    }
    hsa_code_object_reader_destroy(reader);
    return true;
}

bool ggml_hsa_resolve_kernel_symbol(const ggml_hsa_device_info::device_info & dev_info,
                                    hsa_executable_t executable, const std::string & symbol,
                                    ggml_hsa_loaded_kernel & out) {
    hsa_executable_symbol_t sym{};
    if (auto status = hsa_executable_get_symbol(executable, nullptr, symbol.c_str(), dev_info.agent,
                                                0, &sym);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to find symbol %s (%s)", __func__, symbol.c_str(),
                           ggml_hsa_get_status_string(status));
        return false;
    }

    struct symbol_query {
        hsa_executable_symbol_info_t info;
        void * dst;
    };
    const symbol_query queries[] = {
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &out.kernel_object},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &out.private_segment_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &out.group_segment_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &out.kernarg_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &out.kernarg_align},
    };
    for (const auto & query : queries) {
        if (auto status = hsa_executable_symbol_get_info(sym, query.info, query.dst);
            status != HSA_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to query symbol info (%s)", __func__,
                               ggml_hsa_get_status_string(status));
            return false;
        }
    }
    if (out.kernarg_align < 16) {
        out.kernarg_align = 16;
    }
    return true;
}

bool ggml_hsa_load_code_object(const ggml_hsa_device_info::device_info & dev_info,
                               const void * code_object, std::size_t size,
                               const std::string & symbol, ggml_hsa_loaded_kernel & out) {
    if (!ggml_hsa_load_executable(dev_info, code_object, size, out.executable)) {
        return false;
    }
    if (!ggml_hsa_resolve_kernel_symbol(dev_info, out.executable, symbol, out)) {
        hsa_executable_destroy(out.executable);
        out.executable = hsa_executable_t{};
        return false;
    }
    return true;
}

/**
 * @brief Reads the optional launch-metadata sidecar for a GPU code object.
 *
 * Triton-generated kernels bake their launch geometry (workgroup size, number of workgroups) at
 * compile time; it is emitted alongside the @c .hsaco as @c <name>.hsaco.json. Hand-written HIP
 * kernels have no sidecar and derive their grid from the element count at dispatch.
 */
static bool ggml_hsa_read_launch_sidecar(const fs::path & hsaco_path, std::uint32_t & grid_size_x,
                                         std::uint32_t & workgroup_size_x,
                                         std::uint32_t & group_segment_size) {
    fs::path sidecar = hsaco_path;
    sidecar += ".json";
    if (!ggml_hsa_is_file(sidecar)) {
        return false;
    }
    std::ifstream is(sidecar);
    const std::string text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    const auto read_uint = [&](const char * key, std::uint32_t & out) -> bool {
        auto pos = text.find(key);
        if (pos == std::string::npos) {
            return false;
        }
        pos = text.find(':', pos);
        if (pos == std::string::npos) {
            return false;
        }
        out = static_cast<std::uint32_t>(std::strtoul(text.c_str() + pos + 1, nullptr, 10));
        return true;
    };
    bool ok = read_uint("\"grid_size_x\"", grid_size_x);
    ok = read_uint("\"workgroup_size_x\"", workgroup_size_x) && ok;
    // Optional: dynamic LDS required by the kernel (Triton tl.dot etc.); absent for HIP kernels.
    read_uint("\"group_segment_size\"", group_segment_size);
    return ok;
}

/**
 * @brief Creates a GPU kernel by loading a @c .hsaco code object (JIT-compiling it if absent).
 *
 * @param[in] dev_info device information
 * @param[in] kernel_name kernel name
 * @param[in] tensor tensor to compile a kernel for (used by the JIT fallback)
 * @param[out] kernel kernel for the operation
 */
static ggml_status ggml_hsa_create_gpu_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                              const std::string & kernel_name,
                                              const ggml_tensor & tensor,
                                              std::shared_ptr<ggml_hsa_kernel> & kernel) {
    const std::string symbol_base = ggml_hsa_sanitize_kernel_name(kernel_name);

    fs::path hsaco_path;
    if (!ggml_hsa_find_gpu_kernel_files(dev_info.name, symbol_base, hsaco_path)) {
#ifdef GGML_HSA_JIT_COMPILE
        // code object not found; JIT-compile it (writes <cache>/<device>/<symbol_base>.hsaco).
        if (auto status =
                ggml_hsa_compile_kernel(dev_info, tensor, symbol_base, cached_kernel_dir);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
        if (!ggml_hsa_find_gpu_kernel_files(dev_info.name, symbol_base, hsaco_path)) {
            return GGML_STATUS_FAILED;
        }
#else
        GGML_HSA_LOG_INFO("%s: JIT compilation is disabled, kernel cannot be compiled", __func__);
        return GGML_STATUS_FAILED;
#endif
    }

    // Read the code object once; the bytes feed both the HSA loader and the comgr metadata parser.
    std::ifstream is(hsaco_path, std::ios::binary | std::ios::ate);
    if (!is) {
        GGML_HSA_LOG_ERROR("%s: could not open %s", __func__, hsaco_path.c_str());
        return GGML_STATUS_FAILED;
    }
    const std::streamsize blob_size = is.tellg();
    std::vector<char> blob(static_cast<std::size_t>(blob_size));
    if (!is.seekg(0, std::ios::beg).read(blob.data(), blob_size)) {
        GGML_HSA_LOG_ERROR("%s: failed to read %s", __func__, hsaco_path.c_str());
        return GGML_STATUS_FAILED;
    }

    // Read the kernel symbol and argument layout from the code object metadata. This makes the
    // loader independent of the producer's naming (hipcc uses the ggml kernel name, Triton uses
    // the @triton.jit function name).
    auto gpu_kernel = std::make_shared<ggml_hsa_gpu_kernel>();
    std::string symbol_name;
    if (!ggml_hsa_parse_kernel_metadata(blob, symbol_name, gpu_kernel->args)) {
        GGML_HSA_LOG_ERROR("%s: failed to parse metadata for %s", __func__, symbol_base.c_str());
        return GGML_STATUS_FAILED;
    }

    // Load the code object and resolve the kernel using the shared native loader helpers (also used
    // by the HIP host ABI for hipcc fatbins).
    ggml_hsa_loaded_kernel loaded;
    if (!ggml_hsa_load_code_object(dev_info, blob.data(), blob.size(), symbol_name, loaded)) {
        GGML_HSA_LOG_ERROR("%s: failed to load code object %s", __func__, hsaco_path.c_str());
        return GGML_STATUS_FAILED;
    }
    gpu_kernel->executable = loaded.executable;
    gpu_kernel->kernel_object = loaded.kernel_object;
    gpu_kernel->private_segment_size = loaded.private_segment_size;
    gpu_kernel->group_segment_size = loaded.group_segment_size;
    gpu_kernel->kernarg_size = loaded.kernarg_size;
    gpu_kernel->kernarg_align = loaded.kernarg_align;

    // Optional launch geometry (Triton kernels bake grid/block at compile time).
    ggml_hsa_read_launch_sidecar(hsaco_path, gpu_kernel->grid_size_x, gpu_kernel->workgroup_size_x,
                                 gpu_kernel->sidecar_group_segment_size);

    kernel = std::move(gpu_kernel);

    return GGML_STATUS_SUCCESS;
}

#endif // GGML_HSA_GPU

ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const std::string & kernel_name,
                                   const ggml_tensor & tensor,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel) {
    GGML_UNUSED(tensor);
    switch (dev_info.type) {
#ifdef GGML_HSA_AIE
        case HSA_DEVICE_TYPE_AIE:
            return ggml_hsa_create_aie_kernel(dev_info, kernel_name, tensor, kernel);
#endif
#ifdef GGML_HSA_GPU
        case HSA_DEVICE_TYPE_GPU:
            return ggml_hsa_create_gpu_kernel(dev_info, kernel_name, tensor, kernel);
#endif

        // unsupported device types
        default:
            GGML_HSA_LOG_ERROR("%s: unsupported device %s", __func__, dev_info.name.c_str());
            return GGML_STATUS_FAILED;
    }
}
