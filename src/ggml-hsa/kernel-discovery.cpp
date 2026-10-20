// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-discovery.hpp"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string_view>

#include "ggml-impl.h"

#ifdef GGML_HSA_AIE
#include "ggml-hsa/aie-kernel.hpp"
#endif

#ifdef GGML_HSA_GPU
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <amd_comgr/amd_comgr.h>

#include "ggml-hsa/gpu-kernel.hpp"
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

/// Code object file suffix.
static constexpr std::string_view hsaco_file_suffix = ".hsaco";

/**
 * @brief Returns if the code object for a @ref ggml_hsa_kernel exists in any of the
 * directories.
 */
static bool ggml_hsa_find_kernel_file(const std::string & device_name,
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

#ifdef GGML_HSA_AIE

/**
 * @brief Creates an AIE kernel by loading its hsaco (JIT-compiling it if absent).
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to compile a kernel for (used by the JIT fallback)
 * @param[in] op_name operation name; if provided, it overrides the op name derived from @p tensor
 * @param[in] kernel_name kernel name
 * @param[out] kernel kernel for the operation
 */
static ggml_status ggml_hsa_create_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                              const ggml_tensor & tensor,
                                              std::optional<std::string> op_name,
                                              const std::string & kernel_name,
                                              std::shared_ptr<ggml_hsa_kernel> & kernel) {
    fs::path hsaco_path;

    // search for kernel file
    if (!ggml_hsa_find_kernel_file(dev_info.name, kernel_name, hsaco_path)) {
#ifdef GGML_HSA_JIT_COMPILE
        // kernel file not found, compile kernel
        if (auto status =
                ggml_hsa_compile_kernel(dev_info, tensor, op_name, kernel_name, cached_kernel_dir);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }

        // search for kernel file after compilation
        if (!ggml_hsa_find_kernel_file(dev_info.name, kernel_name, hsaco_path)) {
            return GGML_STATUS_FAILED;
        }
#else
        GGML_HSA_LOG_INFO("%s: JIT compilation is disabled, kernel cannot be compiled", __func__);
        return GGML_STATUS_FAILED;
#endif
    }

    auto aie_kernel = std::make_shared<ggml_hsa_aie_kernel>();
    if (auto status = aie_kernel->load(dev_info.agent, hsaco_path, kernel_name);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    kernel = std::move(aie_kernel);

    return GGML_STATUS_SUCCESS;
}

#endif // GGML_HSA_AIE

#ifdef GGML_HSA_GPU

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
 * @brief Reads the string value of a comgr metadata node.
 */
static bool ggml_hsa_comgr_get_string(amd_comgr_metadata_node_t node, std::string & out) {
    std::size_t sz = 0;
    if (amd_comgr_get_metadata_string(node, &sz, nullptr) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    std::string s(sz, '\0');
    if (amd_comgr_get_metadata_string(node, &sz, s.data()) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    if (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    out = std::move(s);
    return true;
}

/**
 * @brief Looks up a string-valued key in a comgr metadata map.
 */
static bool
ggml_hsa_comgr_lookup_string(amd_comgr_metadata_node_t map, const char * key, std::string & out) {
    amd_comgr_metadata_node_t node;
    if (amd_comgr_metadata_lookup(map, key, &node) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }
    bool ok = ggml_hsa_comgr_get_string(node, out);
    amd_comgr_destroy_metadata(node);
    return ok;
}

/**
 * @brief Parses the kernel symbol and argument layout from a code object's metadata.
 *
 * Reads the AMDGPU code object metadata (via comgr) for the single kernel it contains, recording
 * the kernel descriptor symbol and the value kind, byte offset, and size of each argument
 * (including the hidden arguments). Reading the symbol from the metadata makes the loader
 * independent of the producer's naming (e.g. hipcc's ggml kernel name vs Triton's function name),
 * and reading the layout avoids hardcoding the kernarg ABI.
 *
 * @param[in] blob in-memory code object bytes
 * @param[out] out_symbol kernel descriptor symbol (e.g. "add_8f32_8f32_8f32.kd" or "gemm_kernel.kd")
 * @param[out] out_args parsed argument descriptors
 */
static bool ggml_hsa_parse_kernel_metadata(const std::vector<char> & blob,
                                           std::string & out_symbol,
                                           std::vector<ggml_hsa_gpu_kernel_arg> & out_args) {
    amd_comgr_data_t data;
    if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) != AMD_COMGR_STATUS_SUCCESS) {
        return false;
    }

    bool result = false;
    bool have_root = false;
    amd_comgr_metadata_node_t root{};
    do {
        if (amd_comgr_set_data(data, blob.size(), blob.data()) != AMD_COMGR_STATUS_SUCCESS) {
            break;
        }
        if (amd_comgr_get_data_metadata(data, &root) != AMD_COMGR_STATUS_SUCCESS) {
            break;
        }
        have_root = true;

        amd_comgr_metadata_node_t kernels;
        if (amd_comgr_metadata_lookup(root, "amdhsa.kernels", &kernels) !=
            AMD_COMGR_STATUS_SUCCESS) {
            break;
        }

        std::size_t nkernels = 0;
        if (amd_comgr_get_metadata_list_size(kernels, &nkernels) != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_destroy_metadata(kernels);
            break;
        }
        // Each code object we load contains exactly one kernel; use the first.
        for (std::size_t i = 0; i < nkernels && !result; ++i) {
            amd_comgr_metadata_node_t kernel;
            if (amd_comgr_index_list_metadata(kernels, i, &kernel) != AMD_COMGR_STATUS_SUCCESS) {
                continue;
            }

            if (ggml_hsa_comgr_lookup_string(kernel, ".symbol", out_symbol)) {
                amd_comgr_metadata_node_t args;
                if (amd_comgr_metadata_lookup(kernel, ".args", &args) == AMD_COMGR_STATUS_SUCCESS) {
                    // Dispatch fills the arguments by position, so an argument that cannot be read
                    // completely rejects the code object rather than being skipped or defaulted.
                    const auto parse_u32 = [](const std::string & text, std::uint32_t & value) {
                        const char * end = text.data() + text.size();
                        const auto [last, ec] = std::from_chars(text.data(), end, value);
                        return !text.empty() && ec == std::errc{} && last == end;
                    };
                    std::size_t nargs = 0;
                    bool args_ok = amd_comgr_get_metadata_list_size(args, &nargs) ==
                                   AMD_COMGR_STATUS_SUCCESS;
                    for (std::size_t j = 0; args_ok && j < nargs; ++j) {
                        amd_comgr_metadata_node_t arg;
                        if (amd_comgr_index_list_metadata(args, j, &arg) !=
                            AMD_COMGR_STATUS_SUCCESS) {
                            args_ok = false;
                            break;
                        }
                        std::string name, kind, off, size;
                        ggml_hsa_comgr_lookup_string(arg, ".name", name); // explicit args only
                        ggml_hsa_gpu_kernel_arg info;
                        args_ok = ggml_hsa_comgr_lookup_string(arg, ".value_kind", kind) &&
                                  ggml_hsa_comgr_lookup_string(arg, ".offset", off) &&
                                  ggml_hsa_comgr_lookup_string(arg, ".size", size) &&
                                  parse_u32(off, info.offset) && parse_u32(size, info.size);
                        amd_comgr_destroy_metadata(arg);
                        if (args_ok) {
                            info.name = std::move(name);
                            info.value_kind = std::move(kind);
                            out_args.push_back(std::move(info));
                        }
                    }
                    amd_comgr_destroy_metadata(args);
                    if (!args_ok) {
                        out_args.clear();
                    }
                    result = args_ok && !out_args.empty() && !out_symbol.empty();
                }
            }
            amd_comgr_destroy_metadata(kernel);
        }
        amd_comgr_destroy_metadata(kernels);
    } while (false);

    if (have_root) {
        amd_comgr_destroy_metadata(root);
    }
    amd_comgr_release_data(data);
    return result;
}

/// @brief Result of reading a launch-metadata sidecar.
enum class ggml_hsa_sidecar_status {
    absent,    ///< No sidecar file.
    valid,     ///< Sidecar read; the launch geometry is set.
    malformed, ///< Sidecar present but unreadable, incomplete, or inconsistent.
};

/**
 * @brief Reads the launch-metadata sidecar of a GPU code object.
 *
 * Triton-generated kernels bake their launch geometry (workgroup size, number of workgroups) at
 * compile time; it is emitted alongside the @c .hsaco as @c <name>.hsaco.json. Hand-written HIP
 * kernels have no sidecar and derive their grid from the element count at dispatch.
 *
 * @p grid_size_x and @p workgroup_size_x are set only if the sidecar is valid: both are present,
 * positive, fit in their AQL packet fields, and the grid is a whole number of workgroups.
 *
 * The optional @c group_segment_size key is the dynamic LDS, in bytes, the kernel requests at
 * launch (Triton's shared memory, e.g. for @c tl.dot), which the code object's fixed group segment
 * size does not include. @p dynamic_group_segment_size is set to it, or to 0 if it is absent.
 */
static ggml_hsa_sidecar_status ggml_hsa_read_launch_sidecar(const fs::path & hsaco_path,
                                                            std::uint32_t & grid_size_x,
                                                            std::uint32_t & workgroup_size_x,
                                                            std::uint32_t & dynamic_group_segment_size) {
    fs::path sidecar = hsaco_path;
    sidecar += ".json";
    if (!ggml_hsa_is_file(sidecar)) {
        return ggml_hsa_sidecar_status::absent;
    }
    std::ifstream is(sidecar);
    if (!is) {
        return ggml_hsa_sidecar_status::malformed;
    }
    const std::string text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    const auto read_uint = [&](const char * key, std::uint64_t & out) -> bool {
        auto pos = text.find(key);
        if (pos == std::string::npos) {
            return false;
        }
        pos = text.find(':', pos);
        if (pos == std::string::npos) {
            return false;
        }
        const char * end = text.c_str() + text.size();
        const auto skip_space = [end](const char * p) {
            while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
                ++p;
            }
            return p;
        };
        const char * first = skip_space(text.c_str() + pos + 1);
        const auto [last, ec] = std::from_chars(first, end, out);
        if (ec != std::errc{} || last == first) {
            return false;
        }
        // The value must be complete: a sidecar cut short mid-number (e.g. by an interrupted
        // write) is malformed, not a smaller value.
        const char * next = skip_space(last);
        return next < end && (*next == ',' || *next == '}');
    };
    std::uint64_t grid = 0;
    std::uint64_t workgroup = 0;
    if (!read_uint("\"grid_size_x\"", grid) || !read_uint("\"workgroup_size_x\"", workgroup) ||
        grid == 0 || workgroup == 0 || grid > UINT32_MAX || workgroup > UINT16_MAX ||
        grid % workgroup != 0) {
        return ggml_hsa_sidecar_status::malformed;
    }
    std::uint64_t dynamic_lds = 0;
    if (text.find("\"group_segment_size\"") != std::string::npos &&
        (!read_uint("\"group_segment_size\"", dynamic_lds) || dynamic_lds > UINT32_MAX)) {
        return ggml_hsa_sidecar_status::malformed;
    }
    grid_size_x = static_cast<std::uint32_t>(grid);
    workgroup_size_x = static_cast<std::uint32_t>(workgroup);
    dynamic_group_segment_size = static_cast<std::uint32_t>(dynamic_lds);
    return ggml_hsa_sidecar_status::valid;
}

/**
 * @brief Creates a GPU kernel by loading a @c .hsaco code object (JIT-compiling it if absent).
 *
 * @param[in] dev_info device information
 * @param[in] tensor tensor to compile a kernel for (used by the JIT fallback)
 * @param[in] op_name operation name; if provided, it overrides the op name derived from @p tensor
 * @param[in] kernel_name kernel name
 * @param[out] kernel kernel for the operation
 */
static ggml_status ggml_hsa_create_gpu_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                              const ggml_tensor & tensor,
                                              std::optional<std::string> op_name,
                                              const std::string & kernel_name,
                                              std::shared_ptr<ggml_hsa_kernel> & kernel) {
    const std::string symbol_base = ggml_hsa_sanitize_kernel_name(kernel_name);

    fs::path hsaco_path;
    if (!ggml_hsa_find_kernel_file(dev_info.name, symbol_base, hsaco_path)) {
#ifdef GGML_HSA_JIT_COMPILE
        // code object not found; JIT-compile it (writes <cache>/<device>/<symbol_base>.hsaco).
        if (auto status =
                ggml_hsa_compile_kernel(dev_info, tensor, op_name, symbol_base, cached_kernel_dir);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
        if (!ggml_hsa_find_kernel_file(dev_info.name, symbol_base, hsaco_path)) {
            return GGML_STATUS_FAILED;
        }
#else
        GGML_UNUSED(tensor);
        GGML_UNUSED(op_name);
        GGML_HSA_LOG_INFO("%s: could not find code object for kernel %s (JIT compilation is "
                          "disabled)",
                          __func__, symbol_base.c_str());
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

    hsa_code_object_reader_t reader{};
    auto status = hsa_code_object_reader_create_from_memory(blob.data(), blob.size(), &reader);
    if (status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to read code object %s (%s)", __func__, hsaco_path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    auto gpu_kernel = std::make_shared<ggml_hsa_gpu_kernel>();

    if (status =
            hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &gpu_kernel->executable);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to create executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (status = hsa_executable_load_agent_code_object(gpu_kernel->executable, dev_info.agent,
                                                       reader, nullptr, nullptr);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to load code object (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (status = hsa_executable_freeze(gpu_kernel->executable, nullptr);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: failed to freeze executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }
    hsa_code_object_reader_destroy(reader);

    // Read the kernel symbol and argument layout from the code object metadata. This makes the
    // loader independent of the producer's naming (hipcc uses the ggml kernel name, Triton uses
    // the @triton.jit function name).
    std::string symbol_name;
    if (!ggml_hsa_parse_kernel_metadata(blob, symbol_name, gpu_kernel->args)) {
        GGML_HSA_LOG_ERROR("%s: failed to parse metadata for %s", __func__, symbol_base.c_str());
        return GGML_STATUS_FAILED;
    }

    hsa_executable_symbol_t symbol{};
    if (status = hsa_executable_get_symbol(gpu_kernel->executable, nullptr, symbol_name.c_str(),
                                           dev_info.agent, 0, &symbol);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to find symbol %s (%s)", __func__, symbol_name.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    struct symbol_query {
        hsa_executable_symbol_info_t info;
        void * dst;
    };
    const symbol_query queries[] = {
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &gpu_kernel->kernel_object},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &gpu_kernel->private_segment_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &gpu_kernel->group_segment_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &gpu_kernel->kernarg_size},
        {HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &gpu_kernel->kernarg_align},
    };
    for (const auto & query : queries) {
        if (status = hsa_executable_symbol_get_info(symbol, query.info, query.dst);
            status != HSA_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to query symbol info (%s)", __func__,
                               ggml_hsa_get_status_string(status));
            return GGML_STATUS_FAILED;
        }
    }
    if (gpu_kernel->kernarg_align < 16) {
        gpu_kernel->kernarg_align = 16;
    }

    // Dispatch writes each argument at its metadata offset, so check once that they all lie in the
    // kernarg segment the symbol reports, and that each argument dispatch fills from a fixed-size
    // value has that size: a mismatch would otherwise be truncated or over-read at every launch.
    // The by_value arguments after the element count are copied from ggml_tensor::op_params in
    // order, so together they must fit in it.
    std::size_t by_value_idx = 0;
    std::size_t op_params_bytes = 0;
    for (const auto & arg : gpu_kernel->args) {
        if (static_cast<std::uint64_t>(arg.offset) + arg.size > gpu_kernel->kernarg_size) {
            GGML_HSA_LOG_ERROR("%s: argument at offset %u (%u bytes) of %s exceeds its %u-byte "
                               "kernarg segment",
                               __func__, arg.offset, arg.size, symbol_name.c_str(),
                               gpu_kernel->kernarg_size);
            return GGML_STATUS_FAILED;
        }
        const std::uint32_t expected_size =
            ggml_hsa_gpu_fixed_arg_size(arg.value_kind, by_value_idx);
        if (arg.value_kind == "by_value") {
            const bool from_op_params = by_value_idx > 0;
            ++by_value_idx;
            if (from_op_params) {
                op_params_bytes += arg.size;
                // The same bound the dispatch asserts.
                if (op_params_bytes > sizeof(ggml_tensor::op_params)) {
                    GGML_HSA_LOG_ERROR("%s: the scalar arguments of %s after the element count "
                                       "take %zu bytes, more than the %zu-byte op_params they are "
                                       "copied from",
                                       __func__, symbol_name.c_str(), op_params_bytes,
                                       sizeof(ggml_tensor::op_params));
                    return GGML_STATUS_FAILED;
                }
            }
        }
        if (expected_size != 0 && arg.size != expected_size) {
            GGML_HSA_LOG_ERROR("%s: %s argument at offset %u of %s is %u bytes, expected %u",
                               __func__, arg.value_kind.c_str(), arg.offset, symbol_name.c_str(),
                               arg.size, expected_size);
            return GGML_STATUS_FAILED;
        }
    }

    // Launch geometry. HIP kernels carry hidden arguments and derive their grid from the element
    // count; Triton kernels have none and bake their geometry in at compile time, so they cannot
    // be launched without a valid sidecar.
    const bool has_hidden_args =
        std::any_of(gpu_kernel->args.begin(), gpu_kernel->args.end(),
                    [](const auto & arg) { return arg.value_kind.rfind("hidden_", 0) == 0; });
    switch (ggml_hsa_read_launch_sidecar(hsaco_path, gpu_kernel->grid_size_x,
                                         gpu_kernel->workgroup_size_x,
                                         gpu_kernel->dynamic_group_segment_size)) {
        case ggml_hsa_sidecar_status::valid:
            if (gpu_kernel->dynamic_group_segment_size >
                UINT32_MAX - gpu_kernel->group_segment_size) {
                GGML_HSA_LOG_ERROR("%s: %s needs more LDS than an AQL packet can describe",
                                   __func__, hsaco_path.c_str());
                return GGML_STATUS_FAILED;
            }
            break;
        case ggml_hsa_sidecar_status::absent:
            if (!has_hidden_args) {
                GGML_HSA_LOG_ERROR("%s: %s has no hidden arguments, so it needs a launch "
                                   "sidecar, but %s.json does not exist",
                                   __func__, hsaco_path.c_str(), hsaco_path.c_str());
                return GGML_STATUS_FAILED;
            }
            break;
        case ggml_hsa_sidecar_status::malformed:
            GGML_HSA_LOG_ERROR("%s: malformed launch sidecar %s.json", __func__,
                               hsaco_path.c_str());
            return GGML_STATUS_FAILED;
    }

    kernel = std::move(gpu_kernel);

    return GGML_STATUS_SUCCESS;
}

#endif // GGML_HSA_GPU

ggml_status ggml_hsa_create_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                   const ggml_tensor & tensor,
                                   std::optional<std::string> op_name,
                                   const std::string & kernel_name,
                                   std::shared_ptr<ggml_hsa_kernel> & kernel) {
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_AIE:
#ifdef GGML_HSA_AIE
            return ggml_hsa_create_aie_kernel(dev_info, tensor, op_name, kernel_name, kernel);
#else
            break;
#endif
        case HSA_DEVICE_TYPE_GPU:
#ifdef GGML_HSA_GPU
            return ggml_hsa_create_gpu_kernel(dev_info, tensor, op_name, kernel_name, kernel);
#else
            break;
#endif
        default:
            break;
    }

    // device type not supported, or not enabled in this build
    GGML_HSA_LOG_ERROR("%s: unsupported device %s", __func__, dev_info.name.c_str());
    return GGML_STATUS_FAILED;
}
