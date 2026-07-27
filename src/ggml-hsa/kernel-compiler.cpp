// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-compiler.hpp"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "ggml-hsa/common.hpp"
#include "ggml-impl.h"

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

/// @brief If @c true, JIT compilation will print verbose output.
const bool verbose_compilation = [] {
    const char * env = std::getenv("GGML_HSA_JIT_VERBOSE");
    return env != nullptr && ggml_hsa_string_to_bool(env);
}();

/// @brief Ordered backend names to try during JIT compilation.
///
/// Set via @c GGML_HSA_JIT_COMPILER_ORDER as a comma-separated list (e.g. "iron,triton").
/// Matching is case-insensitive; a kernel whose backend is not listed is dropped. When unset
/// (or empty), the dispatch function's order is used unchanged.
const std::vector<std::string> compiler_order = [] {
    const char * env = std::getenv("GGML_HSA_JIT_COMPILER_ORDER");
    const std::string list = (env != nullptr) ? env : "";

    std::vector<std::string> compilers;
    std::stringstream ss(list);
    std::string name;
    while (std::getline(ss, name, ',')) {
        // trim surrounding whitespace
        const auto begin = name.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            continue;
        }
        const auto end = name.find_last_not_of(" \t");
        compilers.push_back(name.substr(begin, end - begin + 1));
    }
    return compilers;
}();

/// @brief Path to the shared library directory.
const std::filesystem::path ggml_hsa_library_dir = [] {
    // retrieve the shared library path
    Dl_info info;
    if (dladdr(static_cast<const void *>(&ggml_hsa_library_dir), &info) == 0) {
        GGML_ABORT("Could not retrieve library directory\n");
    }
    return std::filesystem::path{info.dli_fname}.parent_path();
}();

/// @brief Path to AIE kernels.
const fs::path kernel_path = ggml_hsa_library_dir / "kernels";

/// @brief Owns the embedded Python interpreter and the module handles used for JIT compilation.
///
/// The interpreter is initialized once and its handles resolved once, avoiding a repeated module
/// import and attribute lookup on every compile. The interpreter is intentionally never finalized
/// and the instance is leaked to process teardown (see @ref python_compiler_instance): the JIT
/// compile path imports native C-extension modules (numpy, aie.iron, torch/triton) that cannot be
/// safely unloaded by Py_Finalize(), and the cached handles must likewise not be released during
/// static destruction (see pybind11 embedding docs).
///
/// @c initialize_interpreter() leaves the calling thread holding the GIL. The constructor releases
/// it once setup is done and keeps the release alive for the process lifetime, so the GIL is not
/// held by whichever thread happened to initialize first. Each call to @ref ggml_hsa_compile_kernel
/// re-acquires the GIL, making compilation safe to invoke from any thread.
struct python_compiler {
    py::object create_tensor_desc;                     ///< tensor_desc.ggml_tensor_to_tensordesc
    py::object compiler_config;                        ///< build.CompilerConfig
    py::object compile_op;                             ///< build.ggml_compile_op
    std::optional<py::gil_scoped_release> gil_release; ///< holds the GIL released for the process

    python_compiler() {
        py::initialize_interpreter();
        {
            // resolve the module handles in an inner scope so the temporary py::module_ objects
            // while the GIL is still held
            auto sys = py::module_::import("sys");
            sys.attr("path").attr("append")(kernel_path.string());

            auto tensor_desc_mod = py::module_::import("tensor_desc");
            create_tensor_desc = tensor_desc_mod.attr("ggml_tensor_to_tensordesc");

            auto build_mod = py::module_::import("build");
            compiler_config = build_mod.attr("CompilerConfig");
            compile_op = build_mod.attr("ggml_compile_op");
        }

        // release the GIL acquired by initialize_interpreter(); reacquired per compile call
        gil_release.emplace();
    }
};

/// @brief Returns the JIT compiler instance, or @c nullptr if interpreter initialization failed.
///
/// The interpreter and its module handles are created on first use rather than at library load, so
/// a process that never JIT-compiles a kernel (e.g. all kernels already cached) never pays the
/// native-module import cost (numpy, aie.iron, torch/triton). The instance is intentionally leaked
/// (never deleted) so it survives to process teardown without running Python teardown during static
/// destruction.
python_compiler * get_python_compiler() {
    static python_compiler * const instance = []() -> python_compiler * {
        try {
            return new python_compiler{};
        } catch (const std::exception & ex) {
            GGML_HSA_LOG_ERROR("Failed to initialize Python interpreter: %s\n", ex.what());
            return nullptr;
        }
    }();
    return instance;
}

/**
 * @brief Creates a @p py::tuple from the tensor shape.
 */
py::tuple ggml_hsa_tensor_ne_as_pytuple(const ggml_tensor & tensor) {
    auto ne = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i] = py::int_(tensor.ne[i]);
    }
    return ne;
}

/**
 * @brief Creates a @p py::tuple from the tensor strides.
 */
py::tuple ggml_hsa_tensor_nb_as_pytuple(const ggml_tensor & tensor) {
    auto nb = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        nb[i] = py::int_(tensor.nb[i]);
    }
    return nb;
}

} // namespace

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor & tensor,
                                    std::optional<std::string> op_name,
                                    const std::string & kernel_name,
                                    const std::filesystem::path & output_path) {
    using namespace py::literals;

    auto * const compiler = get_python_compiler();
    if (compiler == nullptr) {
        return GGML_STATUS_FAILED;
    }

    const std::string op_name_to_use =
        op_name.has_value() ? std::move(op_name.value()) : ggml_op_desc(&tensor);

    const auto output_directory = output_path / dev_info.name;

    try {
        // acquire the GIL for the duration of this call
        py::gil_scoped_acquire gil;

        // convert a GGML tensor to input and output TensorDesc objects
        const auto & create_tensor_desc = compiler->create_tensor_desc;
        const auto src_tensor_count = ggml_hsa_nsrcs(tensor);
        auto input_tensors = py::list(src_tensor_count);
        for (auto i = 0; i < src_tensor_count; ++i) {
            const auto src_tensor = tensor.src[i];
            if (src_tensor == nullptr) {
                // handle "holes", e.g., for SOFT_MAX src[0] = input, src[1] = mask (can be
                // nullptr), src[2] = sinks (can be non-null)
                input_tensors[i] = py::none();
            } else {
                input_tensors[i] =
                    create_tensor_desc("dtype"_a = ggml_type_name(src_tensor->type),
                                       "ne"_a = ggml_hsa_tensor_ne_as_pytuple(*src_tensor),
                                       "nb"_a = ggml_hsa_tensor_nb_as_pytuple(*src_tensor),
                                       "contiguous"_a = ggml_is_contiguous(src_tensor));
            }
        }
        auto output_tensor = create_tensor_desc("dtype"_a = ggml_type_name(tensor.type),
                                                "ne"_a = ggml_hsa_tensor_ne_as_pytuple(tensor),
                                                "nb"_a = ggml_hsa_tensor_nb_as_pytuple(tensor),
                                                "contiguous"_a = ggml_is_contiguous(&tensor));

        auto op_params = py::bytearray(reinterpret_cast<const char *>(tensor.op_params),
                                       sizeof(tensor.op_params));

        // compile the kernel
        auto config = compiler->compiler_config("output_directory"_a = output_directory.string(),
                                                "compilers"_a = py::tuple(py::cast(compiler_order)),
                                                "verbose"_a = verbose_compilation);
        compiler->compile_op("op_name"_a = op_name_to_use, "arch"_a = dev_info.name,
                             "input_tensors"_a = std::move(input_tensors),
                             "output_tensor"_a = std::move(output_tensor),
                             "op_params"_a = std::move(op_params), "exported_name"_a = kernel_name,
                             "config"_a = std::move(config));
    } catch (const py::error_already_set & ex) {
        GGML_HSA_LOG_ERROR("%s: failed to compile kernel %s for tensor \"%s\" (%s): %s", __func__,
                           kernel_name.c_str(), tensor.name, op_name_to_use.c_str(), ex.what());
        return GGML_STATUS_FAILED;
    }

    GGML_HSA_LOG_INFO("%s: generated kernel %s in %s for tensor \"%s\" (%s)", __func__,
                      kernel_name.c_str(), output_directory.c_str(), tensor.name,
                      op_name_to_use.c_str());

    return GGML_STATUS_SUCCESS;
}
