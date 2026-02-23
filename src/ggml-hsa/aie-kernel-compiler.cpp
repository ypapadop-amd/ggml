// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/aie-kernel-compiler.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <string>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#include <pybind11/embed.h>

#include "ggml-hsa/common.hpp"
#include "ggml-impl.h"

namespace fs = std::filesystem;
namespace py = pybind11;

/// @brief If @c true, JIT compilation will print verbose output.
static const bool verbose_compilation = [] {
    const char * env = std::getenv("GGML_HSA_JIT_VERBOSE");
    return env != nullptr && ggml_hsa_string_to_bool(env);
}();

/// @brief Path to the shared library directory.
static const std::filesystem::path ggml_hsa_library_dir = [] {
    // retrieve the shared library path
    Dl_info info;
    if (dladdr(static_cast<const void *>(&ggml_hsa_library_dir), &info) == 0) {
        GGML_ABORT("Could not retrieve library directory\n");
    }
    return std::filesystem::path{info.dli_fname}.parent_path();
}();

/// @brief Path to AIE kernels.
static const fs::path kernel_path = ggml_hsa_library_dir / "kernels";

/// @brief Python interpreter initialization guard.
static py::scoped_interpreter python_interpreter_guard = [] {
    py::scoped_interpreter guard;
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("append")(kernel_path.string());
    return guard;
}();

/**
 * @brief Creates a @p py::tuple from the tensor shape.
 */
static py::tuple ggml_hsa_tensor_ne_as_pytuple(const ggml_tensor & tensor) {
    auto ne = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i] = py::int_(tensor.ne[i]);
    }
    return ne;
}

/**
 * @brief Creates a @p py::tuple from the tensor strides.
 */
static py::tuple ggml_hsa_tensor_nb_as_pytuple(const ggml_tensor & tensor) {
    auto nb = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        nb[i] = py::int_(tensor.nb[i]);
    }
    return nb;
}

ggml_status ggml_hsa_compile_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                        const ggml_tensor & tensor,
                                        const std::string & op_name,
                                        const std::string & exported_name,
                                        const std::filesystem::path & output_path) {
    using namespace py::literals;

    const auto output_directory = output_path / dev_info.name;

    try {
        // convert a GGML tensor to input and output TensorDesc objects
        auto tensor_desc_mod = py::module_::import("tensor_desc");
        auto create_tensor_desc = tensor_desc_mod.attr("ggml_tensor_to_tensordesc");
        const auto src_tensor_count = ggml_hsa_nsrcs(tensor);
        auto input_tensors = py::list(src_tensor_count);
        for (auto i = 0; i < src_tensor_count; ++i) {
            const auto src_tensor = tensor.src[i];
            input_tensors[i] =
                create_tensor_desc("type"_a = ggml_type_name(src_tensor->type),
                                   "ne"_a = ggml_hsa_tensor_ne_as_pytuple(*src_tensor),
                                   "nb"_a = ggml_hsa_tensor_nb_as_pytuple(*src_tensor),
                                   "contiguous"_a = ggml_is_contiguous(src_tensor));
        }
        auto output_tensor = create_tensor_desc("type"_a = ggml_type_name(tensor.type),
                                                "ne"_a = ggml_hsa_tensor_ne_as_pytuple(tensor),
                                                "nb"_a = ggml_hsa_tensor_nb_as_pytuple(tensor),
                                                "contiguous"_a = ggml_is_contiguous(&tensor));

        auto op_params = py::bytearray(reinterpret_cast<const char *>(tensor.op_params),
                                       sizeof(tensor.op_params));

        // compile the kernel
        auto build_mod = py::module_::import("build");
        auto compile_kernel = build_mod.attr("compile_kernel");
        compile_kernel("ggml_op"_a = op_name, "arch"_a = dev_info.name,
                       "input_tensors"_a = std::move(input_tensors),
                       "output_tensor"_a = std::move(output_tensor),
                       "op_params"_a = std::move(op_params), "exported_name"_a = exported_name,
                       "output_directory"_a = output_directory.string(),
                       "verbose"_a = verbose_compilation);
    } catch (const py::error_already_set & ex) {
        GGML_HSA_LOG_INFO("%s: failed to compile kernel %s for tensor \"%s\" (%s): %s", __func__,
                          exported_name.c_str(), tensor.name, op_name.c_str(), ex.what());
        return GGML_STATUS_FAILED;
    }

    GGML_HSA_LOG_INFO("%s: generated kernel %s in %s for tensor \"%s\" (%s)", __func__,
                      exported_name.c_str(), output_directory.c_str(), tensor.name,
                      op_name.c_str());

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_compile_aie_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                        const ggml_tensor & tensor,
                                        const std::string & exported_name,
                                        const std::filesystem::path & output_path) {
    return ggml_hsa_compile_aie_kernel(dev_info, tensor, ggml_op_desc(&tensor), exported_name,
                                       output_path);
}
