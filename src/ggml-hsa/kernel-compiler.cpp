// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel-compiler.hpp"

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
    if (dladdr(reinterpret_cast<void *>(&ggml_hsa_compile_kernel), &info) == 0) {
        GGML_ABORT("Could not retrieve library directory\n");
    }
    return std::filesystem::path{info.dli_fname}.parent_path();
}();

/// @brief Path to IRON kernel support.
static const fs::path iron_path = ggml_hsa_library_dir / "iron";

/// @brief Path to IRON kernels.
static const fs::path kernel_path = iron_path / "kernels";

/**
 * @brief Enum to control if a kernel is device specific or generic.
 */
enum class ggml_hsa_aie_kernel_type { GENERIC, DEVICE_SPECIFIC };

/// @brief Python interpreter initialization guard.
static pybind11::scoped_interpreter python_interpreter_guard = [] {
    const auto utils_path = iron_path / "utils";

    pybind11::scoped_interpreter guard;
    auto sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")(iron_path.string());
    sys.attr("path").attr("append")(utils_path.string());
    return guard;
}();

/**
 * @brief Information to drive JIT compilation for a kernel.
 *
 * Operations that cannot be JIT compiled have default constructed
 * @ref ggml_hsa_aie_jit_kernel_info objects.
 */
struct ggml_hsa_aie_jit_kernel_info {
    std::string_view name;                                            ///< Kernel name.
    ggml_hsa_aie_kernel_type type{ggml_hsa_aie_kernel_type::GENERIC}; ///< Kernel type.
    fs::path source; ///< Kernel source file relative to the kernel directory.

    ggml_hsa_aie_jit_kernel_info() = default;

    ggml_hsa_aie_jit_kernel_info(std::string_view name,
                                 ggml_hsa_aie_kernel_type type,
                                 fs::path source) :
        name{name}, type{type}, source{std::move(source)} {}

    bool is_valid() const { return !source.empty(); }
};

/**
 * @brief JIT compilation information for operations.
 */
static auto ggml_backend_hsa_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_OP_COUNT> kernels = {};
    kernels[GGML_OP_ADD] = {"ggml_op_add", ggml_hsa_aie_kernel_type::GENERIC, "binary_ops.py"};
    kernels[GGML_OP_SUB] = {"ggml_op_sub", ggml_hsa_aie_kernel_type::GENERIC, "binary_ops.py"};
    kernels[GGML_OP_MUL] = {"ggml_op_mul", ggml_hsa_aie_kernel_type::GENERIC, "binary_ops.py"};
    kernels[GGML_OP_DIV] = {"ggml_op_div", ggml_hsa_aie_kernel_type::GENERIC, "binary_ops.py"};
    kernels[GGML_OP_SQR] = {"ggml_op_sqr", ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_OP_SQRT] = {"ggml_op_sqrt", ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_OP_LOG] = {"ggml_op_log", ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_OP_SIN] = {"ggml_op_sin", ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_OP_COS] = {"ggml_op_cos", ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_OP_MUL_MAT] = {"ggml_op_mul_mat", ggml_hsa_aie_kernel_type::DEVICE_SPECIFIC,
                                "mul_mat.py"};
    return kernels;
}();

/**
 * @brief JIT compilation information for unary operations.
 */
static auto ggml_backend_hsa_unary_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_UNARY_OP_COUNT> kernels = {};
    kernels[GGML_UNARY_OP_ABS] = {"ggml_unary_op_abs", ggml_hsa_aie_kernel_type::GENERIC,
                                  "unary_ops.py"};
    kernels[GGML_UNARY_OP_SGN] = {"ggml_unary_op_sgn", ggml_hsa_aie_kernel_type::GENERIC,
                                  "unary_ops.py"};
    kernels[GGML_UNARY_OP_NEG] = {"ggml_unary_op_neg", ggml_hsa_aie_kernel_type::GENERIC,
                                  "unary_ops.py"};
    kernels[GGML_UNARY_OP_STEP] = {"ggml_unary_op_step", ggml_hsa_aie_kernel_type::GENERIC,
                                   "unary_ops.py"};
    kernels[GGML_UNARY_OP_TANH] = {"ggml_unary_op_tanh", ggml_hsa_aie_kernel_type::GENERIC,
                                   "unary_ops.py"};
    kernels[GGML_UNARY_OP_ELU] = {"ggml_unary_op_elu", ggml_hsa_aie_kernel_type::GENERIC,
                                  "unary_ops.py"};
    kernels[GGML_UNARY_OP_RELU] = {"ggml_unary_op_relu", ggml_hsa_aie_kernel_type::GENERIC,
                                   "unary_ops.py"};
    kernels[GGML_UNARY_OP_SIGMOID] = {"ggml_unary_op_sigmoid", ggml_hsa_aie_kernel_type::GENERIC,
                                      "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU] = {"ggml_unary_op_gelu", ggml_hsa_aie_kernel_type::GENERIC,
                                   "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU_QUICK] = {"ggml_unary_op_gelu_quick",
                                         ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_UNARY_OP_SILU] = {"ggml_unary_op_silu", ggml_hsa_aie_kernel_type::GENERIC,
                                   "unary_ops.py"};
    kernels[GGML_UNARY_OP_HARDSWISH] = {"ggml_unary_op_hardswish",
                                        ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_UNARY_OP_HARDSIGMOID] = {"ggml_unary_op_hardsigmoid",
                                          ggml_hsa_aie_kernel_type::GENERIC, "unary_ops.py"};
    kernels[GGML_UNARY_OP_EXP] = {"ggml_unary_op_exp", ggml_hsa_aie_kernel_type::GENERIC,
                                  "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU_ERF] = {"ggml_unary_op_gelu_erf", ggml_hsa_aie_kernel_type::GENERIC,
                                       "unary_ops.py"};
    return kernels;
}();

/**
 * @brief Returns the JIT compilation information for the given operation.
 */
static const ggml_hsa_aie_jit_kernel_info &
ggml_hsa_get_kernel_jit_info(const ggml_tensor & tensor) {
    assert((tensor.op > GGML_OP_NONE) && (tensor.op < GGML_OP_COUNT) &&
           "Tensor operation index out of bounds");

    if (tensor.op == GGML_OP_UNARY) {
        // for unary operations, we need to get the specific unary operation type
        return ggml_backend_hsa_unary_kernel_jit_info[ggml_get_unary_op(&tensor)];
    }

    return ggml_backend_hsa_kernel_jit_info[tensor.op];
}

/**
 * @brief Returns the absolute kernel source path for the given device and kernel info.
 */
static fs::path ggml_hsa_get_kernel_source_path(const ggml_hsa_device_info::device_info & dev_info,
                                                const ggml_hsa_aie_jit_kernel_info & kernel_info) {
    fs::path path = kernel_path;
    if (kernel_info.type == ggml_hsa_aie_kernel_type::DEVICE_SPECIFIC) {
        path /= dev_info.name;
    }
    path /= kernel_info.source;
    return path;
}

/**
 * @brief Creates a @p py::tuple from the tensor shape.
 */
static py::tuple ggml_hsa_tensor_ne_as_pytuple(const ggml_tensor & tensor) {
    auto shape = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        shape[i] = py::int_(tensor.ne[i]);
    }
    return shape;
}

/**
 * @brief Creates a @p py::tuple from the tensor strides.
 */
static py::tuple ggml_hsa_tensor_nb_as_pytuple(const ggml_tensor & tensor) {
    auto stride = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        stride[i] = py::int_(tensor.nb[i]);
    }
    return stride;
}

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor & tensor,
                                    const std::string & exported_name,
                                    const std::filesystem::path & output_path) {
    using namespace pybind11::literals;

    // retrieve the compilation information for the kernel
    const auto & kernel_jit_info = ggml_hsa_get_kernel_jit_info(tensor);
    if (!kernel_jit_info.is_valid()) {
        GGML_HSA_LOG_INFO("%s: kernel does not exist for operation %s in for tensor \"%s\"",
                          __func__, ggml_op_desc(&tensor), tensor.name);
        return GGML_STATUS_FAILED;
    }

    // compile kernel
    const auto kernel_source_path = ggml_hsa_get_kernel_source_path(dev_info, kernel_jit_info);
    const auto output_directory = output_path / dev_info.name;

    try {
        // convert a GGML tensor to input and output TensorDesc objects
        auto utils = py::module_::import("utils");
        auto tensor_desc_ctor = utils.attr("tensordesc");
        const auto src_tensor_count = ggml_hsa_nsrcs(tensor);
        auto input_tensors = py::list(src_tensor_count);
        for (auto i = 0; i < src_tensor_count; ++i) {
            const auto src_tensor = tensor.src[i];
            input_tensors[i] =
                tensor_desc_ctor("dtype"_a = ggml_type_name(src_tensor->type),
                                 "shape"_a = ggml_hsa_tensor_ne_as_pytuple(*src_tensor),
                                 "stride"_a = ggml_hsa_tensor_nb_as_pytuple(*src_tensor),
                                 "contiguous"_a = ggml_is_contiguous(src_tensor));
        }
        auto output_tensor = tensor_desc_ctor("dtype"_a = ggml_type_name(tensor.type),
                                              "shape"_a = ggml_hsa_tensor_ne_as_pytuple(tensor),
                                              "stride"_a = ggml_hsa_tensor_nb_as_pytuple(tensor),
                                              "contiguous"_a = ggml_is_contiguous(&tensor));

        // compile the kernel
        auto iron_compiler = py::module_::import("build");
        auto compile_kernel = iron_compiler.attr("compile_kernel");
        compile_kernel(
            "kernel_name"_a = kernel_jit_info.name, "kernel_source"_a = kernel_source_path.string(),
            "arch"_a = dev_info.name, "input_tensors"_a = std::move(input_tensors),
            "output_tensor"_a = std::move(output_tensor), "exported_name"_a = exported_name,
            "output_directory"_a = output_directory.string(), "verbose"_a = verbose_compilation);
    } catch (const pybind11::error_already_set & ex) {
        GGML_HSA_LOG_INFO("%s: failed to compile kernel %s for tensor \"%s\" (%s): %s", __func__,
                          exported_name.c_str(), tensor.name, ggml_op_desc(&tensor), ex.what());
        return GGML_STATUS_FAILED;
    }

    GGML_HSA_LOG_INFO("%s: generated kernel %s in %s for tensor \"%s\" (%s)", __func__,
                      exported_name.c_str(), output_directory.c_str(), tensor.name,
                      ggml_op_desc(&tensor));

    return GGML_STATUS_SUCCESS;
}
