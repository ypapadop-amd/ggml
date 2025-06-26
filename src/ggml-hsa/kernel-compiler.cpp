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

/**
 * @brief Information to drive JIT compilation for a kernel.
 *
 * Operations that cannot be JIT compiled have default constructed
 * @ref ggml_hsa_aie_jit_kernel_info objects.
 */
struct ggml_hsa_aie_jit_kernel_info {
    std::string_view name; ///< Kernel name.
    fs::path source;       ///< Kernel source file relative to the kernel directory.

    ggml_hsa_aie_jit_kernel_info() = default;

    ggml_hsa_aie_jit_kernel_info(std::string_view name, fs::path source) :
        name{name}, source{std::move(source)} {}

    bool is_valid() const { return !source.empty(); }
};

/**
 * @brief JIT compilation information for operations.
 */
static auto ggml_backend_hsa_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_OP_COUNT> kernels = {};
    kernels[GGML_OP_ADD] = {"ggml_op_add", "binary_ops.py"};
    kernels[GGML_OP_SUB] = {"ggml_op_sub", "binary_ops.py"};
    kernels[GGML_OP_MUL] = {"ggml_op_mul", "binary_ops.py"};
    kernels[GGML_OP_DIV] = {"ggml_op_div", "binary_ops.py"};
    kernels[GGML_OP_SQR] = {"ggml_op_sqr", "unary_ops.py"};
    kernels[GGML_OP_SQRT] = {"ggml_op_sqrt", "unary_ops.py"};
    kernels[GGML_OP_LOG] = {"ggml_op_log", "unary_ops.py"};
    kernels[GGML_OP_SIN] = {"ggml_op_sin", "unary_ops.py"};
    kernels[GGML_OP_COS] = {"ggml_op_cos", "unary_ops.py"};
    kernels[GGML_OP_MUL_MAT] = {"ggml_op_mul_mat", "mul_mat.py"};
    return kernels;
}();

/**
 * @brief JIT compilation information for unary operations.
 */
static auto ggml_backend_hsa_unary_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_UNARY_OP_COUNT> kernels = {};
    kernels[GGML_UNARY_OP_ABS] = {"ggml_unary_op_abs", "unary_ops.py"};
    kernels[GGML_UNARY_OP_SGN] = {"ggml_unary_op_sgn", "unary_ops.py"};
    kernels[GGML_UNARY_OP_NEG] = {"ggml_unary_op_neg", "unary_ops.py"};
    kernels[GGML_UNARY_OP_STEP] = {"ggml_unary_op_step", "unary_ops.py"};
    kernels[GGML_UNARY_OP_TANH] = {"ggml_unary_op_tanh", "unary_ops.py"};
    kernels[GGML_UNARY_OP_ELU] = {"ggml_unary_op_elu", "unary_ops.py"};
    kernels[GGML_UNARY_OP_RELU] = {"ggml_unary_op_relu", "unary_ops.py"};
    kernels[GGML_UNARY_OP_SIGMOID] = {"ggml_unary_op_sigmoid", "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU] = {"ggml_unary_op_gelu", "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU_QUICK] = {"ggml_unary_op_gelu_quick", "unary_ops.py"};
    kernels[GGML_UNARY_OP_SILU] = {"ggml_unary_op_silu", "unary_ops.py"};
    kernels[GGML_UNARY_OP_HARDSWISH] = {"ggml_unary_op_hardswish", "unary_ops.py"};
    kernels[GGML_UNARY_OP_HARDSIGMOID] = {"ggml_unary_op_hardsigmoid", "unary_ops.py"};
    kernels[GGML_UNARY_OP_EXP] = {"ggml_unary_op_exp", "unary_ops.py"};
    kernels[GGML_UNARY_OP_GELU_ERF] = {"ggml_unary_op_gelu_erf", "unary_ops.py"};
    return kernels;
}();

/**
 * @brief Returns the JIT compilation information for the given operation.
 */
static const ggml_hsa_aie_jit_kernel_info &
ggml_hsa_get_kernel_jit_info(const ggml_tensor * tensor) {
    assert((tensor->op > GGML_OP_NONE) && (tensor->op < GGML_OP_COUNT) &&
           "Tensor operation index out of bounds");

    if (tensor->op == GGML_OP_UNARY) {
        // for unary operations, we need to get the specific unary operation type
        return ggml_backend_hsa_unary_kernel_jit_info[ggml_get_unary_op(tensor)];
    }

    return ggml_backend_hsa_kernel_jit_info[tensor->op];
}

/**
 * @brief Creates a @p py::tuple from the tensor dimensions.
 */
static py::tuple ggml_hsa_tensor_dims_as_pytuple(const ggml_tensor * tensor) {
    auto dims_tuple = py::tuple(GGML_MAX_DIMS);
    for (auto i = 0; i < GGML_MAX_DIMS; ++i) {
        dims_tuple[i] = py::int_(tensor->ne[i]);
    }
    return dims_tuple;
}

/**
 * @brief Returns if the tensor is not supported.
 */
static bool ggml_hsa_unsupported_tensor(const ggml_tensor * tensor) {
    // non-contiguous tensors are not yet supported
    auto unsupported_tensor = [](const ggml_tensor * tensor) {
        return tensor != nullptr && !ggml_is_contiguous(tensor);
    };

    return unsupported_tensor(tensor) ||
           std::any_of(tensor->src, std::next(tensor->src, GGML_MAX_SRC), unsupported_tensor);
}

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & exported_name,
                                    const std::filesystem::path & output_path) {
    using namespace pybind11::literals;

    if (ggml_hsa_unsupported_tensor(tensor)) {
        GGML_LOG_INFO("%s: Tensor \"%s\" is not supported\n", __func__, ggml_get_name(tensor));
        return GGML_STATUS_FAILED;
    }

    // retrieve the JIT compilation information for the kernel
    const auto & kernel_jit_info = ggml_hsa_get_kernel_jit_info(tensor);
    if (!kernel_jit_info.is_valid()) {
        // no JIT compilable kernel
        return GGML_STATUS_FAILED;
    }

    // JIT compile kernel
    const auto kernel_path = ggml_hsa_library_dir / "iron-kernels";
    const auto device_kernel_path = kernel_path / dev_info.name;
    const auto kernel_source_path = device_kernel_path / kernel_jit_info.source;
    const auto output_directory = output_path / dev_info.name;

    py::scoped_interpreter guard{};
    try {
        // import build and kernel scripts
        auto sys = py::module_::import("sys");
        sys.attr("path").attr("append")(kernel_path.string());
        sys.attr("path").attr("append")(device_kernel_path.string());
        auto iron_compiler = py::module_::import("build");

        // convert a GGML tensor to input and output TensorDesc objects
        auto tensor_desc_ctor = iron_compiler.attr("tensordesc");
        const auto src_tensor_count = ggml_hsa_nsrcs(tensor);
        auto input_tensors = py::list(src_tensor_count);
        for (auto i = 0; i < src_tensor_count; ++i) {
            const auto src_tensor = tensor->src[i];
            input_tensors[i] =
                tensor_desc_ctor("shape"_a = ggml_hsa_tensor_dims_as_pytuple(src_tensor),
                                 "dtype"_a = ggml_type_name(src_tensor->type));
        }
        auto output_tensor = tensor_desc_ctor("shape"_a = ggml_hsa_tensor_dims_as_pytuple(tensor),
                                              "dtype"_a = ggml_type_name(tensor->type));

        // compile the kernel
        auto compile_kernel = iron_compiler.attr("compile_kernel");
        compile_kernel(
            "kernel_name"_a = kernel_jit_info.name, "kernel_source"_a = kernel_source_path.string(),
            "device"_a = dev_info.name, "input_tensors"_a = std::move(input_tensors),
            "output_tensor"_a = std::move(output_tensor), "exported_name"_a = exported_name,
            "output_directory"_a = output_directory.string(), "verbose"_a = verbose_compilation);
    } catch (const pybind11::error_already_set & ex) {
        GGML_LOG_ERROR("%s: JIT compilation for kernel %s failed:\n%s\n", __func__,
                       exported_name.c_str(), ex.what());
        return GGML_STATUS_FAILED;
    }

    GGML_LOG_INFO("%s: Generated kernel %s in %s\n", __func__, exported_name.c_str(),
                  output_directory.c_str());

    return GGML_STATUS_SUCCESS;
}
