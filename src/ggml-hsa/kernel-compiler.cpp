// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "kernel-compiler.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>

#include <pybind11/embed.h>

#include "ggml-impl.h"

namespace fs = std::filesystem;
namespace py = pybind11;

/**
 * @brief Information to drive JIT compilation for a kernel.
 */
struct ggml_hsa_aie_jit_kernel_info {
    std::string_view name; ///< Kernel name.
    fs::path source;       ///< Kernel relative path.

    ggml_hsa_aie_jit_kernel_info() = default;

    ggml_hsa_aie_jit_kernel_info(std::string_view name, fs::path source) :
        name{name}, source{std::move(source)} {}

    bool is_valid() const { return !source.empty(); }
};

/**
 * @brief Outputs the tensor description for use in IRON python kernel scripts.
 */
static void ggml_hsa_output_tensors(const ggml_tensor * tensor, std::ostream & os) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const auto * src_tensor = tensor->src[i];
        if (src_tensor == nullptr) {
            break;
        }
        os << '(';
        ggml_hsa_output_tensor_shape(src_tensor, os, ',');
        os << ")/" << ggml_type_name(src_tensor->type) << ' ';
    }

    os << '(';
    ggml_hsa_output_tensor_shape(tensor, os, ',');
    os << ")/" << ggml_type_name(tensor->type) << ' ';
}

/**
 * @brief JIT compilation information for all operations.
 *
 * Operations that cannot be JIT compiled yet will have default constructed
 * @ref ggml_hsa_aie_jit_kernel_info objects.
 */
static auto ggml_backend_hsa_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_OP_COUNT> kernels = {};
    kernels[GGML_OP_ADD] = {"ggml_op_add", "binary_ops.py"};
    kernels[GGML_OP_SUB] = {"ggml_op_sub", "binary_ops.py"};
    kernels[GGML_OP_MUL] = {"ggml_op_mul", "binary_ops.py"};
    kernels[GGML_OP_DIV] = {"ggml_op_div", "binary_ops.py"};
    kernels[GGML_OP_MUL_MAT] = {"mul_mat", "mul_mat.py"};
    return kernels;
}();

/**
 * @brief Creates a TensorDesc object from the tensor.
 */
template <typename F>
py::object ggml_hsa_tensor_as_tensor_desc(F && ctor_f, const ggml_tensor * tensor) {
    using namespace pybind11::literals;

    // create tuple of dimensions
    const auto ndims = ggml_n_dims(tensor);
    auto dims_tuple = py::tuple(ndims);
    for (auto i = 0; i < ndims; ++i) {
        dims_tuple[i] = py::int_(tensor->ne[i]);
    }

    return std::forward<F>(ctor_f)("shape"_a = std::move(dims_tuple),
                                   "dtype"_a = ggml_type_name(tensor->type));
}

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & exported_name,
                                    const std::filesystem::path & output_path) {
    using namespace pybind11::literals;

    if ((tensor->op < GGML_OP_NONE) || (tensor->op >= GGML_OP_COUNT)) {
        GGML_LOG_ERROR("%s: Tensor operation index out of bounds (%d >= GGML_OP_COUNT)\n", __func__,
                       tensor->op);
        return GGML_STATUS_FAILED;
    }

    const auto & kernel_jit_info = ggml_backend_hsa_kernel_jit_info[tensor->op];
    if (!kernel_jit_info.is_valid()) {
        // no JIT compilable kernel
        return GGML_STATUS_FAILED;
    }

    const auto & library_dir = ggml_hsa_library_path();
    const auto module_path = library_dir / "iron_kernels";
    const auto kernel_source_path = module_path / kernel_jit_info.source;
    const auto output_directory = output_path / dev_info.name;

    py::scoped_interpreter guard{};
    try {
        // import packages
        auto sys = py::module_::import("sys");
        sys.attr("path").attr("append")(module_path.string());
        auto iron_compiler = py::module_::import("compiler");

        // convert a GGML tensor to input and output TensorDesc objects
        auto tensor_desc_ctor = iron_compiler.attr("tensordesc");
        const auto src_tensor_count = ggml_hsa_nsrcs(tensor);
        auto input_tensors = py::list(src_tensor_count);
        for (auto i = 0; i < src_tensor_count; ++i) {
            input_tensors[i] = ggml_hsa_tensor_as_tensor_desc(tensor_desc_ctor, tensor->src[i]);
        }
        auto output_tensor = ggml_hsa_tensor_as_tensor_desc(tensor_desc_ctor, tensor);

        // compile the kernel
        auto compile_kernel = iron_compiler.attr("compile_kernel");
        compile_kernel(
            "kernel_name"_a = kernel_jit_info.name, "kernel_source"_a = kernel_source_path.string(),
            "device"_a = dev_info.name, "input_tensors"_a = std::move(input_tensors),
            "output_tensor"_a = std::move(output_tensor), "exported_name"_a = exported_name,
            "output_directory"_a = output_directory.string());
    } catch (const pybind11::error_already_set & ex) {
        GGML_LOG_ERROR("%s: JIT compilation failed: %s\n", __func__, ex.what());
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}
