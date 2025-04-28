// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "kernel_compiler.hpp"

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
    using argument_generator = void (*)(const ggml_hsa_device_info::device_info &,
                                        const ggml_tensor *,
                                        std::ostream &);

    ggml_op op{GGML_OP_NONE};
    fs::path kernel_source; ///< Kernel relative path from the library.
    argument_generator arg_gen;

    bool is_valid() const { return !kernel_source.empty(); }
};

/**
 * @brief JIT compilation information for all operations.
 *
 * Operations that cannot be JIT compiled yet will have default constructed
 * @ref ggml_hsa_aie_jit_kernel_info objects.
 */
static auto ggml_backend_hsa_kernel_jit_info = []() {
    std::array<ggml_hsa_aie_jit_kernel_info, GGML_OP_COUNT> kernels = {};
    kernels[GGML_OP_ADD] = {GGML_OP_ADD, "iron-kernels/add.py",
                            [](const ggml_hsa_device_info::device_info & dev_info,
                               const ggml_tensor * tensor, std::ostream & os) {
                                os << "--dev " << dev_info.name << " --dtype "
                                   << ggml_type_name(tensor->type) << " --dims "
                                   << ggml_nelements(tensor);
                            }};
    return kernels;
}();

/**
 * @brief Creates the compile arguments for the kernel.
 */
template <typename F>
std::string ggml_hsa_create_kernel_compile_args(const ggml_hsa_device_info::device_info & dev_info,
                                                const ggml_tensor * tensor,
                                                F && f) {
    std::ostringstream oss;
    std::forward<F>(f)(dev_info, tensor, oss);
    return oss.str();
}

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & kernel_name,
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
    const auto kernel_source_path = library_dir / kernel_jit_info.kernel_source;
    const auto kernel_compile_args =
        ggml_hsa_create_kernel_compile_args(dev_info, tensor, kernel_jit_info.arg_gen);
    const auto output_directory = output_path / dev_info.name;

    py::scoped_interpreter guard{};
    auto sys = py::module_::import("sys");
    sys.attr("path").attr("append")(library_dir.string());
    auto compiler = py::module_::import("iron-kernels.compiler");
    compiler.attr("compile_kernel")("name"_a = kernel_name,
                                    "kernel_source"_a = kernel_source_path.string(),
                                    "kernel_compile_args"_a = kernel_compile_args,
                                    "output_directory"_a = output_directory.string());

    return GGML_STATUS_SUCCESS;
}
