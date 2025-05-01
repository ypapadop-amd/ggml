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
    using gen_t = void (*)(const ggml_hsa_device_info::device_info &,
                           const ggml_tensor *,
                           std::ostream &);

    fs::path kernel_source;          ///< Kernel relative path.
    gen_t kernel_args{};             ///< Kernel compile arguments generator.
    fs::path single_core_source;     ///< Single-core source relative path.
    gen_t single_core_source_args{}; ///< Single-core source compile arguments generator.
    fs::path single_core_object;     ///< Single-core object filename.

    ggml_hsa_aie_jit_kernel_info() = default;

    ggml_hsa_aie_jit_kernel_info(fs::path kernel_source, gen_t kernel_args) :
        kernel_source{std::move(kernel_source)}, kernel_args{kernel_args} {}

    ggml_hsa_aie_jit_kernel_info(fs::path kernel_source,
                                 gen_t kernel_args,
                                 fs::path single_core_source,
                                 gen_t single_core_source_args,
                                 fs::path single_core_object) :
        kernel_source{std::move(kernel_source)},
        kernel_args{kernel_args},
        single_core_source{std::move(single_core_source)},
        single_core_source_args{single_core_source_args},
        single_core_object{std::move(single_core_object)} {}

    bool is_valid() const { return !kernel_source.empty(); }

    bool has_single_core_source() const { return !single_core_source.empty(); }
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
    kernels[GGML_OP_ADD] = {"iron_kernels/add.py",
                            [](const ggml_hsa_device_info::device_info & dev_info,
                               const ggml_tensor * tensor, std::ostream & os) {
                                os << "--dev " << dev_info.name << " --tensors ";
                                ggml_hsa_output_tensors(tensor, os);
                            }};
    kernels[GGML_OP_MUL_MAT] = {
        "iron_kernels/mul_mat.py",
        [](const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor,
           std::ostream & os) {
            os << "-M 32 -K 32 -N 32 -m 8 -k 8 -n 8 --dtype_in i16 --dtype_out i16 --n-aie-cols 4 "
                  "--b-col-maj 0";

            os << "--dev " << dev_info.name << " --dtype " << ggml_type_name(tensor->type)
               << " --dims " << ggml_nelements(tensor);
        },
        "iron_kernels/mm.cc",
        [](const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor * tensor,
           std::ostream & os) { os << "i16_i16_ONLY DIM_M=8 DIM_K=8 DIM_N=8"; },
        "mm_8x8x8.o"};
    return kernels;
}();

/**
 * @brief Creates compile arguments as a string.
 */
template <typename F>
std::string ggml_hsa_create_compile_args(const ggml_hsa_device_info::device_info & dev_info,
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
        ggml_hsa_create_compile_args(dev_info, tensor, kernel_jit_info.kernel_args);
    const auto output_directory = output_path / dev_info.name;

    py::scoped_interpreter guard{};
    try {
        auto sys = py::module_::import("sys");
        sys.attr("path").attr("append")(library_dir.string());
        auto iron_kernels = py::module_::import("iron_kernels");
        auto compile_kernel = iron_kernels.attr("compile_kernel");
        if (!kernel_jit_info.has_single_core_source()) {
            compile_kernel("name"_a = kernel_name, "device"_a = dev_info.name,
                           "kernel_source"_a = kernel_source_path.string(),
                           "kernel_compile_args"_a = kernel_compile_args,
                           "output_directory"_a = output_directory.string());
        } else {
            compile_kernel("name"_a = kernel_name, "device"_a = dev_info.name,
                           "kernel_source"_a = kernel_source_path.string(),
                           "kernel_compile_args"_a = kernel_compile_args,
                           "output_directory"_a = output_directory.string());
        }
    } catch (const pybind11::error_already_set & ex) {
        GGML_LOG_ERROR("%s: Could not JIT compile (%s)\n", __func__, ex.what());
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}
