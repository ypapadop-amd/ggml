#include "kernel_compiler.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>

#include "ggml-impl.h"

namespace fs = std::filesystem;

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
                                os << "\"--dev " << dev_info.name
                                   << " --dtype " << ggml_type_name(tensor->type)
                                   << " --dims " << ggml_nelements(tensor) << "\"";
                            }};
    return kernels;
}();

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & kernel_name,
                                    const std::filesystem::path & output_path) {
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
    std::ostringstream oss;
    oss << "python3 " << (library_dir / "iron-kernels/compile.py");
    oss << " --output_directory=" << (output_path / dev_info.name);
    oss << " --name=" << kernel_name;
    oss << " --kernel_source=" << (library_dir / kernel_jit_info.kernel_source);
    oss << " --kernel_compile_args=";
    kernel_jit_info.arg_gen(dev_info, tensor, oss);

    if (std::system(oss.str().c_str()) != 0) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}
