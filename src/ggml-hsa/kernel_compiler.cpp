#include "kernel_compiler.hpp"

ggml_status ggml_hsa_compile_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                    const ggml_tensor * tensor,
                                    const std::string & kernel_name,
                                    const std::filesystem::path & output_path) {
    return GGML_STATUS_FAILED;
}