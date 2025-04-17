#include "kernels.hpp"

#include "ggml-impl.h"
#include "kernel_discovery.hpp"

ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, ggml_tensor * tensor) {
    auto & info = ggml_hsa_info();
    auto & dev_info = info.devices[ctx.device];

    auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(tensor->extra);
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * dst = tensor;

    if (!tensor_extra.kernel.is_valid()) {
        if (auto status = ggml_hsa_find_aie_kernel(ctx, tensor, tensor_extra.kernel);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    const auto & kernel = tensor_extra.kernel;

    const std::size_t packet_dwords = 12;
    const std::int64_t element_count = ggml_nelements(src0);
    hsa_amd_aie_ert_start_kernel_data_t * cmd_payload = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, 64, 0,
                                                   reinterpret_cast<void **>(&cmd_payload));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate hsa_amd_aie_ert_start_kernel_data_t (%d)\n",
                       __func__, status);
        return GGML_STATUS_FAILED;
    }
    cmd_payload->pdi_addr = kernel.pdi.data; // PDI to use with this command
    cmd_payload->data[0] = 0x3;              // Transaction opcode
    cmd_payload->data[1] = 0x0;
    std::tie(cmd_payload->data[3], cmd_payload->data[2]) = ggml_hsa_addr_to_hilo(kernel.insts.data);
    cmd_payload->data[4] = static_cast<std::uint32_t>(kernel.insts.size);
    std::tie(cmd_payload->data[6], cmd_payload->data[5]) = ggml_hsa_addr_to_hilo(src0->data);
    std::tie(cmd_payload->data[8], cmd_payload->data[7]) = ggml_hsa_addr_to_hilo(src1->data);
    std::tie(cmd_payload->data[10], cmd_payload->data[9]) = ggml_hsa_addr_to_hilo(dst->data);
    cmd_payload->data[11] = element_count * sizeof(std::uint32_t);
    cmd_payload->data[12] = element_count * sizeof(std::uint32_t);
    cmd_payload->data[13] = element_count * sizeof(std::uint32_t);

    ggml_hsa_dispatch_packet(ctx, cmd_payload, packet_dwords);

    return GGML_STATUS_SUCCESS;
}
