// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "kernels.hpp"

#include "ggml-impl.h"
#include "kernel-discovery.hpp"

ggml_status ggml_hsa_sqr(ggml_backend_hsa_context & ctx, ggml_tensor * tensor) {
    auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(tensor->extra);
    if (!tensor_extra.kernel.is_valid()) {
        if (auto status = ggml_hsa_create_aie_kernel(ctx, tensor, tensor_extra.kernel);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }

    const auto & kernel = tensor_extra.kernel;
    auto & info = ggml_hsa_info();
    auto & dev_info = info.devices[ctx.device];
    const ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * dst = tensor;
    const std::size_t packet_dwords = 10;
    hsa_amd_aie_ert_start_kernel_data_t * cmd_payload = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, 64, 0,
                                                   reinterpret_cast<void **>(&cmd_payload));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate space for the packet (%d)\n", __func__, status);
        return GGML_STATUS_ALLOC_FAILED;
    }
    cmd_payload->pdi_addr = kernel.pdi.data; // PDI to use with this command
    cmd_payload->data[0] = 0x3;              // Transaction opcode
    cmd_payload->data[1] = 0x0;
    std::tie(cmd_payload->data[3], cmd_payload->data[2]) = ggml_hsa_addr_to_hilo(kernel.insts.data);
    cmd_payload->data[4] = static_cast<std::uint32_t>(kernel.insts.size);
    std::tie(cmd_payload->data[6], cmd_payload->data[5]) = ggml_hsa_addr_to_hilo(src0->data);
    std::tie(cmd_payload->data[8], cmd_payload->data[7]) = ggml_hsa_addr_to_hilo(dst->data);
    cmd_payload->data[9] = ggml_nbytes(src0);
    cmd_payload->data[10] = ggml_nbytes(dst);

    ggml_hsa_dispatch_packet(ctx, cmd_payload, packet_dwords);

    return GGML_STATUS_SUCCESS;
}
