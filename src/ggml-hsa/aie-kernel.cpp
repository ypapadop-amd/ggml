// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/aie-kernel.hpp"
#include "ggml-impl.h"

/**
 * @brief Dispatches a packet to an AIE agent queue.
 *
 * @todo @p dispatch_signal is not used yet.
 *
 * @param[in] queue queue to enqueue the packet
 * @param[in] signal signal to notify for packet completion
 * @param[in] payload packet payload
 * @param[in] payload_size payload size in dwords
 */
static void ggml_hsa_aie_dispatch_packet(hsa_queue_t * queue,
                                         hsa_signal_t /* dispatch_signal */,
                                         hsa_amd_aie_ert_start_kernel_data_t * payload,
                                         std::size_t payload_size) {
    hsa_amd_aie_ert_packet_t pkt{};
    pkt.header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
    pkt.state = HSA_AMD_AIE_ERT_STATE_NEW;
    pkt.count = payload_size;
    pkt.opcode = HSA_AMD_AIE_ERT_START_CU;
    pkt.payload_data = reinterpret_cast<std::uint64_t>(payload);
    // TODO add pkt->completion_signal = dispatch_signal

    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    const std::uint64_t packet_id = wr_idx % queue->size;
    *(static_cast<hsa_amd_aie_ert_packet_t *>(queue->base_address) + packet_id) = pkt;

    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
}

ggml_status ggml_hsa_aie_kernel::dispatch(ggml_backend_hsa_context & ctx,
                                          ggml_tensor * src_tensors[],
                                          std::size_t num_src_tensors,
                                          ggml_tensor & dst_tensor) const {
    const auto & dev_info = ggml_hsa_get_device_info(ctx.device);
    const std::size_t packet_dwords =
        3 /* instructions */ + (num_src_tensors + 1) * 3 /* source and destination tensors */;
    void * ptr = nullptr;
    if (auto status =
            hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, 64, 0, &ptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate hsa_queue packet storage (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }
    ctx.pending_payloads.emplace_back(ptr);

    auto cmd_payload = static_cast<hsa_amd_aie_ert_start_kernel_data_t *>(ptr);
    cmd_payload->pdi_addr =
        const_cast<void *>(static_cast<const void *>(pdi.data())); // PDI to use with this command

    // transaction opcode; not counted in packet_dwords (see assert below)
    cmd_payload->data[0] = 0x3;
    cmd_payload->data[1] = 0x0;

    std::size_t dword_idx = 2;

    // instructions; 3 dwords
    cmd_payload->data[dword_idx] = reinterpret_cast<std::uintptr_t>(insts.data()) & 0xFFFFFFFF;
    cmd_payload->data[dword_idx + 1] = reinterpret_cast<std::uintptr_t>(insts.data()) >> 32;
    cmd_payload->data[dword_idx + 2] = static_cast<std::uint32_t>(insts.size());
    dword_idx += 3;

    // sources; 2 dwords each
    for (std::size_t src_idx = 0; src_idx < num_src_tensors; ++src_idx, dword_idx += 2) {
        cmd_payload->data[dword_idx] =
            reinterpret_cast<std::uintptr_t>(src_tensors[src_idx]->data) & 0xFFFFFFFF;
        cmd_payload->data[dword_idx + 1] =
            reinterpret_cast<std::uintptr_t>(src_tensors[src_idx]->data) >> 32;
    }

    // destination; 2 dwords
    cmd_payload->data[dword_idx] = reinterpret_cast<std::uintptr_t>(dst_tensor.data) & 0xFFFFFFFF;
    cmd_payload->data[dword_idx + 1] = reinterpret_cast<std::uintptr_t>(dst_tensor.data) >> 32;
    dword_idx += 2;

    // sizes; 1 dword per tensor
    for (std::size_t src_idx = 0; src_idx < num_src_tensors; ++src_idx, ++dword_idx) {
        cmd_payload->data[dword_idx] = ggml_nbytes(src_tensors[src_idx]);
    }
    cmd_payload->data[dword_idx] = ggml_nbytes(&dst_tensor);

    assert(dword_idx == packet_dwords + 1); // 2 extra uncounted dwords (transaction opcode)

    ggml_hsa_aie_dispatch_packet(ctx.queue, ctx.dispatch_signal, cmd_payload, packet_dwords);

    return GGML_STATUS_SUCCESS;
}
