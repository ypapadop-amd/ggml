// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/aie-kernel.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "hsa/hsa_ext_amd_aie.h"

#include "ggml-impl.h"

ggml_status ggml_hsa_aie_kernel::dispatch(ggml_backend_hsa_context & ctx,
                                          ggml_tensor * src_tensors[],
                                          std::size_t num_src_tensors,
                                          ggml_tensor & dst_tensor) const {
    const auto & dev_info = ggml_hsa_get_device_info(ctx.device);
    const auto num_kernargs = num_src_tensors + 1 /* destination tensor */;
    const std::size_t kernarg_bytes = num_kernargs * 2 * sizeof(std::uint64_t);

    // create kernargs
    uint64_t * kernargs = nullptr;
    if (auto status =
            hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, kernarg_bytes, 0,
                                         reinterpret_cast<void **>(&kernargs));
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate kernargs (%zu bytes) (%s)", __func__,
                           kernarg_bytes, ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }
    ctx.kernargs.emplace_back(kernargs); // track kernargs for cleanup after dispatch

    // add tensor kernargs
    std::size_t kernarg_idx = 0;
    for (std::size_t src_idx = 0; src_idx < num_src_tensors; ++src_idx) {
        assert(src_tensors[src_idx]->data != nullptr);
        kernargs[kernarg_idx++] = reinterpret_cast<std::uintptr_t>(src_tensors[src_idx]->data);
    }
    assert(dst_tensor.data != nullptr);
    kernargs[kernarg_idx++] = reinterpret_cast<std::uintptr_t>(dst_tensor.data);

    assert(kernarg_idx == num_kernargs);

    // add tensor sizes
    for (std::size_t src_idx = 0; src_idx < num_src_tensors; ++src_idx) {
        kernargs[kernarg_idx++] = ggml_nbytes(src_tensors[src_idx]);
    }
    kernargs[kernarg_idx++] = ggml_nbytes(&dst_tensor);

    assert(kernarg_idx == num_kernargs * 2); // each tensor has 2 kernargs: pointer and size

    // create packet
    hsa_amd_aie_kernel_dispatch_packet_t pkt{};
    pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
    pkt.count = 24;
    pkt.completion_signal.handle = 0; // TODO add ctx.dispatch_signal
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts.data()) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts.data()) >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.insts_size = insts.size();
    pkt.pdi_addr = pdi.data(); // PDI to use with this command

    auto queue = ctx.queue;

    // Queue is full when (write_index - read_index) >= queue->size. Wait until there is space.
    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    while (wr_idx - hsa_queue_load_read_index_scacquire(queue) >= queue->size) {
        ggml_hsa_wait_dispatches(ctx);
    }

    const std::uint64_t packet_id = wr_idx % queue->size;
    *(static_cast<hsa_amd_aie_kernel_dispatch_packet_t *>(queue->base_address) + packet_id) = pkt;

    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

    return GGML_STATUS_SUCCESS;
}
