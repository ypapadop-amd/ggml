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
    // kernarg buffer layout (uint64_t entries): [src_ptrs..., dst_ptr, src_sizes..., dst_size]
    constexpr std::size_t kernarg_entries_per_tensor = 2;

    const auto num_kernargs = num_src_tensors + 1 /* destination tensor */;
    const std::size_t kernarg_bytes =
        num_kernargs * kernarg_entries_per_tensor * sizeof(std::uint64_t);

    // number of bytes in the packet after completion_signal up to kernarg_address; the AIE dispatch
    // packet ABI requires this to be exactly 24 (see hsa_amd_aie_kernel_dispatch_packet_t)
    constexpr std::uint16_t aie_packet_count = 24;

    // create packet (kernarg_address is filled in once the kernargs are allocated below)
    hsa_amd_aie_kernel_dispatch_packet_t pkt{};
    pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                 (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
    pkt.count = aie_packet_count;
    pkt.completion_signal = ctx.dispatch_signal;
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts.data()) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts.data()) >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.insts_size = insts.size();
    pkt.pdi_addr = pdi.data(); // PDI to use with this command

    auto queue = ctx.queue;

    // Wait until the queue ring has a free slot. Queue is full when (write_index - read_index) >=
    // queue->size. The wait synchronizes and recycles the kernarg arena, so the kernarg buffer must
    // be obtained *after* this point to avoid handing out a slice that a still-in-flight packet
    // references. We poll the write index instead of reserving one so that a kernarg allocation
    // failure below cannot leave a reserved-but-unused slot that permanently consumes ring
    // capacity. This is safe because the queue is single-producer (HSA_QUEUE_TYPE_SINGLE): between
    // this check and the reservation below no other thread advances the write index, and the read
    // index only moves forward, so the observed free slot remains free.
    while (hsa_queue_load_write_index_relaxed(queue) - hsa_queue_load_read_index_scacquire(queue) >=
           queue->size) {
        ggml_hsa_wait_dispatches(ctx);
    }

    // create kernargs
    auto * kernargs = static_cast<uint64_t *>(ctx.kernargs.allocate(kernarg_bytes));
    if (kernargs == nullptr) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate kernargs (%zu bytes)", __func__, kernarg_bytes);
        return GGML_STATUS_ALLOC_FAILED;
    }
    pkt.kernarg_address = kernargs;

    // reserve the queue slot now that the kernargs are in place
    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);

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

    assert(kernarg_idx == num_kernargs * kernarg_entries_per_tensor);

    const std::uint64_t packet_id = wr_idx % queue->size;
    *(static_cast<hsa_amd_aie_kernel_dispatch_packet_t *>(queue->base_address) + packet_id) = pkt;

    hsa_signal_add_relaxed(ctx.dispatch_signal, 1);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

    return GGML_STATUS_SUCCESS;
}
