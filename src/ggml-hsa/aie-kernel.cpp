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
    const auto num_kernargs = num_src_tensors + 1 /* destination tensor */;

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

    // Wait for a free ring slot (queue full when write_index - read_index >= queue->size) and
    // drain; this also drains completed packets. Safe under HSA_QUEUE_TYPE_SINGLE: no other thread
    // advances the write index between this check and the reservation below, so the free slot stays
    // free.
    while (hsa_queue_load_write_index_relaxed(queue) - hsa_queue_load_read_index_scacquire(queue) >=
           queue->size) {
        ggml_hsa_wait_dispatches(ctx);
    }

    // reserve the queue slot
    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    const std::uint64_t packet_id = wr_idx % queue->size;

    // Each ring slot owns a fixed kernarg slot of the same index, sized for the worst case, so the
    // slot claimed above always has room. Reusing slot packet_id is safe only once the prior kernel
    // using it has finished reading its kernargs.
    // kernarg buffer layout (uint64_t entries): [src_ptrs..., dst_ptr, src_sizes..., dst_size]
    // NOTE: under async submission, we need to revisit if reuse must be gated on the completion
    // signal.
    auto * kernargs = static_cast<uint64_t *>(ctx.kernargs.slot(packet_id));

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

    assert(kernarg_idx == num_kernargs * 2 /*kernarg_entries_per_tensor*/);

    pkt.kernarg_address = kernargs;

    *(static_cast<hsa_amd_aie_kernel_dispatch_packet_t *>(queue->base_address) + packet_id) = pkt;

    hsa_signal_add_relaxed(ctx.dispatch_signal, 1);

    // Ring the doorbell only once a full batch is written; it submits every packet up to the most
    // recent write index. Synchronization points flush any remaining pending packets separately.
    ++ctx.n_batched;
    if (ctx.n_batched >= ctx.dispatch_batch_size) {
        ggml_hsa_flush_dispatches(ctx);
    }

    return GGML_STATUS_SUCCESS;
}
