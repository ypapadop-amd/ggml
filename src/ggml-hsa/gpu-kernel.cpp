// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/gpu-kernel.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ggml-impl.h"

namespace {

/**
 * @brief Rounds @p value up to the next multiple of @p alignment (a power of two).
 */
std::uintptr_t align_up(std::uintptr_t value, std::uintptr_t alignment) {
    assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
    return (value + (alignment - 1)) & ~(alignment - 1);
}

/**
 * @brief Writes an AQL kernel dispatch packet into the queue ring buffer.
 *
 * Copies every field except the header/setup, which must be written last and atomically.
 */
void write_aql_to_queue(const hsa_kernel_dispatch_packet_t & pkt, hsa_queue_t * queue,
                        std::uint64_t pkt_idx) {
    const std::uint32_t mask = queue->size - 1;
    auto * slot = &(static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[pkt_idx & mask];

    slot->setup = pkt.setup;
    slot->workgroup_size_x = pkt.workgroup_size_x;
    slot->workgroup_size_y = pkt.workgroup_size_y;
    slot->workgroup_size_z = pkt.workgroup_size_z;
    slot->reserved0 = 0;
    slot->grid_size_x = pkt.grid_size_x;
    slot->grid_size_y = pkt.grid_size_y;
    slot->grid_size_z = pkt.grid_size_z;
    slot->private_segment_size = pkt.private_segment_size;
    slot->group_segment_size = pkt.group_segment_size;
    slot->kernel_object = pkt.kernel_object;
    slot->kernarg_address = pkt.kernarg_address;
    slot->reserved2 = 0;
    slot->completion_signal = pkt.completion_signal;
}

/**
 * @brief Atomically writes the packet header and setup (release ordering).
 *
 * Must be done after all other fields are set so the packet processor never observes a
 * partially-written KERNEL_DISPATCH packet.
 */
void atomic_set_packet_header(std::uint16_t header, std::uint16_t setup,
                              hsa_kernel_dispatch_packet_t * queue_packet) {
    __atomic_store_n(reinterpret_cast<std::uint32_t *>(queue_packet),
                     header | (static_cast<std::uint32_t>(setup) << 16), __ATOMIC_RELEASE);
}

} // namespace

ggml_hsa_gpu_kernel::~ggml_hsa_gpu_kernel() {
    if (executable.handle != 0) {
        hsa_executable_destroy(executable);
    }
}

ggml_status ggml_hsa_gpu_kernel::dispatch(ggml_backend_hsa_context & ctx,
                                          ggml_tensor * src_tensors[],
                                          std::size_t num_src_tensors,
                                          ggml_tensor & dst_tensor) const {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = ggml_hsa_get_device_info(ctx.device);

    // Explicit kernel arguments: one pointer per source, one destination pointer, the element
    // count. The HIP-compiled kernels use the signature (src0, ..., srcN, dst, uint64_t N).
    const std::size_t num_ptrs = num_src_tensors + 1; // sources + destination
    const std::size_t explicit_size = num_ptrs * sizeof(std::uint64_t) + sizeof(std::uint64_t);

    // Allocate the kernarg segment from the device kernarg pool. Over-allocate so the segment
    // can be aligned to the kernel's required alignment; the (unaligned) base is tracked for
    // freeing while the aligned interior pointer is handed to the packet.
    const std::uint32_t align = kernarg_align != 0 ? kernarg_align : 16;
    const std::size_t buf_size =
        (explicit_size >= kernarg_size ? explicit_size : kernarg_size) + (static_cast<std::size_t>(align) << 1);

    void * kernarg_base = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, buf_size, 0,
                                                   &kernarg_base);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate kernargs (%zu bytes) (%s)", __func__, buf_size,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }
    ctx.kernargs.emplace_back(kernarg_base); // track base allocation for cleanup after dispatch

    auto * kernarg = reinterpret_cast<std::byte *>(
        align_up(reinterpret_cast<std::uintptr_t>(kernarg_base), align));

    // Zero the entire kernarg segment so any hidden fields we do not set explicitly are well
    // defined.
    std::memset(kernarg, 0, kernarg_size);

    // Explicit arguments: source pointers, destination pointer, element count.
    auto * args = reinterpret_cast<std::uint64_t *>(kernarg);
    std::size_t idx = 0;
    for (std::size_t i = 0; i < num_src_tensors; ++i) {
        assert(src_tensors[i]->data != nullptr);
        args[idx++] = reinterpret_cast<std::uintptr_t>(src_tensors[i]->data);
    }
    assert(dst_tensor.data != nullptr);
    args[idx++] = reinterpret_cast<std::uintptr_t>(dst_tensor.data);
    const std::uint64_t n_elements = ggml_nelements(&dst_tensor);
    args[idx++] = n_elements;

    // Grid configuration. AQL grid_size is the total number of work-items (global size), rounded
    // up to a whole number of workgroups; hidden_block_count is the number of workgroups.
    const std::uint32_t wg = work_group_size != 0 ? work_group_size : 64;
    const std::uint32_t num_groups_x = (static_cast<std::uint32_t>(n_elements) + wg - 1) / wg;
    const std::uint32_t grid_size_x = (num_groups_x != 0 ? num_groups_x : 1) * wg;

    // HIP implicit/hidden arguments (COV5 layout), placed immediately after the explicit args.
    // Required because the kernels use blockIdx/blockDim/threadIdx, which the compiler lowers to
    // reads from these hidden kernarg slots.
    std::byte * hidden = kernarg + explicit_size;
    const auto put_u32 = [&](std::size_t off, std::uint32_t v) { std::memcpy(hidden + off, &v, sizeof(v)); };
    const auto put_u16 = [&](std::size_t off, std::uint16_t v) { std::memcpy(hidden + off, &v, sizeof(v)); };
    const auto put_u64 = [&](std::size_t off, std::uint64_t v) { std::memcpy(hidden + off, &v, sizeof(v)); };
    put_u32(0, num_groups_x != 0 ? num_groups_x : 1); // hidden_block_count_x
    put_u32(4, 1);                                     // hidden_block_count_y
    put_u32(8, 1);                                     // hidden_block_count_z
    put_u16(12, static_cast<std::uint16_t>(wg));       // hidden_group_size_x
    put_u16(14, 1);                                    // hidden_group_size_y
    put_u16(16, 1);                                    // hidden_group_size_z
    put_u16(18, static_cast<std::uint16_t>(static_cast<std::uint32_t>(n_elements) % wg)); // remainder_x
    put_u16(20, 0);                                    // hidden_remainder_y
    put_u16(22, 0);                                    // hidden_remainder_z
    // bytes 24..31 are padding so the global-offset trio is 8-aligned
    put_u64(32, 0);                                    // hidden_global_offset_x
    put_u64(40, 0);                                    // hidden_global_offset_y
    put_u64(48, 0);                                    // hidden_global_offset_z
    put_u16(56, 1);                                    // hidden_grid_dims (1D)

    // Grant the GPU and CPU agents access to the kernarg allocation (unified memory).
    std::vector<hsa_agent_t> agents;
    if (dev_info.agent.handle != 0) {
        agents.push_back(dev_info.agent);
    }
    if (info.cpu_agent.handle != 0) {
        agents.push_back(info.cpu_agent);
    }
    if (!agents.empty()) {
        if (auto status = hsa_amd_agents_allow_access(static_cast<std::uint32_t>(agents.size()),
                                                      agents.data(), nullptr, kernarg_base);
            status != HSA_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to grant kernarg access (%s)", __func__,
                               ggml_hsa_get_status_string(status));
            return GGML_STATUS_FAILED;
        }
    }

    // Build the AQL kernel dispatch packet.
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pkt.workgroup_size_x = static_cast<std::uint16_t>(wg);
    pkt.workgroup_size_y = 1;
    pkt.workgroup_size_z = 1;
    pkt.grid_size_x = grid_size_x;
    pkt.grid_size_y = 1;
    pkt.grid_size_z = 1;
    pkt.private_segment_size = private_segment_size;
    pkt.group_segment_size = group_segment_size;
    pkt.kernel_object = kernel_object;
    pkt.kernarg_address = kernarg;
    pkt.completion_signal = ctx.dispatch_signal;

    auto * queue = ctx.queue;

    // Reserve a slot, waiting for space if the queue is full.
    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    while (wr_idx - hsa_queue_load_read_index_scacquire(queue) >= queue->size) {
        ggml_hsa_wait_dispatches(ctx);
    }

    // Account for this dispatch on the completion signal; the packet processor decrements it back
    // to zero when the kernel finishes (see ggml_hsa_wait_dispatches).
    hsa_signal_add_screlease(ctx.dispatch_signal, 1);

    write_aql_to_queue(pkt, queue, wr_idx);

    std::uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

    const std::uint32_t mask = queue->size - 1;
    atomic_set_packet_header(
        header, pkt.setup,
        &(static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[wr_idx & mask]);

    // Ring the doorbell with the packet index to launch the kernel.
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

    return GGML_STATUS_SUCCESS;
}
