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

    // Allocate the kernarg segment from the device kernarg pool. Over-allocate so the segment
    // can be aligned to the kernel's required alignment; the (unaligned) base is tracked for
    // freeing while the aligned interior pointer is handed to the packet.
    const std::uint32_t align = kernarg_align != 0 ? kernarg_align : 16;
    const std::size_t buf_size = kernarg_size + (static_cast<std::size_t>(align) << 1);

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

    // Grid configuration. AQL grid_size is the total number of work-items (global size).
    //  - Triton kernels bake their launch geometry (fixed workgroup/grid, provided via the
    //    sidecar); they read program ids from the dispatch packet and have no hidden kernargs.
    //  - Hand-written HIP kernels use one work-item per element (workgroup default), deriving the
    //    grid from the element count and filling the COV5 hidden block/group args below.
    const std::uint64_t n_elements = ggml_nelements(&dst_tensor);
    std::uint32_t wg;
    std::uint32_t grid_size_total;
    std::uint32_t num_groups_x;
    if (workgroup_size_x != 0) {
        wg = workgroup_size_x;
        grid_size_total = grid_size_x;
        num_groups_x = wg != 0 ? grid_size_total / wg : 1;
    } else {
        wg = work_group_size != 0 ? work_group_size : 64;
        num_groups_x = (static_cast<std::uint32_t>(n_elements) + wg - 1) / wg;
        if (num_groups_x == 0) {
            num_groups_x = 1;
        }
        grid_size_total = num_groups_x * wg;
    }

    // Populate the kernarg segment using the layout parsed from the code object metadata. Explicit
    // global_buffer arguments are filled in order (sources first, then destination); the single
    // by_value argument is the element count; the HIP hidden arguments are derived from the grid
    // configuration. Offsets/sizes come from the kernel itself, so the layout is ABI-robust.
    const auto put = [&](std::uint32_t off, const void * src, std::uint32_t sz) {
        if (static_cast<std::size_t>(off) + sz <= kernarg_size) {
            std::memcpy(kernarg + off, src, sz);
        }
    };
    // The HIP kernel ABI is positional (argument names are not emitted in the code object
    // metadata, so arguments are matched by kind and position rather than by name). Explicit
    // arguments are, in order:
    //   1. the source buffers, then the destination buffer (global_buffer);
    //   2. the scalar (by_value) parameters: index 0 is the element count, and any further
    //      scalars are taken in order from the op parameter block (ggml_tensor::op_params),
    //      copied raw. This lets ops such as SCALE (one factor) or CLAMP (min, max) reuse this
    //      dispatch path by declaring their extra scalar parameters after N, laid out to match
    //      op_params byte-for-byte.
    const auto * op_params_bytes = reinterpret_cast<const std::byte *>(dst_tensor.op_params);
    std::size_t global_buffer_idx = 0;
    std::size_t by_value_idx = 0;
    std::size_t op_params_cursor = 0;
    for (const auto & arg : args) {
        if (arg.value_kind == "global_buffer") {
            // The first num_src+1 global buffers are the source tensors then the destination.
            // Any additional global buffers are compiler scratch (e.g. Triton's global/profile
            // scratch, which are zero-sized here) and are left null.
            void * ptr = nullptr;
            if (global_buffer_idx < num_src_tensors) {
                ptr = src_tensors[global_buffer_idx]->data;
            } else if (global_buffer_idx == num_src_tensors) {
                ptr = dst_tensor.data;
            }
            std::uint64_t v = reinterpret_cast<std::uintptr_t>(ptr);
            put(arg.offset, &v, arg.size);
            ++global_buffer_idx;
        } else if (arg.value_kind == "by_value") {
            if (by_value_idx == 0) {
                put(arg.offset, &n_elements, arg.size); // first scalar is the element count
            } else if (op_params_cursor + arg.size <= sizeof(dst_tensor.op_params)) {
                // subsequent scalars come from the op parameter block, in order
                put(arg.offset, op_params_bytes + op_params_cursor, arg.size);
                op_params_cursor += arg.size;
            } else {
                GGML_HSA_LOG_WARN("%s: by_value kernarg #%zu (name \"%s\", size %u) exceeds "
                                  "op_params; zeroed",
                                  __func__, by_value_idx, arg.name.c_str(), arg.size);
            }
            ++by_value_idx;
        } else if (arg.value_kind == "hidden_block_count_x") {
            std::uint32_t v = num_groups_x != 0 ? num_groups_x : 1;
            put(arg.offset, &v, arg.size);
        } else if (arg.value_kind == "hidden_block_count_y" ||
                   arg.value_kind == "hidden_block_count_z") {
            std::uint32_t v = 1;
            put(arg.offset, &v, arg.size);
        } else if (arg.value_kind == "hidden_group_size_x") {
            std::uint16_t v = static_cast<std::uint16_t>(wg);
            put(arg.offset, &v, arg.size);
        } else if (arg.value_kind == "hidden_group_size_y" ||
                   arg.value_kind == "hidden_group_size_z") {
            std::uint16_t v = 1;
            put(arg.offset, &v, arg.size);
        } else if (arg.value_kind == "hidden_remainder_x") {
            std::uint16_t v = static_cast<std::uint16_t>(static_cast<std::uint32_t>(n_elements) % wg);
            put(arg.offset, &v, arg.size);
        } else if (arg.value_kind == "hidden_grid_dims") {
            std::uint16_t v = 1; // 1D grid
            put(arg.offset, &v, arg.size);
        }
        // all other hidden arguments (remainder_y/z, global_offset_*, etc.) remain zero
    }

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
    pkt.grid_size_x = grid_size_total;
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
