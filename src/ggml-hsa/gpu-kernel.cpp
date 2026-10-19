// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/gpu-kernel.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ggml-impl.h"

namespace {

/**
 * @brief Writes an AQL kernel dispatch packet into the queue ring buffer.
 *
 * Copies every field except the header/setup, which must be written last and atomically.
 */
void write_aql_to_queue(const hsa_kernel_dispatch_packet_t & pkt,
                        hsa_queue_t * queue,
                        std::uint64_t pkt_idx) {
    const std::uint32_t mask = queue->size - 1;
    auto * slot =
        &(static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[pkt_idx & mask];

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
void atomic_set_packet_header(std::uint16_t header,
                              std::uint16_t setup,
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
    auto * queue = ctx.queue;

    if (kernarg_size > ctx.kernargs.slot_size() || kernarg_align > ggml_hsa_gpu_kernarg_alignment) {
        GGML_HSA_LOG_ERROR("%s: kernarg segment (%u bytes, alignment %u) does not fit a kernarg "
                           "slot (%zu bytes, alignment %zu)",
                           __func__, kernarg_size, kernarg_align, ctx.kernargs.slot_size(),
                           ggml_hsa_gpu_kernarg_alignment);
        return GGML_STATUS_FAILED;
    }

    // Grid configuration. AQL grid_size is the total number of work-items (global size).
    //  - Triton kernels bake their launch geometry (fixed workgroup/grid, provided via the
    //    sidecar); they read program ids from the dispatch packet and have no hidden kernargs.
    //  - Hand-written HIP kernels use one work-item per element (workgroup default), deriving the
    //    grid from the element count and filling the COV5 hidden block/group args below.
    // Computed before a queue slot is reserved, so a launch that does not fit leaves none behind.
    const std::uint64_t n_elements = ggml_nelements(&dst_tensor);
    std::uint32_t wg;
    std::uint64_t grid_size_total;
    if (workgroup_size_x != 0) {
        wg = workgroup_size_x;
        grid_size_total = grid_size_x;
    } else {
        wg = work_group_size != 0 ? work_group_size : 64;
        if (wg > UINT16_MAX) {
            // the packet's workgroup size is 16 bits; a larger value would be truncated
            GGML_HSA_LOG_ERROR("%s: workgroup of %u work-items exceeds the AQL packet's 16-bit "
                               "workgroup size",
                               __func__, wg);
            return GGML_STATUS_FAILED;
        }
        const std::uint64_t num_groups = std::max<std::uint64_t>((n_elements + wg - 1) / wg, 1);
        grid_size_total = num_groups * wg;
    }
    if (grid_size_total > UINT32_MAX) {
        GGML_HSA_LOG_ERROR("%s: grid of %llu work-items for %llu elements exceeds the AQL "
                           "packet's 32-bit grid size",
                           __func__, static_cast<unsigned long long>(grid_size_total),
                           static_cast<unsigned long long>(n_elements));
        return GGML_STATUS_FAILED;
    }
    const auto num_groups_x = static_cast<std::uint32_t>(grid_size_total / wg);

    // Reserve the queue slot. A kernarg segment must stay unchanged until its kernel completes, so
    // the acquire waits for the slot's previous packet to complete, not merely to be consumed.
    std::uint64_t wr_idx = 0;
    if (const ggml_status status = ggml_hsa_acquire_packet_slot(ctx, wr_idx);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }
    const std::uint64_t packet_id = wr_idx % queue->size;
    auto * kernarg = static_cast<std::byte *>(ctx.kernargs.slot(packet_id));

    // Zero the entire kernarg segment so any hidden fields we do not set explicitly are well
    // defined.
    std::memset(kernarg, 0, kernarg_size);

    // Populate the kernarg segment using the layout parsed from the code object metadata. Explicit
    // global_buffer arguments are filled in order (sources first, then destination); the single
    // by_value argument is the element count; the HIP hidden arguments are derived from the grid
    // configuration. Offsets/sizes come from the kernel itself, so the layout is ABI-robust.
    // ggml_hsa_create_gpu_kernel rejects a kernel whose arguments do not all lie in its segment, or
    // whose arguments filled from a fixed-size value below differ from ggml_hsa_gpu_fixed_arg_size.
    // put_scalar checks the value's type against that too, in every build: a mismatch would
    // over-read the value or write past the argument.
    const auto put = [&](std::uint32_t off, const void * src, std::uint32_t sz) {
        assert(static_cast<std::size_t>(off) + sz <= kernarg_size);
        std::memcpy(kernarg + off, src, sz);
    };
    const auto put_scalar = [&](const auto & arg, auto value) {
        GGML_ASSERT(arg.size == sizeof(value));
        put(arg.offset, &value, sizeof(value));
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
            put_scalar(arg, ggml_hsa_gpu_pointer_arg_t{reinterpret_cast<std::uintptr_t>(ptr)});
            ++global_buffer_idx;
        } else if (arg.value_kind == "by_value") {
            if (by_value_idx == 0) {
                // first scalar is the element count
                put_scalar(arg, ggml_hsa_gpu_element_count_arg_t{n_elements});
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
            put_scalar(arg, ggml_hsa_gpu_block_count_arg_t{num_groups_x});
        } else if (arg.value_kind == "hidden_block_count_y" ||
                   arg.value_kind == "hidden_block_count_z") {
            put_scalar(arg, ggml_hsa_gpu_block_count_arg_t{1});
        } else if (arg.value_kind == "hidden_group_size_x") {
            put_scalar(arg, static_cast<ggml_hsa_gpu_group_dim_arg_t>(wg));
        } else if (arg.value_kind == "hidden_group_size_y" ||
                   arg.value_kind == "hidden_group_size_z") {
            put_scalar(arg, ggml_hsa_gpu_group_dim_arg_t{1});
        } else if (arg.value_kind == "hidden_remainder_x") {
            put_scalar(arg, static_cast<ggml_hsa_gpu_group_dim_arg_t>(n_elements % wg));
        } else if (arg.value_kind == "hidden_grid_dims") {
            put_scalar(arg, ggml_hsa_gpu_group_dim_arg_t{1}); // 1D grid
        }
        // all other hidden arguments (remainder_y/z, global_offset_*, etc.) remain zero
    }

    // Build the AQL kernel dispatch packet.
    hsa_kernel_dispatch_packet_t pkt{};
    pkt.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    pkt.workgroup_size_x = static_cast<std::uint16_t>(wg);
    pkt.workgroup_size_y = 1;
    pkt.workgroup_size_z = 1;
    pkt.grid_size_x = static_cast<std::uint32_t>(grid_size_total);
    pkt.grid_size_y = 1;
    pkt.grid_size_z = 1;
    pkt.private_segment_size = private_segment_size;
    // Triton kernels request their shared memory as dynamic LDS, which the code object does not
    // count in its fixed group segment size; the launch sidecar carries it instead.
    pkt.group_segment_size = group_segment_size + dynamic_group_segment_size;
    pkt.kernel_object = kernel_object;
    pkt.kernarg_address = kernarg;
    pkt.completion_signal = ctx.dispatch_signal;

    write_aql_to_queue(pkt, queue, wr_idx);

    // The barrier bit makes the packet processor wait for every earlier packet to complete before
    // launching this one; without it, consecutive dispatches may overlap, and a dispatch would race
    // with the one producing its sources. Completion tracking needs it too: packets must complete
    // in order for the count in ggml_backend_hsa_context::dispatch_signal to say which packets
    // finished, and so which ring and kernarg slots are free.
    std::uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    header |= 1 << HSA_PACKET_HEADER_BARRIER;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    atomic_set_packet_header(
        header, pkt.setup,
        &(static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address))[packet_id]);

    // Ring the doorbell only once a full batch is written (see ggml_hsa_flush_dispatches).
    ++ctx.n_batched;
    if (ctx.n_batched >= ctx.dispatch_batch_size) {
        return ggml_hsa_flush_dispatches(ctx);
    }

    return GGML_STATUS_SUCCESS;
}
