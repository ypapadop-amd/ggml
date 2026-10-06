// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/aie-kernel.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <fcntl.h>
#include <unistd.h>

#include "hsa/hsa_ext_amd_aie.h"

#include "ggml-impl.h"

ggml_hsa_aie_kernel::~ggml_hsa_aie_kernel() {
    if (m_executable.handle != 0) {
        GGML_HSA_CHECK_WARN(hsa_executable_destroy(m_executable));
    }
    if (m_reader.handle != 0) {
        GGML_HSA_CHECK_WARN(hsa_code_object_reader_destroy(m_reader));
    }
}

ggml_status ggml_hsa_aie_kernel::load(hsa_agent_t agent,
                                      const std::filesystem::path & path,
                                      const std::string & kernel_name,
                                      std::shared_ptr<ggml_hsa_aie_kernel> & kernel) {
    auto aie_kernel = std::make_shared<ggml_hsa_aie_kernel>();

    // the reader maps the file, so the descriptor is not needed once the reader exists
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        GGML_HSA_LOG_ERROR("%s: could not open file %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    const hsa_status_t reader_status =
        hsa_code_object_reader_create_from_file(fd, &aie_kernel->m_reader);
    close(fd);
    if (reader_status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not read code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(reader_status));
        return GGML_STATUS_FAILED;
    }

    if (auto status =
            hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &aie_kernel->m_executable);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not create executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_load_agent_code_object(aie_kernel->m_executable, agent,
                                                            aie_kernel->m_reader, nullptr, nullptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not load code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_freeze(aie_kernel->m_executable, nullptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not freeze executable for %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    hsa_executable_symbol_t symbol{};
    if (auto status = hsa_executable_get_symbol_by_name(aie_kernel->m_executable,
                                                        kernel_name.c_str(), &agent, &symbol);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: kernel %s not found in %s (%s)", __func__, kernel_name.c_str(),
                           path.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &aie_kernel->m_kernel_object);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernel object of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            &aie_kernel->m_kernarg_size);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernarg size of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    kernel = std::move(aie_kernel);

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_aie_kernel::dispatch(ggml_backend_hsa_context & ctx,
                                          ggml_tensor * src_tensors[],
                                          std::size_t num_src_tensors,
                                          ggml_tensor & dst_tensor) const {
    const auto num_kernargs = num_src_tensors + 1 /* destination tensor */;

    // each kernarg is an address and a size entry (see hsa_amd_aie_kernel_dispatch_packet_t)
    assert(m_kernarg_size == num_kernargs * 2 * sizeof(std::uint64_t));

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
    pkt.kernel_object_low = m_kernel_object & 0xFFFFFFFF;
    pkt.kernel_object_high = m_kernel_object >> 32;
    pkt.num_kernargs = num_kernargs;

    auto queue = ctx.queue;

    // Wait for a free ring slot (queue full when write_index - read_index >= queue->size) and
    // drain; this also drains completed packets. Safe under HSA_QUEUE_TYPE_SINGLE: no other thread
    // advances the write index between this check and the reservation below, so the free slot stays
    // free.
    while (hsa_queue_load_write_index_relaxed(queue) - hsa_queue_load_read_index_scacquire(queue) >=
           queue->size) {
        // A suspended queue never consumes its packets, so the read index stops advancing and the
        // slot this is waiting for never frees. The wait reports that rather than draining, which
        // is what keeps this loop from spinning; fail the dispatch and let graph_compute report it.
        if (const ggml_status status = ggml_hsa_wait_dispatches(ctx);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
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
        return ggml_hsa_flush_dispatches(ctx);
    }

    return GGML_STATUS_SUCCESS;
}
