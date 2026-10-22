// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/aie-kernel.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hsa/hsa_ext_amd_aie.h"

#include "ggml-impl.h"

ggml_status ggml_hsa_aie_kernel::load(hsa_agent_t agent,
                                      const std::filesystem::path & path,
                                      const std::string & kernel_name) {
    std::vector<char> blob; // not needed afterward: the kernel is found by kind, not metadata
    if (auto status = load_executable(agent, path, kernel_name, blob);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    // Each hsaco holds exactly one kernel, found by kind rather than by name: a kernel packed from
    // a full ELF is named by the ELF (e.g. "main:sequence"), not after the hsaco.
    struct kernel_search {
        hsa_executable_symbol_t symbol{};
        std::size_t count = 0;
    } search;
    if (auto status = hsa_executable_iterate_agent_symbols(
            executable(), agent,
            [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t symbol, void * data) {
                hsa_symbol_kind_t kind{};
                if (auto status = hsa_executable_symbol_get_info(
                        symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
                    status != HSA_STATUS_SUCCESS) {
                    return status;
                }
                if (kind == HSA_SYMBOL_KIND_KERNEL) {
                    auto & search = *static_cast<kernel_search *>(data);
                    search.symbol = symbol;
                    ++search.count;
                }
                return HSA_STATUS_SUCCESS;
            },
            &search);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not list the kernels in %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }
    if (search.count != 1) {
        GGML_HSA_LOG_ERROR("%s: %s holds %zu kernels, expected exactly one for kernel %s", __func__,
                           path.c_str(), search.count, kernel_name.c_str());
        return GGML_STATUS_FAILED;
    }
    const hsa_executable_symbol_t symbol = search.symbol;

    std::uint64_t kernel_obj{};
    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_obj);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernel object of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }
    set_kernel_object(kernel_obj);

    std::uint32_t kernarg_sz{};
    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_sz);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernarg size of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }
    set_kernarg_size(kernarg_sz);

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_aie_kernel::dispatch(ggml_backend_hsa_context & ctx,
                                          ggml_tensor * src_tensors[],
                                          std::size_t num_src_tensors,
                                          ggml_tensor & dst_tensor) const {
    const auto num_kernargs = num_src_tensors + 1 /* destination tensor */;

    // each kernarg is an address and a size entry (see hsa_amd_aie_kernel_dispatch_packet_t); this
    // is reported by the hsaco itself, so check it at runtime rather than assert, which release
    // builds compile out, and dispatching a mismatched layout would write past the kernarg slot
    const auto expected_kernarg_size = num_kernargs * 2 * sizeof(std::uint64_t);
    if (kernarg_size() != expected_kernarg_size) {
        GGML_HSA_LOG_ERROR("%s: kernel kernarg size %u does not match the %zu bytes expected for "
                           "%zu arguments",
                           __func__, kernarg_size(), expected_kernarg_size, num_kernargs);
        return GGML_STATUS_FAILED;
    }

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
    pkt.kernel_object_low = static_cast<std::uint32_t>(kernel_object() & 0xFFFFFFFF);
    pkt.kernel_object_high = static_cast<std::uint32_t>(kernel_object() >> 32);
    pkt.num_kernargs = num_kernargs;

    auto queue = ctx.queue;

    // Reserve the queue slot: the acquire waits for the slot's previous packet to complete, which
    // also frees its kernarg slot. The AIE queue runs packets in order, which the shared
    // completion count relies on.
    std::uint64_t wr_idx = 0;
    if (const ggml_status status = ggml_hsa_acquire_packet_slot(ctx, wr_idx);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }
    pkt.completion_signal = ctx.dispatch_signal;
    const std::uint64_t packet_id = wr_idx % queue->size;

    // Each ring slot owns a fixed kernarg slot of the same index, sized for the worst case, so the
    // slot claimed above always has room; the prior kernel using it has completed.
    // kernarg buffer layout (uint64_t entries): [src_ptrs..., dst_ptr, src_sizes..., dst_size]
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

    // Ring the doorbell only once a full batch is written; it submits every packet up to the most
    // recent write index. Synchronization points flush any remaining pending packets separately.
    ++ctx.n_batched;
    if (ctx.n_batched >= ctx.dispatch_batch_size) {
        return ggml_hsa_flush_dispatches(ctx);
    }

    return GGML_STATUS_SUCCESS;
}
