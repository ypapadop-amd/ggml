// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "kernel-discovery.hpp"

#include "ggml-hsa/common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>

#ifdef GGML_HSA_CPU_FALLBACK
#include "ggml-cpu.h"
#endif

// The following data types are natively supported by AIEs:
// - GGML_TYPE_F32 (emulated)
// - GGML_TYPE_I8
// - GGML_TYPE_I16
// - GGML_TYPE_I32
// - GGML_TYPE_BF16

#define MATRIX_ROW_PADDING                                                                         \
    512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define NOT_IMPLEMENTED()                                                                          \
    do {                                                                                           \
        GGML_ABORT("(%s:%d) %s not implemented\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);       \
    } while (false)

bool ggml_hsa_string_to_bool(std::string_view s) {
    return s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes" || s == "Yes" ||
           s == "YES";
}

const char * ggml_hsa_get_status_string(hsa_status_t status) {
    const char * msg = nullptr;
    if (hsa_status_string(status, &msg) != HSA_STATUS_SUCCESS) {
        return "unknown";
    }
    return msg;
}

[[noreturn]]
void ggml_hsa_error(
    const char * stmt, const char * func, const char * file, int line, hsa_status_t status) {
    const char * msg = ggml_hsa_get_status_string(status);
    GGML_LOG_ERROR("HSA error: %s\n", msg);
    GGML_LOG_ERROR("  in function %s at %s:%d\n", func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT("HSA error");
}

std::int64_t ggml_hsa_nsrcs(const ggml_tensor * tensor) {
    std::int64_t nsrcs = 0;
    for (; (nsrcs < GGML_MAX_SRC) && (tensor->src[nsrcs] != nullptr); ++nsrcs)
        ;
    return nsrcs;
}

static const std::filesystem::path library_dir = [] {
    // retrieve the shared library path
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&ggml_hsa_library_path), &info) == 0) {
        GGML_ABORT("Could not retrieve library directory\n");
    }
    return std::filesystem::path{info.dli_fname}.parent_path();
}();

const std::filesystem::path & ggml_hsa_library_path() { return library_dir; }

/**
 * @brief Creates a device name from the device index @p device.
 */
static std::string ggml_hsa_format_name(std::int32_t device) {
    return GGML_HSA_NAME + std::to_string(device);
}

/**
 * @brief Retrieves the agent info for the given agent @p agent.
 */
static std::string ggml_hsa_agent_name(hsa_agent_t agent) {
    constexpr std::size_t agent_name_size = 64;
    char agent_name[agent_name_size];
    HSA_CHECK_THROW(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &agent_name));
    return std::string{agent_name};
}

// Returns the minimum queue size
static std::uint32_t ggml_hsa_get_agent_min_queue_size(hsa_agent_t agent) {
    std::uint32_t min_queue_size = 0;
    HSA_CHECK_THROW(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size));
    return min_queue_size;
}

/**
 * @brief Populates the information in @p info from @p pool.
 */
static hsa_status_t ggml_hsa_set_memory_pool_info(hsa_amd_memory_pool_t pool,
                                                  ggml_hsa_device_info::memory_pool_info & info) {
    bool alloc_allowed = true;
    if (auto status = hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
        (status != HSA_STATUS_SUCCESS) || !alloc_allowed) {
        // ignore pools that we can't allocate from
        return status;
    }

    std::size_t size = 0;
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &size);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    std::size_t alignment = 0;
    if (auto status = hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &alignment);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    std::size_t max_alloc_size = 0;
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                                   &max_alloc_size);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    info.memory_pool = pool;
    info.size = size;
    info.alignment = alignment;
    info.max_alloc_size = max_alloc_size;

    return HSA_STATUS_SUCCESS;
}

/**
 * @brief Discovers HSA memory pools.
 */
static hsa_status_t ggml_hsa_find_hsa_memory_pools(hsa_amd_memory_pool_t pool, void * data) {
    // query only global segments
    hsa_amd_segment_t segment_type = {};
    if (auto status =
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
        (status != HSA_STATUS_SUCCESS) || (segment_type != HSA_AMD_SEGMENT_GLOBAL)) {
        return status;
    }

    hsa_amd_memory_pool_global_flag_t pool_flags = {};
    if (auto status =
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flags);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    auto & dev_info = *static_cast<ggml_hsa_device_info::device_info *>(data);
    const bool kernarg_pool = (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0x0;
    if (kernarg_pool) {
        return ggml_hsa_set_memory_pool_info(pool, dev_info.kernarg_memory);
    }

    std::size_t max_alloc_size;
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                                   &max_alloc_size);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    const bool coarse_grained_pool =
        (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0x0;
    if (coarse_grained_pool) {
        if (max_alloc_size == 0) {
            // XDNA dev heap has max_alloc_size == 0
            return ggml_hsa_set_memory_pool_info(pool, dev_info.dev_memory);
        } else {
            return ggml_hsa_set_memory_pool_info(pool, dev_info.data_memory);
        }
    }

    return HSA_STATUS_SUCCESS;
}

/**
 * @brief Discovers HSA agents.
 */
static hsa_status_t ggml_hsa_find_hsa_agents(hsa_agent_t agent, void * data) {
    hsa_device_type_t type = {};
    if (auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }
    if (type != HSA_DEVICE_TYPE_AIE) {
        // only consider NPUs for now
        return HSA_STATUS_SUCCESS;
    }

    auto & info = *static_cast<ggml_hsa_device_info *>(data);
    if (info.device_count == GGML_HSA_MAX_DEVICES - 1) {
        GGML_ABORT("%s: Exceeded GGML_HSA_MAX_DEVICES limit (%d)", __func__, GGML_HSA_MAX_DEVICES);
    }

    // create device information (agent, type, name, memory pools, etc.)
    auto & dev_info = info.devices[info.device_count];
    dev_info.agent = agent;
    dev_info.type = type;

    char name[64] = {};
    if (auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }
    dev_info.name = std::string(name);

    if (auto status =
            hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_hsa_memory_pools, &dev_info);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    // add device to known devices
    ++info.device_count;

    return HSA_STATUS_SUCCESS;
}

/**
 * @brief Initialize HSA device information.
 *
 * This function initializes HSA and retrieves all the appropriate agents and
 * memory pools.
 */
static ggml_hsa_device_info ggml_hsa_init() {
    HSA_CHECK_THROW(hsa_init());

    ggml_hsa_device_info info = {};
    HSA_CHECK_THROW(hsa_iterate_agents(ggml_hsa_find_hsa_agents, &info));

    return info;
}

const ggml_hsa_device_info & ggml_hsa_info() {
    static ggml_hsa_device_info info = ggml_hsa_init();
    return info;
}

ggml_backend_hsa_context::ggml_backend_hsa_context(
    std::int32_t device, const ggml_hsa_device_info::device_info & dev_info) :
    device(device), name(ggml_hsa_format_name(device)) {
    hsa_agent_t agent = dev_info.agent;

    // create queue
    const std::uint32_t min_queue_size = ggml_hsa_get_agent_min_queue_size(agent);
    if (auto status = hsa_queue_create(agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                       nullptr, 0, 0, &queue);
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_queue_create failed: %s", __func__,
                       ggml_hsa_get_status_string(status));
        throw std::runtime_error("hsa_queue_create failed");
    }

    // create signal to wait for packets
    if (auto status = hsa_signal_create(0, 0, nullptr, &dispatch_signal);
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: hsa_signal_create failed: %s", __func__,
                       ggml_hsa_get_status_string(status));
        throw std::runtime_error("hsa_signal_create failed");
    }

#ifdef GGML_HSA_CPU_FALLBACK
    // create fallback backend
    if (fallback_backend = ggml_backend_cpu_init(); fallback_backend == nullptr) {
        throw std::runtime_error("ggml_backend_cpu_init failed");
    }
    auto buft = ggml_backend_get_default_buffer_type(fallback_backend);
    if (fallback_galloc = ggml_gallocr_new(buft); fallback_galloc == nullptr) {
        throw std::runtime_error("ggml_gallocr_new failed");
    }
#endif
}

ggml_backend_hsa_context::~ggml_backend_hsa_context() {
    destroy_kernels();
    HSA_CHECK_ABORT(hsa_signal_destroy(dispatch_signal));
    HSA_CHECK_ABORT(hsa_queue_destroy(queue));
#ifdef GGML_HSA_CPU_FALLBACK
    ggml_gallocr_free(fallback_galloc);
    ggml_backend_free(fallback_backend);
#endif
}

void ggml_backend_hsa_context::destroy_kernels() {
    for (auto & t : aie_kernels) {
        ggml_hsa_destroy_aie_kernel(*this, t.second);
    }
    aie_kernels.clear();
}

void ggml_backend_hsa_context::free_pending_payloads() {
    for (auto ptr : pending_payloads) {
        hsa_amd_memory_pool_free(ptr);
    }
    pending_payloads.clear();
}

/**
 * @brief Dispatches an HSA packet.
 *
 * @note This function assumes ownership of @p payload.
 *
 * @param[in] ctx backend context
 * @param[in] payload packet payload
 * @param[in] payload_size payload size in dwords
 */
static void ggml_hsa_dispatch_packet(ggml_backend_hsa_context & ctx,
                                     hsa_amd_aie_ert_start_kernel_data_t * payload,
                                     std::size_t payload_size) {
    auto * queue = ctx.queue;

    hsa_amd_aie_ert_packet_t pkt{};
    pkt.header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    pkt.header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
    pkt.state = HSA_AMD_AIE_ERT_STATE_NEW;
    pkt.count = payload_size;
    pkt.opcode = HSA_AMD_AIE_ERT_START_CU;
    pkt.payload_data = reinterpret_cast<std::uint64_t>(payload);
    // TODO add pkt->completion_signal = ctx.dispatch_signal

    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);
    const std::uint64_t packet_id = wr_idx % queue->size;
    *(static_cast<hsa_amd_aie_ert_packet_t *>(queue->base_address) + packet_id) = pkt;

    ctx.pending_payloads.push_back(payload);

    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
}

ggml_status ggml_hsa_dispatch_kernel(ggml_backend_hsa_context & ctx, ggml_tensor * tensor) {
    auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(tensor->extra);
    const auto & kernel = tensor_extra.kernel;
    auto & info = ggml_hsa_info();
    auto & dev_info = info.devices[ctx.device];
    const auto nsrcs = kernel.num_src_tensors;
    const std::size_t packet_dwords =
        3 /* instructions */ + (nsrcs + 1) * 3 /* source and destination tensors */;
    hsa_amd_aie_ert_start_kernel_data_t * cmd_payload = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.kernarg_memory.memory_pool, 64, 0,
                                                   reinterpret_cast<void **>(&cmd_payload));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate space for the packet (%d)\n", __func__, status);
        return GGML_STATUS_ALLOC_FAILED;
    }
    cmd_payload->pdi_addr = kernel.pdi.data; // PDI to use with this command

    // transaction opcode; not counted in packet_dwords
    cmd_payload->data[0] = 0x3;
    cmd_payload->data[1] = 0x0;

    std::size_t dword_idx = 2;

    // instructions; 3 dwords
    std::tie(cmd_payload->data[dword_idx + 1], cmd_payload->data[dword_idx]) =
        ggml_hsa_addr_to_hilo(kernel.insts.data);
    cmd_payload->data[dword_idx + 2] = static_cast<std::uint32_t>(kernel.insts.size);

    dword_idx += 3;

    // sources; 2 dwords each
    for (std::size_t src_idx = 0; src_idx < nsrcs; ++src_idx, dword_idx += 2) {
        const ggml_tensor * src = tensor->src[src_idx];
        std::tie(cmd_payload->data[dword_idx + 1], cmd_payload->data[dword_idx]) =
            ggml_hsa_addr_to_hilo(src->data);
    }

    // destination; 2 dwords
    std::tie(cmd_payload->data[dword_idx + 1], cmd_payload->data[dword_idx]) =
        ggml_hsa_addr_to_hilo(tensor->data);

    dword_idx += 2;

    // sizes; 1 dword per tensor
    for (std::size_t src_idx = 0; src_idx < nsrcs; ++src_idx, ++dword_idx) {
        const ggml_tensor * src = tensor->src[src_idx];
        cmd_payload->data[dword_idx] = ggml_nbytes(src);
    }
    cmd_payload->data[dword_idx] = ggml_nbytes(tensor);

    assert(dword_idx == packet_dwords + 1); // 2 extra uncounted dwords

    ggml_hsa_dispatch_packet(ctx, cmd_payload, packet_dwords);

    return GGML_STATUS_SUCCESS;
}

// HSA buffer

/**
 * @brief Context for managing a HSA buffer associated with a specific device.
 */
struct ggml_backend_hsa_buffer_context {
    std::int32_t device{}; ///< Device ID associated with this buffer context.
    void * dev_ptr{};      ///< Pointer to the device memory.
    std::vector<std::unique_ptr<ggml_backend_hsa_tensor_extra>> tensor_extras;

    ggml_backend_hsa_buffer_context(std::int32_t device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr) {}

    ggml_backend_hsa_buffer_context(const ggml_backend_hsa_buffer_context &) = delete;
    ggml_backend_hsa_buffer_context(ggml_backend_hsa_buffer_context &&) = delete;

    ~ggml_backend_hsa_buffer_context() { HSA_CHECK_ABORT(hsa_amd_memory_pool_free(dev_ptr)); }

    ggml_backend_hsa_buffer_context & operator=(const ggml_backend_hsa_buffer_context &) = delete;
    ggml_backend_hsa_buffer_context & operator=(ggml_backend_hsa_buffer_context &&) = delete;
};

/**
 * @brief Frees resources associated with @p buffer.
 */
static void ggml_backend_hsa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * buf_ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    delete buf_ctx;
}

/**
 * @brief Returns if @p buffer is a HSA buffer.
 */
static bool ggml_backend_buffer_is_hsa(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_hsa_buffer_free_buffer;
}

/**
 * @brief Returns the base pointer of @p buffer.
 */
static void * ggml_backend_hsa_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto & buf_ctx = *static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    return buf_ctx.dev_ptr;
}

/**
 * @brief Initializes the tensor.
 */
static enum ggml_status ggml_backend_hsa_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor * tensor) try {
    auto & buf_ctx = *static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    assert(tensor->extra == nullptr);
    buf_ctx.tensor_extras.push_back(std::make_unique<ggml_backend_hsa_tensor_extra>());
    tensor->extra = buf_ctx.tensor_extras.back().get();

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) &&
        (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        // initialize padding to 0 to avoid possible NaN values
        const std::size_t original_size = ggml_nbytes(tensor);
        const std::size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            std::memset(static_cast<std::byte *>(tensor->data) + original_size, 0,
                        (padded_size - original_size));
        }
    }

    return GGML_STATUS_SUCCESS;
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not create tensor (%s)\n", __func__, ex.what());
    return GGML_STATUS_FAILED;
}

/**
 * @brief Set tensor data to a specific value @p value.
 *
 * @param buffer tensor storage
 * @param tensor destination tensor
 * @param value value to set to the tensor
 * @param offset offset in tensor
 * @param size size of data to set, in bytes
 */
static void ggml_backend_hsa_buffer_memset_tensor(ggml_backend_buffer_t /* buffer */,
                                                  ggml_tensor * tensor,
                                                  uint8_t value,
                                                  size_t offset,
                                                  size_t size) {
    assert(ggml_is_contiguous(tensor) && "Only contiguous tensors supported");
    std::memset(static_cast<std::byte *>(tensor->data) + offset, value, size);
}

/**
 * @brief Set tensor data.
 *
 * @param buffer tensor storage
 * @param tensor destination tensor
 * @param data source data
 * @param offset offset in source data
 * @param size size of source data, in bytes
 */
static void ggml_backend_hsa_buffer_set_tensor(ggml_backend_buffer_t /* buffer */,
                                               ggml_tensor * tensor,
                                               const void * data,
                                               size_t offset,
                                               size_t size) {
    assert(ggml_is_contiguous(tensor) && "Only contiguous tensors supported");
    std::memcpy(static_cast<std::byte *>(tensor->data) + offset, data, size);
}

/**
 * @brief Get tensor data.
 *
 * @param buffer tensor storage
 * @param tensor source tensor
 * @param data pointer to destination buffer
 * @param offset offset in source tensor data
 * @param size size of source data, in bytes
 */
static void ggml_backend_hsa_buffer_get_tensor(ggml_backend_buffer_t /* buffer */,
                                               const ggml_tensor * tensor,
                                               void * data,
                                               size_t offset,
                                               size_t size) {
    assert(ggml_is_contiguous(tensor) && "Only contiguous tensors supported");
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

/**
 * @brief Copy tensor data between buffers if possible.
 *
 * The size of the data to be copied is inferred by the source tensor @p src.
 *
 * @param buffer tensor storage
 * @param src source tensor
 * @param dst destination tensor
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_hsa_buffer_cpy_tensor(ggml_backend_buffer_t /* buffer */,
                                               const ggml_tensor * src,
                                               ggml_tensor * dst) {
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false; // only contiguous tensors supported
    }
    if (ggml_backend_buffer_is_hsa(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(dst));
        return true;
    }
    return false;
}

/**
 * @brief Clear buffer @p buffer by setting all its memory to @p value.
 */
static void ggml_backend_hsa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto & buf_ctx = *static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    std::memset(buf_ctx.dev_ptr, value, buffer->size);
}

/**
 * @brief Interface for HSA buffers.
 */
static const ggml_backend_buffer_i ggml_backend_hsa_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_hsa_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hsa_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_hsa_buffer_init_tensor,
    /* .memset_tensor = */ ggml_backend_hsa_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hsa_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hsa_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_hsa_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hsa_buffer_clear,
    /* .reset         = */ nullptr,
};

// HSA buffer type

/**
 * @brief Context information for HSA backend buffer type.
 */
struct ggml_backend_hsa_buffer_type_context {
    std::int32_t device; ///< ID of the device associated with this buffer type context.
    std::string name;    ///< Name of the buffer type context.

    explicit ggml_backend_hsa_buffer_type_context(std::int32_t device) :
        device(device), name(ggml_hsa_format_name(device)) {}
};

/**
 * @brief Returns the name associated with the buffer type @p buft.
 */
static const char * ggml_backend_hsa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    return buft_ctx.name.c_str();
}

/**
 * @brief Returns if the buffer type @p buft is a HSA buffer type.
 */
static bool ggml_backend_buft_is_hsa(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hsa_buffer_type_get_name;
}

/**
 * @brief Allocates a buffer in @p buft of size @p size.
 */
static ggml_backend_buffer_t
ggml_backend_hsa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) try {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[buft_ctx.device];

    void * buffer = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.data_memory.memory_pool, size,
                                                   /* flags = */ 0, &buffer);
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR(
            "%s: allocating %.2f MiB on device %d: hsa_amd_memory_pool_allocate failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx.device, ggml_hsa_get_status_string(status));
        return nullptr;
    }

    auto * buf_ctx = new ggml_backend_hsa_buffer_context(buft_ctx.device, buffer);
    return ggml_backend_buffer_init(buft, ggml_backend_hsa_buffer_interface, buf_ctx, size);
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not allocate buffer (%s)\n", __func__, ex.what());
    return nullptr;
}

/**
 * @brief Returns the memory alignment requirement for buffer type @p buft in bytes.
 */
static size_t ggml_backend_hsa_buffer_type_get_alignment(ggml_backend_buffer_type_t /* buft */) {
    // TODO: verify if 256bytes is the best alignment for all agents (GPU, AIE)
    return 256;
}

/**
 * @brief Returns the maximum allocation size for buffer type @p buft in bytes.
 */
static size_t ggml_backend_hsa_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[buft_ctx.device];
    return dev_info.data_memory.max_alloc_size;
}

/**
 * @brief Returns the size required for tensor @p tensor in buffer type @p buft.
 */
static size_t ggml_backend_hsa_buffer_type_get_alloc_size(ggml_backend_buffer_type_t /* buft */,
                                                          const ggml_tensor * tensor) {
    std::size_t size = ggml_nbytes(tensor);

    if (ggml_is_quantized(tensor->type)) {
        const auto ne0 = tensor->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

/**
 * @brief Returns if buffer type @p buft is a host buffer type.
 */
static bool ggml_backend_hsa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[buft_ctx.device];

    // we can infer if it is host memory from the agent type since the memory pools are
    // derived from the agent
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_CPU:
        case HSA_DEVICE_TYPE_AIE:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Interface for managing HSA buffer types.
 */
static const ggml_backend_buffer_type_i ggml_backend_hsa_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_hsa_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hsa_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hsa_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hsa_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_backend_hsa_buffer_type_get_alloc_size,
    /* .is_host        = */ ggml_backend_hsa_buffer_type_is_host,
};

/**
 * @brief HSA buffer types.
 */
static struct {
    ggml_backend_buffer_type type[GGML_HSA_MAX_DEVICES];
    std::once_flag flag;
} ggml_backend_hsa_buffer_type_metadata;

ggml_backend_buffer_type_t ggml_backend_hsa_buffer_type(int device) try {
    const auto device_count = ggml_backend_hsa_get_device_count();

    if (device >= device_count) {
        return nullptr;
    }

    std::call_once(ggml_backend_hsa_buffer_type_metadata.flag, [&device_count] {
        for (std::int32_t i = 0; i < device_count; ++i) {
            ggml_backend_hsa_buffer_type_metadata.type[i] = {
                /* .iface   = */ ggml_backend_hsa_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), i),
                /* .context = */ new ggml_backend_hsa_buffer_type_context{i},
            };
        }
    });

    return &ggml_backend_hsa_buffer_type_metadata.type[device];
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not create buffer type (%s)\n", __func__, ex.what());
    return nullptr;
}

// HSA split buffer

// TODO

// HSA split buffer type

/**
 * @brief Returns if @p buft is a split buffer.
 */
static bool ggml_backend_buft_is_hsa_split(ggml_backend_buffer_type_t /* buft */) { return false; }

// host buffer type

static const char * ggml_backend_hsa_host_buffer_type_name(ggml_backend_buffer_type_t /* buft */) {
    return GGML_HSA_NAME "_Host";
}

static void ggml_backend_hsa_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    // TODO free buffer
    NOT_IMPLEMENTED();
}

static void * ggml_hsa_host_malloc(size_t size) {
    // TODO allocate pinned memory
    NOT_IMPLEMENTED();
    return nullptr;
}

static ggml_backend_buffer_t
ggml_backend_hsa_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_hsa_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    auto buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_hsa_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_hsa_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_hsa_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name       = */ ggml_backend_hsa_host_buffer_type_name,
            /* .alloc_buffer   = */ ggml_backend_hsa_host_buffer_type_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size   = */ nullptr, // defaults to SIZE_MAX
            /* .get_alloc_size = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host        = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_hsa_buffer_type_host;
}

////////////////////////////////////////////////////////////////////////////////

// backend

/**
 * @brief Returns the name of backend @p backend.
 */
static const char * ggml_backend_hsa_get_name(ggml_backend_t backend) {
    const auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    return ctx.name.c_str();
}

/**
 * @brief Frees the resources associated with @p backend.
 */
static void ggml_backend_hsa_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_hsa_context *>(backend->context);
    delete ctx;
    delete backend;
}

/**
 * @brief Returns the buffer type of the buffer of tensor @p tensor.
 */
static ggml_backend_buffer_type_t ggml_backend_hsa_get_tensor_buft(const ggml_tensor * tensor) {
    return (tensor->view_src ? tensor->view_src->buffer : tensor->buffer)->buft;
}

/**
 * @brief Set tensor data asynchronously.
 *
 * @param backend backend
 * @param tensor destination tensor
 * @param data source data
 * @param offset offset in source data
 * @param size size of source data, in bytes
 */
static void ggml_backend_hsa_set_tensor_async(
    ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT((ggml_backend_hsa_get_tensor_buft(tensor) ==
                 ggml_backend_dev_buffer_type(backend->device)) &&
                "unsupported buffer type");
    assert(ggml_is_contiguous(tensor) && "Only contiguous tensors supported");
    std::memcpy(static_cast<std::byte *>(tensor->data) + offset, data, size);
    GGML_UNUSED(backend);
}

/**
 * @brief Get tensor data asynchronously.
 *
 * @param backend backend
 * @param tensor source tensor
 * @param data pointer to destination buffer
 * @param offset offset in source tensor data
 * @param size size of source data, in bytes
 */
static void ggml_backend_hsa_get_tensor_async(
    ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT((ggml_backend_hsa_get_tensor_buft(tensor) ==
                 ggml_backend_dev_buffer_type(backend->device)) &&
                "unsupported buffer type");
    assert(ggml_is_contiguous(tensor) && "Only contiguous tensors supported");
    std::memcpy(data, static_cast<std::byte *>(tensor->data) + offset, size);
    GGML_UNUSED(backend);
}

/**
 * @brief Copy tensor data between buffers if possible.
 *
 * The size of the data to be copied is inferred by the source tensor @p src.
 *
 * @param backend_src source backend
 * @param backend_dst destination backend
 * @param src source tensor
 * @param dst destination tensor
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_hsa_cpy_tensor_async(ggml_backend_t backend_src,
                                              ggml_backend_t backend_dst,
                                              const ggml_tensor * src,
                                              ggml_tensor * dst) {
    if (!ggml_backend_is_hsa(backend_src) || !ggml_backend_is_hsa(backend_dst)) {
        return false;
    }
    if (!ggml_backend_buffer_is_hsa(src->buffer) || !ggml_backend_buffer_is_hsa(dst->buffer)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false; // only contiguous tensors supported
    }
    std::memcpy(dst->data, src->data, ggml_nbytes(dst));
    return true;
}

static void ggml_backend_hsa_synchronize(ggml_backend_t backend) {
    auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    if (auto val = hsa_signal_wait_scacquire(ctx.dispatch_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                             UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        val != 0) {
        GGML_ABORT("%s: error: unexpected signal value (%ld)\n", __func__, val);
    }

    ctx.free_pending_payloads();
}

#ifdef GGML_HSA_CPU_FALLBACK

struct ggml_backend_hsa_emulated_tensor {
    ggml_tensor * tensor{};
    ggml_context * ggml_ctx{};
    ggml_cgraph * ggml_graph{};

    ggml_backend_hsa_emulated_tensor(ggml_backend_hsa_context & ctx, ggml_tensor * t) : tensor{t} {
        // create tensor allocator
        std::size_t buffer_size = ggml_nbytes_pad(tensor) + 4096;
        std::int32_t tensor_src_count = 0;
        for (std::int32_t i = 0; i < GGML_MAX_SRC; ++i) {
            if (tensor->src[i] == nullptr) {
                continue;
            }
            ++tensor_src_count;
            buffer_size += ggml_nbytes_pad(tensor->src[i]);
        }
        auto buffer = ggml_backend_alloc_buffer(ctx.fallback_backend, buffer_size);
        auto talloc = ggml_tallocr_new(buffer);

        // create context
        const auto tensor_count = tensor_src_count + 1;
        const ggml_init_params params = {
            /*.mem_size   =*/(tensor_count * ggml_tensor_overhead() +
                              ggml_graph_overhead_custom(tensor_count, false) + 8 * 4096),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_ctx = ggml_init(params);
        if (ggml_ctx == nullptr) {
            throw std::runtime_error("Failed to initialize context");
        }

        // create tensor
        auto new_tensor = ggml_dup_tensor(ggml_ctx, tensor);
        GGML_ASSERT(ggml_are_same_shape(tensor, new_tensor));
        GGML_ASSERT(ggml_are_same_stride(tensor, new_tensor));
        new_tensor->op = tensor->op;
        std::copy_n(tensor->op_params, GGML_MAX_OP_PARAMS / sizeof(int32_t), new_tensor->op_params);
        if (ggml_tallocr_alloc(&talloc, new_tensor) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to allocate tensor");
        }
        ggml_set_name(new_tensor, tensor->name);

        // create source tensors
        for (std::int32_t i = 0; i < GGML_MAX_SRC; ++i) {
            auto * src = tensor->src[i];
            if (src == nullptr) {
                continue;
            }
            const auto tensor_src_dims = ggml_n_dims(src);
            auto * new_src = ggml_new_tensor(ggml_ctx, src->type, tensor_src_dims, src->ne);
            GGML_ASSERT(ggml_are_same_shape(src, new_src));
            ggml_set_name(new_src, src->name);

            new_tensor->src[i] = new_src;
        }

        // create graph
        ggml_graph = ggml_new_graph_custom(ggml_ctx, tensor_count, false);
        ggml_build_forward_expand(ggml_graph, new_tensor);

        if (!ggml_gallocr_alloc_graph(ctx.fallback_galloc, ggml_graph)) {
            throw std::runtime_error("Failed to allocate graph");
        }
    }

    ggml_backend_hsa_emulated_tensor(const ggml_backend_hsa_emulated_tensor &) = delete;
    ggml_backend_hsa_emulated_tensor(ggml_backend_hsa_emulated_tensor &&) = delete;

    ~ggml_backend_hsa_emulated_tensor() { ggml_free(ggml_ctx); }

    ggml_backend_hsa_emulated_tensor & operator=(const ggml_backend_hsa_emulated_tensor &) = delete;
    ggml_backend_hsa_emulated_tensor & operator=(ggml_backend_hsa_emulated_tensor &&) = delete;

    ggml_status operator()() {
        auto new_tensor = ggml_graph_node(ggml_graph, 0);

        // copy input tensors
        for (std::int32_t i = 0; i < GGML_MAX_SRC; ++i) {
            if (tensor->src[i] == nullptr) {
                continue;
            }
            ggml_backend_tensor_copy(tensor->src[i], new_tensor->src[i]);
        }

        // execute
        const auto num_threads = 4;
        ggml_status status = GGML_STATUS_SUCCESS;
        if (status = ggml_graph_compute_with_ctx(ggml_ctx, ggml_graph, num_threads);
            status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("ggml_graph_compute_with_ctx(): failed to compute graph");
            return status;
        }

        // copy output tensor
        ggml_backend_tensor_copy(new_tensor, tensor);

        return GGML_STATUS_SUCCESS;
    }
};

#endif

static enum ggml_status ggml_backend_hsa_graph_compute(ggml_backend_t backend,
                                                       ggml_cgraph * cgraph) try {
    auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    ggml_status status = GGML_STATUS_SUCCESS;

    const int node_count = ggml_graph_n_nodes(cgraph);
    for (int i = 0; (i < node_count) && (status == GGML_STATUS_SUCCESS); ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);
        if (ggml_is_empty(node)) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
                // NOP, no kernel required
                break;

            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_VIEW:
                // implemented as views, so no kernel required
                break;

            default:
                {
                    auto & tensor_extra =
                        *static_cast<ggml_backend_hsa_tensor_extra *>(node->extra);
                    if (!tensor_extra.kernel.is_valid()) {
                        status = ggml_hsa_create_aie_kernel(ctx, node, tensor_extra.kernel);
                    }
                    if (status == GGML_STATUS_SUCCESS) {
                        status = ggml_hsa_dispatch_kernel(ctx, node);
                    }
                }
                break;
        }

#ifdef GGML_HSA_CPU_FALLBACK
        if (status != GGML_STATUS_SUCCESS) {
            auto tensor_extra = static_cast<ggml_backend_hsa_tensor_extra *>(node->extra);
            if (!tensor_extra->emulated_tensor) {
                GGML_LOG_INFO("%s: emulating op %s for \"%s\"\n", __func__, ggml_op_desc(node),
                              node->name);
                tensor_extra->emulated_tensor =
                    std::make_unique<ggml_backend_hsa_emulated_tensor>(ctx, node);
            }
            ggml_backend_hsa_synchronize(backend);
            status = (*tensor_extra->emulated_tensor)();
        }
#endif

        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: op %s failed or not supported for \"%s\"\n", __func__,
                           ggml_op_desc(node), node->name);
        }
    }

    return status;
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not execute graph (%s)\n", __func__, ex.what());
    return GGML_STATUS_FAILED;
}

static void ggml_backend_hsa_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

/**
 * @brief Interface for managing HSA backends.
 */
static const ggml_backend_i ggml_backend_hsa_interface = {
    /* .get_name           = */ ggml_backend_hsa_get_name,
    /* .free               = */ ggml_backend_hsa_free,
    /* .set_tensor_async   = */ ggml_backend_hsa_set_tensor_async,
    /* .get_tensor_async   = */ ggml_backend_hsa_get_tensor_async,
    /* .cpy_tensor_async   = */ ggml_backend_hsa_cpy_tensor_async,
    /* .synchronize        = */ ggml_backend_hsa_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hsa_graph_compute,
    /* .event_record       = */ ggml_backend_hsa_event_record,
    /* .event_wait         = */ ggml_backend_hsa_event_wait,
};

/**
 * @brief Returns the unique identifier of the HSA backend.
 *
 * @note The identifier is a UUID v4 that was randomly generated.
 */
static ggml_guid_t ggml_backend_hsa_guid() {
    static ggml_guid guid = {0xa2, 0xe9, 0xa0, 0x84, 0x2c, 0xf6, 0x4d, 0xa1,
                             0xb3, 0xb2, 0xb1, 0xdc, 0x5d, 0x59, 0x21, 0x95};
    return &guid;
}

/**
 * @brief Returns if @p backend is an HSA backend.
 */
bool ggml_backend_is_hsa(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hsa_guid());
}

/**
 * @brief Returns if the number of devices (i.e., HSA agents) associated with the HSA backend.
 */
int ggml_backend_hsa_get_device_count() { return ggml_hsa_info().device_count; }

/**
 * @brief Returns the device description of device @p device.
 */
void ggml_backend_hsa_get_device_description(int device,
                                             char * description,
                                             size_t description_size) {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[device];
    snprintf(description, description_size, "%s", dev_info.name.data());
}

/**
 * @brief Returns the free and total memory in @p free and @p total respectively for device
 *        @p dev.
 */
void ggml_backend_hsa_get_device_memory(int device, size_t * free, size_t * total) {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[device];
    *total = dev_info.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

bool ggml_backend_hsa_register_host_buffer(void * buffer, size_t size) {
    NOT_IMPLEMENTED();
    return false;
}

void ggml_backend_hsa_unregister_host_buffer(void * buffer) { NOT_IMPLEMENTED(); }

// backend device

/**
 * @brief HSA device context.
 */
struct ggml_backend_hsa_device_context {
    std::int32_t device;
    std::string name;
    std::string description;

    ggml_backend_hsa_device_context(std::int32_t device, hsa_agent_t agent) :
        device(device),
        name(ggml_hsa_format_name(device)),
        description(ggml_hsa_agent_name(agent)) {}
};

/**
 * @brief Returns the device info associated with @p dev.
 */
static const ggml_hsa_device_info::device_info & ggml_hsa_get_device_info(ggml_backend_dev_t dev) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[dev_ctx.device];
    return dev_info;
}

static const char * ggml_backend_hsa_device_get_name(ggml_backend_dev_t dev) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return dev_ctx.name.c_str();
}

static const char * ggml_backend_hsa_device_get_description(ggml_backend_dev_t dev) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return dev_ctx.description.c_str();
}

/**
 * @brief Returns the free and total memory in @p free and @p total respectively for device
 *        @p dev.
 */
static void
ggml_backend_hsa_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const auto & dev_info = ggml_hsa_get_device_info(dev);
    *total = dev_info.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

/**
 * @brief Returns the device type of @p dev.
 */
static enum ggml_backend_dev_type ggml_backend_hsa_device_get_type(ggml_backend_dev_t dev) {
    const auto & dev_info = ggml_hsa_get_device_info(dev);
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_CPU:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case HSA_DEVICE_TYPE_GPU:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case HSA_DEVICE_TYPE_DSP:
        case HSA_DEVICE_TYPE_AIE:
            return GGML_BACKEND_DEVICE_TYPE_ACCEL;
        default:
            GGML_ABORT("%s: error: unknown HSA device type %d", __func__, dev_info.type);
    }
}

static void ggml_backend_hsa_device_get_props(ggml_backend_dev_t dev,
                                              ggml_backend_dev_props * props) {
    props->name = ggml_backend_hsa_device_get_name(dev);
    props->description = ggml_backend_hsa_device_get_description(dev);
    props->type = ggml_backend_hsa_device_get_type(dev);
    ggml_backend_hsa_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_hsa_device_init_backend(ggml_backend_dev_t dev,
                                                           const char * /*params*/) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ggml_backend_hsa_init(dev_ctx.device);
}

static ggml_backend_buffer_type_t ggml_backend_hsa_device_get_buffer_type(ggml_backend_dev_t dev) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ggml_backend_hsa_buffer_type(dev_ctx.device);
}

static ggml_backend_buffer_type_t
ggml_backend_hsa_device_get_host_buffer_type(ggml_backend_dev_t /*dev*/) {
    return ggml_backend_hsa_host_buffer_type();
}

/**
 * @brief Returns if the operation in tensor @p op is supported by device @p dev.
 */
static bool ggml_backend_hsa_device_supports_op(ggml_backend_dev_t dev,
                                                const ggml_tensor * tensor) try {
    bool supported = false;
    switch (tensor->op) {
        case GGML_OP_NONE:
            // NOP, no kernel required
            supported = true;
            break;

        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            // implemented as views, so no kernel required
            supported = true;
            break;

        default:
            // check if the kernel is cached at the tensor level and if not, check if the kernel
            // files exist
            if (auto tensor_extra =
                    static_cast<const ggml_backend_hsa_tensor_extra *>(tensor->extra);
                (tensor->extra != nullptr) && tensor_extra->kernel.is_valid()) {
                supported = true;
            } else {
                supported = ggml_hsa_kernel_exists(ggml_hsa_get_device_info(dev), tensor);
            }
            break;
    }

#ifdef GGML_HSA_CPU_FALLBACK
    if (!supported) {
        auto cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        supported = ggml_backend_dev_supports_op(cpu_dev, tensor);
    }
#endif

    return supported;
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not check operation (%s)\n", __func__, ex.what());
    return false;
}

static bool ggml_backend_hsa_device_supports_buft(ggml_backend_dev_t dev,
                                                  ggml_backend_buffer_type_t buft) {
    return (ggml_backend_buft_is_hsa(buft) || ggml_backend_buft_is_hsa_split(buft)) &&
           buft->device == dev;
}

static std::int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_hsa_device_offload_op(ggml_backend_dev_t /* dev */,
                                               const ggml_tensor * op) {
    const std::int64_t min_batch_size = 32;
    return get_op_batch_size(op) >= min_batch_size;
}

static ggml_backend_event_t ggml_backend_hsa_device_event_new(ggml_backend_dev_t dev) {
    NOT_IMPLEMENTED();
    return nullptr;
}

static void ggml_backend_hsa_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_device_event_synchronize(ggml_backend_dev_t dev,
                                                      ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

/**
 * @brief Interface for managing HSA devices.
 */
static const ggml_backend_device_i ggml_backend_hsa_device_interface = {
    /* .get_name             = */ ggml_backend_hsa_device_get_name,
    /* .get_description      = */ ggml_backend_hsa_device_get_description,
    /* .get_memory           = */ ggml_backend_hsa_device_get_memory,
    /* .get_type             = */ ggml_backend_hsa_device_get_type,
    /* .get_props            = */ ggml_backend_hsa_device_get_props,
    /* .init_backend         = */ ggml_backend_hsa_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hsa_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_hsa_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hsa_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hsa_device_supports_buft,
    /* .offload_op           = */ ggml_backend_hsa_device_offload_op,
    /* .event_new            = */ ggml_backend_hsa_device_event_new,
    /* .event_free           = */ ggml_backend_hsa_device_event_free,
    /* .event_synchronize    = */ ggml_backend_hsa_device_event_synchronize,
};

// backend reg

/**
 * @brief HSA registration context.
 */
struct ggml_backend_hsa_reg_context {
    static inline const char * name = GGML_HSA_NAME;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<ggml_backend_feature> features;
};

static const char * ggml_backend_hsa_reg_get_name(ggml_backend_reg_t /* reg */) {
    return ggml_backend_hsa_reg_context::name;
}

static size_t ggml_backend_hsa_reg_get_device_count(ggml_backend_reg_t reg) {
    const auto & reg_ctx = *static_cast<ggml_backend_hsa_reg_context *>(reg->context);
    return reg_ctx.devices.size();
}

static ggml_backend_dev_t ggml_backend_hsa_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    const auto & reg_ctx = *static_cast<ggml_backend_hsa_reg_context *>(reg->context);
    GGML_ASSERT(index < reg_ctx.devices.size());
    return reg_ctx.devices[index];
}

static ggml_backend_feature * ggml_backend_hsa_get_features(ggml_backend_reg_t reg) {
    auto & reg_ctx = *static_cast<ggml_backend_hsa_reg_context *>(reg->context);
    return reg_ctx.features.data();
}

static void * ggml_backend_hsa_reg_get_proc_address(ggml_backend_reg_t /* reg */,
                                                    const char * name) {
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hsa_register_host_buffer);
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hsa_unregister_host_buffer);
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return reinterpret_cast<void *>(ggml_backend_hsa_get_features);
    }
    return nullptr;
}

/**
 * @brief Interface for managing HSA registration.
 */
static const ggml_backend_reg_i ggml_backend_hsa_reg_interface = {
    /* .get_name          = */ ggml_backend_hsa_reg_get_name,
    /* .get_device_count  = */ ggml_backend_hsa_reg_get_device_count,
    /* .get_device        = */ ggml_backend_hsa_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_hsa_reg_get_proc_address,
};

// backend registry

static struct {
    ggml_backend_reg reg;
    std::once_flag flag;
} ggml_backend_hsa_reg_metadata;

ggml_backend_reg_t ggml_backend_hsa_reg() try {
    std::call_once(ggml_backend_hsa_reg_metadata.flag, [] {
        const auto & info = ggml_hsa_info();

        auto * reg_ctx = new ggml_backend_hsa_reg_context;

        reg_ctx->devices.reserve(info.device_count);
        for (std::int32_t i = 0; i < info.device_count; i++) {
            auto * dev_ctx = new ggml_backend_hsa_device_context{i, info.devices[i].agent};

            auto dev = new ggml_backend_device{/* .iface   = */ ggml_backend_hsa_device_interface,
                                               /* .reg     = */ &ggml_backend_hsa_reg_metadata.reg,
                                               /* .context = */ dev_ctx};
            reg_ctx->devices.push_back(dev);
        }

        ggml_backend_hsa_reg_metadata.reg =
            ggml_backend_reg{/* .api_version = */ GGML_BACKEND_API_VERSION,
                             /* .iface       = */ ggml_backend_hsa_reg_interface,
                             /* .context     = */ reg_ctx};
    });

    return &ggml_backend_hsa_reg_metadata.reg;
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not register backend (%s)\n", __func__, ex.what());
    return nullptr;
}

ggml_backend_t ggml_backend_hsa_init(int device) try {
    const auto & info = ggml_hsa_info();

    if (device < 0 || device >= info.device_count) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    auto * ctx = new ggml_backend_hsa_context{device, info.devices[device]};

    ggml_backend_t hsa_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_hsa_guid(),
        /* .interface = */ ggml_backend_hsa_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), device),
        /* .context   = */ ctx,
    };

    return hsa_backend;
} catch (const std::exception & ex) {
    GGML_LOG_ERROR("%s: Could not initialize backend (%s)\n", __func__, ex.what());
    return nullptr;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hsa_reg)
