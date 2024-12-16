#include "ggml-hsa.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-hsa/common.hpp"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// The following data types are natively supported by AIEs:
// - GGML_TYPE_F32 (emulated)
// - GGML_TYPE_I8
// - GGML_TYPE_I16
// - GGML_TYPE_I32
// - GGML_TYPE_BF16

#define NOT_IMPLEMENTED() \
    do { \
        printf("(%s:%d) %s not implemented\n", __FILE__, __LINE__, __PRETTY_FUNCTION__); \
        abort(); \
    } while (false)

/**
 * @brief Returns the status description of @p status.
 */
static const char* ggml_hsa_get_status_string(hsa_status_t status) {
    const char* msg = nullptr;
    if (hsa_status_string(status, &msg) != HSA_STATUS_SUCCESS) {
        return "unknown";
    }
    return msg;
}

[[noreturn]]
void ggml_hsa_error(const char * stmt, const char * func, const char * file, int line, hsa_status_t status) {
    const char* msg = ggml_hsa_get_status_string(status);
    GGML_LOG_ERROR("HSA error: %s\n", msg);
    GGML_LOG_ERROR("  in function %s at %s:%d\n", func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT("HSA error");
}

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
    HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, &agent_name));
    return GGML_HSA_NAME + std::string{agent_name};
}

/**
 * @brief Populates the information in @p info from @p pool.
 */
static hsa_status_t ggml_hsa_set_memory_pool_info(
    hsa_amd_memory_pool_t pool,
    ggml_hsa_device_info::hsa_memory_pool_info & info) {
    bool alloc_allowed = true;
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
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
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &alignment);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    std::size_t max_alloc_size = 0;
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_alloc_size);
        status != HSA_STATUS_SUCCESS || (max_alloc_size == 0)) {
        // XDNA dev heap has max_alloc_size == 0
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
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
        (status != HSA_STATUS_SUCCESS) || (segment_type != HSA_AMD_SEGMENT_GLOBAL)) {
        return status;
    }

    hsa_amd_memory_pool_global_flag_t pool_flags = {};
    if (auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flags);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }

    auto & device_info = *static_cast<ggml_hsa_device_info::hsa_device_info *>(data);
    const bool kernarg_pool = (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) != 0x0;
    if (kernarg_pool) {
        return ggml_hsa_set_memory_pool_info(pool, device_info.kernarg_memory);
    }

    const bool coarse_grained_pool = (pool_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) != 0x0;
    if (coarse_grained_pool) {
        return ggml_hsa_set_memory_pool_info(pool, device_info.data_memory);
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
    auto & device_info = info.devices[info.device_count];
    device_info.agent = agent;
    device_info.type = type;

    char name[64] = {};
    if (auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
       status != HSA_STATUS_SUCCESS) {
       return status;
    }
    device_info.name = std::string(name);

    if (auto status = hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_hsa_memory_pools, &device_info);
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
    HSA_CHECK(hsa_init());

    ggml_hsa_device_info info = {};
    HSA_CHECK(hsa_iterate_agents(ggml_hsa_find_hsa_agents, &info));

    return info;
}

const ggml_hsa_device_info & ggml_hsa_info() {
    static ggml_hsa_device_info info = ggml_hsa_init();
    return info;
}

ggml_backend_hsa_context::ggml_backend_hsa_context(std::int32_t device) :
        device(device), name(ggml_hsa_format_name(device)) {
}

// HSA buffer

/**
 * @brief Context for managing a HSA buffer associated with a specific device.
 */
struct ggml_backend_hsa_buffer_context {
    std::int32_t device;     ///< Device ID associated with this buffer context.
    void * dev_ptr{nullptr}; ///< Pointer to the device memory.

    ggml_backend_hsa_buffer_context(std::int32_t device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr) {
    }

    ~ggml_backend_hsa_buffer_context() {
        HSA_CHECK(hsa_amd_memory_pool_free(dev_ptr));
    }
};

/**
 * @brief Frees resources associated with @p buffer.
 */
static void ggml_backend_hsa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    delete ctx;
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
    auto * ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    return ctx->dev_ptr;
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
static void ggml_backend_hsa_buffer_memset_tensor(
    ggml_backend_buffer_t /* buffer */,
    ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size) {
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
static void ggml_backend_hsa_buffer_set_tensor(
    ggml_backend_buffer_t /* buffer */,
    ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {
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
static void ggml_backend_hsa_buffer_get_tensor(
    ggml_backend_buffer_t /* buffer */,
    const ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {
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
static bool ggml_backend_hsa_buffer_cpy_tensor(
    ggml_backend_buffer_t /* buffer */,
    const ggml_tensor * src,
    ggml_tensor * dst) {
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
    auto * ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    std::memset(ctx->dev_ptr, value, buffer->size);
}

/**
 * @brief Interface for HSA buffers.
 */
static const ggml_backend_buffer_i ggml_backend_hsa_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_hsa_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_hsa_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_hsa_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_hsa_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_hsa_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_hsa_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_hsa_buffer_clear,
    /* .reset           = */ nullptr,
};

// HSA buffer type

/**
 * @brief Context information for HSA backend buffer type.
 */
struct ggml_backend_hsa_buffer_type_context {
    std::int32_t device; ///< ID of the device associated with this buffer type context.
    std::string name;    ///< Name of the buffer type context.

    ggml_backend_hsa_buffer_type_context(std::int32_t device) :
        device(device), name(ggml_hsa_format_name(device)) {
    }
};

/**
 * @brief Returns the name associated with the buffer type @p buft.
 */
static const char * ggml_backend_hsa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
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
static ggml_backend_buffer_t ggml_backend_hsa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & info = ggml_hsa_info();
    const auto & device = info.devices[buft_ctx->device];

    void * buffer = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(device.data_memory.memory_pool, size, /* flags = */ 0, &buffer);
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: hsa_amd_memory_pool_allocate failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, ggml_hsa_get_status_string(status));
        return nullptr;
    }

    auto * ctx = new ggml_backend_hsa_buffer_context(buft_ctx->device, buffer);
    return ggml_backend_buffer_init(buft, ggml_backend_hsa_buffer_interface, ctx, size);
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
    auto * ctx = static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & info = ggml_hsa_info();
    const auto & device = info.devices[ctx->device];
    return device.data_memory.max_alloc_size;
}

/**
 * @brief Returns the size required for tensor @p tensor in buffer @p buft.
 */
static size_t ggml_backend_hsa_buffer_type_get_alloc_size(ggml_backend_buffer_type_t /* buft */, const ggml_tensor * tensor) {
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
 * @brief Interface for managing HSA buffer types.
 */
static const ggml_backend_buffer_type_i ggml_backend_hsa_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hsa_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_hsa_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hsa_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_hsa_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_hsa_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

/**
 * @brief HSA buffer types metadata.
 */
static struct {
    std::mutex mutex;
    ggml_backend_buffer_type type[GGML_HSA_MAX_DEVICES];
    bool initialized{false};
} ggml_backend_hsa_buffer_type_metadata;

ggml_backend_buffer_type_t ggml_backend_hsa_buffer_type(int device) {
    const auto device_count = ggml_backend_hsa_get_device_count();

    if (device >= device_count) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ggml_backend_hsa_buffer_type_metadata.mutex);

    if (!ggml_backend_hsa_buffer_type_metadata.initialized) {
        for (std::int32_t i = 0; i < device_count; ++i) {
            ggml_backend_hsa_buffer_type_metadata.type[i] = {
                /* .iface    = */ ggml_backend_hsa_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), i),
                /* .context  = */ new ggml_backend_hsa_buffer_type_context{i},
            };
        }
        ggml_backend_hsa_buffer_type_metadata.initialized = true;
    }

    return &ggml_backend_hsa_buffer_type_metadata.type[device];
}

// HSA split buffer

// TODO

// HSA split buffer type

/**
 * @brief Returns if @p buft is a split buffer.
 */
static bool ggml_backend_buft_is_hsa_split(ggml_backend_buffer_type_t /* buft */) {
    return false;
}

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

static ggml_backend_buffer_t ggml_backend_hsa_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
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
            /* .get_name         = */ ggml_backend_hsa_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_hsa_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_hsa_buffer_type_host;
}

/// kernels

#include <Eigen/Dense>

template<typename T>
void ggml_hsa_mul_mat_impl(ggml_backend_hsa_context &, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    auto m_in1 = matrix_type::Map(static_cast<T*>(src0->data), src0->ne[0], src0->ne[1]);
    auto m_in2 = matrix_type::Map(static_cast<T*>(src1->data), src1->ne[0], src1->ne[1]);
    auto m_out = matrix_type::Map(static_cast<T*>(dst->data), dst->ne[0], dst->ne[1]);
    m_out = m_in1.transpose() * m_in2;
}

static void ggml_hsa_mul_mat(ggml_backend_hsa_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    assert(src0->type == src1->type);
    assert(dst->type == dst->type);

    if (ggml_is_transposed(dst->src[0]) ||
        ggml_is_transposed(dst->src[1])) {
        GGML_ABORT("%s: %s: matmul on tranposed tensor not supported", __func__, ggml_op_name(dst->op));
    }

    switch (src0->type) {
        case GGML_TYPE_F16:
            ggml_hsa_mul_mat_impl<Eigen::half>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F32:
            ggml_hsa_mul_mat_impl<float>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F64: {
            ggml_hsa_mul_mat_impl<double>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("%s: Unsupported type %s", __func__, ggml_type_name(src0->type));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

// backend

/**
 * @brief Returns the name of backend @p backend.
 */
static const char * ggml_backend_hsa_get_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_hsa_context *>(backend->context);
    return ctx->name.c_str();
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
static void ggml_backend_hsa_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT((ggml_backend_hsa_get_tensor_buft(tensor) == ggml_backend_dev_buffer_type(backend->device)) &&
                "unsupported buffer type");
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
static void ggml_backend_hsa_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT((ggml_backend_hsa_get_tensor_buft(tensor) == ggml_backend_dev_buffer_type(backend->device)) &&
                "unsupported buffer type");
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
static bool ggml_backend_hsa_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_hsa(backend_src) || !ggml_backend_is_hsa(backend_dst)) {
        return false;
    }
    if (!ggml_backend_buffer_is_hsa(src->buffer) || !ggml_backend_buffer_is_hsa(dst->buffer)) {
        return false;
    }
    std::memcpy(dst->data, src->data, ggml_nbytes(dst));
    return true;
}

static void ggml_backend_hsa_synchronize(ggml_backend_t backend) {
    GGML_LOG_WARN("%s: needs synchronize kernel\n", __func__);
}

static enum ggml_status ggml_backend_hsa_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
#if 1
    auto backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    ggml_backend_graph_compute(backend_cpu, cgraph);
    ggml_backend_free(backend_cpu);
#else
    auto * ctx = static_cast<ggml_backend_hsa_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        if (ggml_is_empty(node)) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_PERMUTE:
            case GGML_OP_RESHAPE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_VIEW:
                continue;
            case GGML_OP_MUL_MAT:
                ggml_hsa_mul_mat(*ctx, node->src[0], node->src[1], node);
                break;
            default: {
                GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
        }
    }
#endif

    return GGML_STATUS_SUCCESS;
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
    /* .get_name                = */ ggml_backend_hsa_get_name,
    /* .free                    = */ ggml_backend_hsa_free,
    /* .set_tensor_async        = */ ggml_backend_hsa_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_hsa_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_hsa_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_hsa_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_hsa_graph_compute,
    /* .event_record            = */ ggml_backend_hsa_event_record,
    /* .event_wait              = */ ggml_backend_hsa_event_wait,
};

/**
 * @brief Returns the unique identifier of the HSA backend.
 *
 * @note The identifier is a UUID v4 that was randomly generated.
 */
static ggml_guid_t ggml_backend_hsa_guid() {
    static ggml_guid guid = {0xa2, 0xe9, 0xa0, 0x84, 0x2c, 0xf6, 0x4d, 0xa1, 0xb3, 0xb2, 0xb1, 0xdc, 0x5d, 0x59, 0x21, 0x95};
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
int ggml_backend_hsa_get_device_count() {
    return ggml_hsa_info().device_count;
}

/**
 * @brief Returns the device description of device @p device.
 */
void ggml_backend_hsa_get_device_description(int device, char * description, size_t description_size) {
    const auto & info = ggml_hsa_info();
    const auto & dev = info.devices[device];
    snprintf(description, description_size, "%s", dev.name.data());
}

/**
 * @brief Returns the free and total memory in @p free and @p total respectively for device @p dev.
 */
void ggml_backend_hsa_get_device_memory(int device, size_t * free, size_t * total) {
    const auto & info = ggml_hsa_info();
    const auto & dev = info.devices[device];
    *total = dev.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

bool ggml_backend_hsa_register_host_buffer(void * buffer, size_t size) {
    NOT_IMPLEMENTED();
    return false;
}

void ggml_backend_hsa_unregister_host_buffer(void * buffer) {
    NOT_IMPLEMENTED();
}

// backend device

struct ggml_backend_hsa_device_context {
    std::int32_t device;
    std::string name;
    std::string description;

    ggml_backend_hsa_device_context(std::int32_t device, hsa_agent_t agent) :
        device(device), name(ggml_hsa_format_name(device)), description(ggml_hsa_agent_name(agent)) {
    }
};

static const char * ggml_backend_hsa_device_get_name(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ctx->name.c_str();
}

static const char * ggml_backend_hsa_device_get_description(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ctx->description.c_str();
}

/**
 * @brief Returns the free and total memory in @p free and @p total respectively for device @p dev.
 */
static void ggml_backend_hsa_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & info = ggml_hsa_info();
    const auto & device = info.devices[ctx->device];
    *total = device.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

/**
 * @brief Returns the device type of @p dev.
 */
static enum ggml_backend_dev_type ggml_backend_hsa_device_get_type(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & info = ggml_hsa_info();
    const auto & device = info.devices[ctx->device];
    switch (device.type) {
        case HSA_DEVICE_TYPE_CPU:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case HSA_DEVICE_TYPE_GPU:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case HSA_DEVICE_TYPE_DSP:
        case HSA_DEVICE_TYPE_AIE:
            return GGML_BACKEND_DEVICE_TYPE_ACCEL;
        default:
            GGML_ABORT("%s: error: unknown HSA device type %d", __func__, device.type);
    }
}

static void ggml_backend_hsa_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hsa_device_get_name(dev);
    props->description = ggml_backend_hsa_device_get_description(dev);
    props->type        = ggml_backend_hsa_device_get_type(dev);
    ggml_backend_hsa_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_hsa_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ggml_backend_hsa_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_hsa_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<ggml_backend_hsa_device_context *>(dev->context);
    return ggml_backend_hsa_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_hsa_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_hsa_host_buffer_type();
}

/**
 * @brief Returns if the operation in tensor @p op is supported by device @p dev.
 */
static bool ggml_backend_hsa_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
#if 1
    auto backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    auto result = ggml_backend_supports_op(backend_cpu, op);
    ggml_backend_free(backend_cpu);
    return result;
#else
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_PERMUTE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return true;
        case GGML_OP_MUL_MAT:
            return true;
        default:
            GGML_LOG_ERROR("%s: error: unknown operator %s", __func__, ggml_op_name(op->op));
            return false;
    }
#endif
}

static bool ggml_backend_hsa_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return (ggml_backend_buft_is_hsa(buft) || ggml_backend_buft_is_hsa_split(buft)) && buft->device == dev;
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
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

static bool ggml_backend_hsa_device_offload_op(ggml_backend_dev_t /* dev */, const ggml_tensor * op) {
    const int min_batch_size = 32;
    return get_op_batch_size(op) >= min_batch_size;
}

static ggml_backend_event_t ggml_backend_hsa_device_event_new(ggml_backend_dev_t dev) {
    NOT_IMPLEMENTED();
    return nullptr;
}

static void ggml_backend_hsa_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    NOT_IMPLEMENTED();
}

/**
 * @brief Interface for managing HSA devices.
 */
static const ggml_backend_device_i ggml_backend_hsa_device_interface = {
    /* .get_name                = */ ggml_backend_hsa_device_get_name,
    /* .get_description         = */ ggml_backend_hsa_device_get_description,
    /* .get_memory              = */ ggml_backend_hsa_device_get_memory,
    /* .get_type                = */ ggml_backend_hsa_device_get_type,
    /* .get_props               = */ ggml_backend_hsa_device_get_props,
    /* .init_backend            = */ ggml_backend_hsa_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_hsa_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_hsa_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ nullptr,
    /* .supports_op             = */ ggml_backend_hsa_device_supports_op,
    /* .supports_buft           = */ ggml_backend_hsa_device_supports_buft,
    /* .offload_op              = */ ggml_backend_hsa_device_offload_op,
    /* .event_new               = */ ggml_backend_hsa_device_event_new,
    /* .event_free              = */ ggml_backend_hsa_device_event_free,
    /* .event_synchronize       = */ ggml_backend_hsa_device_event_synchronize,
};

// backend reg

struct ggml_backend_hsa_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_hsa_reg_get_name(ggml_backend_reg_t /* reg */) {
    return GGML_HSA_NAME;
}

static size_t ggml_backend_hsa_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = static_cast<ggml_backend_hsa_reg_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_hsa_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<ggml_backend_hsa_reg_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_hsa_get_features(ggml_backend_reg_t /* reg */) {
    static std::vector<ggml_backend_feature> features = [] {
        return std::vector<ggml_backend_feature>{};
    }();
    return features.data();
}

static void * ggml_backend_hsa_reg_get_proc_address(ggml_backend_reg_t /* reg */, const char * name) {
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
    std::mutex mutex;
    bool initialized{false};
} ggml_backend_hsa_reg_metadata;

ggml_backend_reg_t ggml_backend_hsa_reg() {
    std::lock_guard<std::mutex> lock(ggml_backend_hsa_reg_metadata.mutex);
    if (!ggml_backend_hsa_reg_metadata.initialized) {
        const auto & info = ggml_hsa_info();

        auto * ctx = new ggml_backend_hsa_reg_context;

        ctx->devices.reserve(info.device_count);
        for (std::int32_t i = 0; i <  info.device_count; i++) {
            auto * dev_ctx = new ggml_backend_hsa_device_context{i, info.devices[i].agent};

            auto dev = new ggml_backend_device {
                /* .iface   = */ ggml_backend_hsa_device_interface,
                /* .reg     = */ &ggml_backend_hsa_reg_metadata.reg,
                /* .context = */ dev_ctx
            };
            ctx->devices.push_back(dev);
        }

        ggml_backend_hsa_reg_metadata.reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_hsa_reg_interface,
            /* .context     = */ ctx
        };

        ggml_backend_hsa_reg_metadata.initialized = true;
    }

    return &ggml_backend_hsa_reg_metadata.reg;
}

ggml_backend_t ggml_backend_hsa_init(int device) {
    const auto & info = ggml_hsa_info();

    if (device < 0 || device >= info.device_count) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_hsa_context * ctx = nullptr;
    try {
        ctx = new ggml_backend_hsa_context{device};
    } catch (const std::bad_alloc&) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t hsa_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_hsa_guid(),
        /* .interface = */ ggml_backend_hsa_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), device),
        /* .context   = */ ctx,
    };

    return hsa_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hsa_reg)
