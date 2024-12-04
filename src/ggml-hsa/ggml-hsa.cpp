#include "ggml-hsa.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-hsa/common.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#define NOT_IMPLEMENTED() \
    do { \
        printf("%s not implemented\n", __PRETTY_FUNCTION__); \
        abort(); \
    } while (false)

[[noreturn]]
void ggml_hsa_error(const char * stmt, const char * func, const char * file, int line, hsa_status_t status) {
    const char* msg = nullptr;
    if (hsa_status_string(status, &msg) != HSA_STATUS_SUCCESS) {
        msg = "unknown";
    }

    GGML_LOG_ERROR("HSA error: %s\n", msg);
    GGML_LOG_ERROR("  in function %s at %s:%d\n", func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT("HSA error");
}

static std::string ggml_hsa_format_name(int device) {
    return GGML_HSA_NAME + std::to_string(device);
}

static ggml_hsa_device_info ggml_hsa_init() {
    HSA_CHECK(hsa_init());

    auto agent_visitor = [](hsa_agent_t agent, void* data) {
        hsa_device_type_t type = {};
        HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
        if (type != HSA_DEVICE_TYPE_AIE) {
            return HSA_STATUS_SUCCESS;
        }

        auto & info = *static_cast<ggml_hsa_device_info *>(data);
        ++info.device_count;
        return HSA_STATUS_SUCCESS;
    };

    ggml_hsa_device_info info = {};
    HSA_CHECK(hsa_iterate_agents(agent_visitor, &info));
    return info;
}

const ggml_hsa_device_info & ggml_hsa_info() {
    static ggml_hsa_device_info info = ggml_hsa_init();
    return info;
}

// HSA buffer

struct ggml_backend_hsa_buffer_context {
    void * dev_ptr{nullptr};
    std::string name;

    ggml_backend_hsa_buffer_context(void * dev_ptr) :
        dev_ptr(dev_ptr), name(ggml_hsa_format_name(0)) {
    }

    ~ggml_backend_hsa_buffer_context() {
        // TODO deallocate memory
        NOT_IMPLEMENTED();
    }
};

static void ggml_backend_hsa_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    delete ctx;
}

static bool ggml_backend_buffer_is_hsa(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_hsa_buffer_free_buffer;
}

static void * ggml_backend_hsa_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    return ctx->dev_ptr;
}

static void ggml_backend_hsa_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    // TODO memset(tensor, value)
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // TODO memcpy(tensor, value)
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // TODO memcpy(data, tensor)
    NOT_IMPLEMENTED();
}

static bool ggml_backend_hsa_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // TODO memcpy(dst, src)
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO memset(buffer, value); sync
    NOT_IMPLEMENTED();
}

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
struct ggml_backend_hsa_buffer_type_context {
    std::string name;

    ggml_backend_hsa_buffer_type_context(int device) :
        name(ggml_hsa_format_name(device)) {
    }
};

static const char * ggml_backend_hsa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_hsa(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hsa_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_hsa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // TODO
    NOT_IMPLEMENTED();
    return {};
}

static size_t ggml_backend_hsa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // TODO is this true?
    NOT_IMPLEMENTED();
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_hsa_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // TODO is this true?
    NOT_IMPLEMENTED();
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_hsa_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_hsa_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_hsa_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_hsa_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_hsa_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

static struct {
    std::mutex mutex;
    ggml_backend_buffer_type type[GGML_HSA_MAX_DEVICES];
    bool initialized{false};
} ggml_backend_hsa_buffer_type_metadata;

ggml_backend_buffer_type_t ggml_backend_hsa_buffer_type(int device) {
    std::lock_guard<std::mutex> lock(ggml_backend_hsa_buffer_type_metadata.mutex);

    auto const device_count = ggml_backend_hsa_get_device_count();
    if (device >= device_count) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_hsa_buffer_types[GGML_HSA_MAX_DEVICES];

    if (!ggml_backend_hsa_buffer_type_metadata.initialized) {
        for (int i = 0; i < device_count; i++) {
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


// host buffer type

static const char * ggml_backend_hsa_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_HSA_NAME "_Host";

    GGML_UNUSED(buft);
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

static bool ggml_hsa_compute_forward(ggml_backend_hsa_context & ctx, struct ggml_tensor * dst) {
    NOT_IMPLEMENTED();
    return false;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_hsa_get_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_hsa_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_hsa_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_hsa_context *>(backend->context);
    delete ctx;
    delete backend;
}

static void ggml_backend_hsa_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // TODO
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // TODO
    NOT_IMPLEMENTED();
}

static bool ggml_backend_hsa_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    // TODO
    NOT_IMPLEMENTED();
    return false;
}

static void ggml_backend_hsa_synchronize(ggml_backend_t backend) {
    // TODO
    NOT_IMPLEMENTED();
}

static enum ggml_status ggml_backend_hsa_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // TODO
    NOT_IMPLEMENTED();
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_hsa_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    // TODO
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // TODO
    NOT_IMPLEMENTED();
}

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

static ggml_guid_t ggml_backend_hsa_guid() {
    // TODO: add UUID
    static ggml_guid guid = { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0, 0x0, 0x0 };
    return &guid;
}

bool ggml_backend_is_hsa(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_hsa_guid());
}

int ggml_backend_hsa_get_device_count() {
    return ggml_hsa_info().device_count;
}

void ggml_backend_hsa_get_device_description(int device, char * description, size_t description_size) {
    // TODO
    NOT_IMPLEMENTED();
}

void ggml_backend_hsa_get_device_memory(int device, size_t * free, size_t * total) {
    // TODO
    NOT_IMPLEMENTED();
}

bool ggml_backend_hsa_register_host_buffer(void * buffer, size_t size) {
    // TODO
    NOT_IMPLEMENTED();
    return false;
}

void ggml_backend_hsa_unregister_host_buffer(void * buffer) {
    // TODO
    NOT_IMPLEMENTED();
}

// backend device

struct ggml_backend_hsa_device_context {
    int device;
    std::string name;
    std::string description;

    ggml_backend_hsa_device_context(int device) :
        device(device), name(ggml_hsa_format_name(device)) {
        // TODO get HSA name
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

static void ggml_backend_hsa_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    NOT_IMPLEMENTED();
}

static enum ggml_backend_dev_type ggml_backend_hsa_device_get_type(ggml_backend_dev_t dev) {
    // TODO if (dev == NPU) return GGML_BACKEND_DEVICE_TYPE_ACCEL; if (dev == GPU) return GGML_BACKEND_DEVICE_TYPE_GPU
    NOT_IMPLEMENTED();
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
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

static bool ggml_backend_hsa_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // TODO
    NOT_IMPLEMENTED();
    return false;
}

static bool ggml_backend_hsa_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return (ggml_backend_buft_is_hsa(buft) /* || ggml_backend_buft_is_cuda_split(buft) */) && buft->device == dev;
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

static bool ggml_backend_hsa_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return get_op_batch_size(op) >= min_batch_size;

    GGML_UNUSED(dev);
}

static ggml_backend_event_t ggml_backend_hsa_device_event_new(ggml_backend_dev_t dev) {
    // TODO
    NOT_IMPLEMENTED();
    return nullptr;
}

static void ggml_backend_hsa_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO
    NOT_IMPLEMENTED();
}

static void ggml_backend_hsa_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    // TODO
    NOT_IMPLEMENTED();
}

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

static const char * ggml_backend_hsa_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
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

static ggml_backend_feature * ggml_backend_hsa_get_features(ggml_backend_reg_t reg) {
    // TODO
    NOT_IMPLEMENTED();
    return nullptr;
}

static void * ggml_backend_hsa_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
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

static const ggml_backend_reg_i ggml_backend_hsa_reg_interface = {
    /* .get_name          = */ ggml_backend_hsa_reg_get_name,
    /* .get_device_count  = */ ggml_backend_hsa_reg_get_device_count,
    /* .get_device_get    = */ ggml_backend_hsa_reg_get_device,
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
        auto * ctx = new ggml_backend_hsa_reg_context;

        auto const device_count = ggml_hsa_info().device_count;
        for (int i = 0; i < device_count; i++) {
            auto * dev_ctx = new ggml_backend_hsa_device_context{i};

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
    HSA_CHECK(hsa_init());

    if (device < 0 || device >= ggml_backend_hsa_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    auto * ctx = new ggml_backend_hsa_context(device);
    if (ctx == nullptr) {
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
