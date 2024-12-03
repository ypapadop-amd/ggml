#include "ggml-hsa.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdint>
#include <mutex>
#include <string>

#include "ggml-hsa/common.hpp"

#include <hsa/hsa.h>

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

static ggml_hsa_device_info ggml_hsa_init() {
    HSA_CHECK(hsa_init());

    auto agent_visitor = [](hsa_agent_t agent, void* data) {
        hsa_device_type_t type = {};
        HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));
        if (type != HSA_DEVICE_TYPE_AIE) {
            return HSA_STATUS_SUCCESS;
        }

        auto & count = *static_cast<int *>(data);
        ++count;
        return HSA_STATUS_SUCCESS;
    };

    int device_count = 0;
    HSA_CHECK(hsa_iterate_agents(agent_visitor, &device_count));

    ggml_hsa_device_info info = {};
    info.device_count = device_count;
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
        dev_ptr(dev_ptr),
        name(GGML_HSA_NAME + std::to_string(0)) {
    }

    ~ggml_backend_hsa_buffer_context() {
        // TODO deallocate memory
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
}

static void ggml_backend_hsa_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    // TODO memcpy(tensor, value)
}

static void ggml_backend_hsa_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // TODO memcpy(data, tensor)
}

static bool ggml_backend_hsa_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // TODO memcpy(dst, src)
}

static void ggml_backend_hsa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // TODO memset(buffer, value); sync
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
        name(GGML_HSA_NAME + std::to_string(device)) {
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
    return {};
}

static size_t ggml_backend_hsa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    // TODO is this true?
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_hsa_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // TODO is this true?
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
}

static void * ggml_hsa_host_malloc(size_t size) {
    // TODO allocate pinned memory
    return nullptr;
}

static ggml_backend_buffer_t ggml_backend_hsa_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_hsa_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
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

static const ggml_backend_i ggml_backend_hsa_interface = {
    /* .get_name                = */ ggml_backend_hsa_get_name,
    /* .free                    = */ ggml_backend_hsa_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ nullptr,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
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

// backend device

// backend reg

// backend registry
ggml_backend_reg_t ggml_backend_hsa_reg() {
    static ggml_backend_reg reg;
    return &reg;
}

ggml_backend_t ggml_backend_hsa_init(int device) {
    HSA_CHECK(hsa_init());

    if (device < 0 || device >= ggml_backend_hsa_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_hsa_context * ctx = new ggml_backend_hsa_context(device);
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
