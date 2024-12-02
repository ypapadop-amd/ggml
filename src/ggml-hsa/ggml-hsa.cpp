#include "ggml-hsa.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstdint>

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
}

static void ggml_backend_hsa_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
}

static void ggml_backend_hsa_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
}

static bool ggml_backend_hsa_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
}

static void ggml_backend_hsa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
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

// host buffer type

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
