// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-hsa/common.hpp"
#include "ggml-hsa/host-ops.hpp"
#include "ggml-hsa/kernel-discovery.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

bool g_ggml_hsa_verbose = [] {
    if (const char * verbose = std::getenv("GGML_HSA_ENABLE_LOG"); verbose != nullptr) {
        return ggml_hsa_string_to_bool(verbose);
    }
#if defined(NDEBUG)
    return false;
#else
    return true;
#endif
}();

/// @brief Whether to report SCALE / DIAG_MASK_INF / SOFT_MAX as supported despite the in-graph AIE
/// queue fault that otherwise routes them to the CPU (see ggml_backend_hsa_device_supports_op).
/// Read once from @c GGML_HSA_ENABLE_FAULTING_OPS at startup. Intended for the standalone device
/// tests of those kernels, which dispatch one op at a time and so do not trigger the fault.
bool g_ggml_hsa_enable_faulting_ops = [] {
    if (const char * enable = std::getenv("GGML_HSA_ENABLE_FAULTING_OPS"); enable != nullptr) {
        return ggml_hsa_string_to_bool(enable);
    }
    return false;
}();

/// @brief Packets to accumulate before ringing the doorbell, or 0 if unset/invalid (use the
/// per-queue default). Read once from @c GGML_HSA_DISPATCH_BATCH_SIZE at startup.
static const std::size_t g_ggml_hsa_dispatch_batch_size = [] {
    const char * env = std::getenv("GGML_HSA_DISPATCH_BATCH_SIZE");
    if (env == nullptr) {
        return static_cast<std::size_t>(0);
    }
    std::size_t parsed = 0;
    const auto * end = env + std::strlen(env);
    const auto [ptr, ec] = std::from_chars(env, end, parsed);
    if (ec != std::errc{} || ptr != end || parsed == 0) {
        GGML_HSA_LOG_WARN("ggml_hsa: ignoring invalid GGML_HSA_DISPATCH_BATCH_SIZE (\"%s\")", env);
        return static_cast<std::size_t>(0);
    }
    return parsed;
}();

/// @brief Last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses.
#define MATRIX_ROW_PADDING 512

/// @brief Size in bytes of the buffer used to query @c HSA_AGENT_INFO_NAME.
static constexpr std::size_t ggml_hsa_agent_name_size = 64;

#define NOT_IMPLEMENTED()                                                                          \
    do {                                                                                           \
        GGML_ABORT("(%s:%d) %s not implemented\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);       \
    } while (false)

bool ggml_hsa_string_to_bool(std::string_view s) {
    return s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "yes" || s == "Yes" ||
           s == "YES" || s == "on" || s == "On" || s == "ON";
}

const char * ggml_hsa_op_name(ggml_hsa_op op) {
    switch (op) {
        case GGML_HSA_OP_CONVERT_PAD:
            return "HSA_CONVERT_PAD";
        case GGML_HSA_OP_DEPAD:
            return "HSA_DEPAD";
        case GGML_HSA_OP_CONVERT:
            return "HSA_CONVERT";
        case GGML_HSA_OP_COUNT:
            break;
    }
    GGML_ABORT("invalid ggml_hsa_op: %d", static_cast<int>(op));
}

/**
 * @brief Returns a printable operation name for @p t, accepting HSA-only operators.
 *
 * @c ggml_op_desc indexes @c GGML_OP_NAME, an array of exactly @c GGML_OP_COUNT entries, without a
 * bounds check, so passing it a tensor carrying an HSA-only op (numbered above @c GGML_OP_COUNT)
 * reads past the end of that array. Error paths shared by both kinds of node must use this instead.
 */
static const char * ggml_hsa_tensor_op_desc(const ggml_tensor & t) {
    if (ggml_hsa_is_hsa_op(t.op)) {
        return ggml_hsa_op_name(static_cast<ggml_hsa_op>(t.op));
    }
    return ggml_op_desc(&t);
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
    GGML_LOG_ERROR("HSA error (%s) in function %s at %s:%d: %s\n",
                   ggml_hsa_get_status_string(status), func, file, line, stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT("HSA error");
}

std::int32_t ggml_hsa_nsrcs(const ggml_tensor & tensor) {
    // count backwards to handle "holes" in the src[] array - e.g., for SOFT_MAX with mask=nullptr
    // but sinks!=nullptr has src[0]=input, src[1]=nullptr, src[2]=sinks
    std::int32_t last_src_idx = GGML_MAX_SRC - 1;
    for (; (last_src_idx >= 0) && (tensor.src[last_src_idx] == nullptr); --last_src_idx) {
    }
    return last_src_idx + 1;
}

/**
 * @brief Checks whether all operation parameters of a tensor are zero.
 *
 * This function inspects the tensor's op_params array and
 * determines if every 32-bit element is zero.
 *
 * @param[in] tensor Tensor whose operation parameters are to be checked.
 *
 * @return `true` if all elements of op_params are zero;
 *         `false` otherwise.
 */
static bool ggml_hsa_op_params_all_zero(const ggml_tensor & tensor) {
    const std::int32_t * params = tensor.op_params;
    const std::size_t num_elements = GGML_MAX_OP_PARAMS / sizeof(std::int32_t);
    return std::all_of(params, params + num_elements, [](int32_t x) { return x == 0; });
}

/**
 * @brief Returns if @p op is a unary operation.
 */
constexpr bool ggml_hsa_is_unary_op(ggml_op op) {
    return (op == GGML_OP_UNARY) || (op == GGML_OP_SQR) || (op == GGML_OP_SQRT) ||
           (op == GGML_OP_LOG) || (op == GGML_OP_SIN) || (op == GGML_OP_COS) ||
           (op == GGML_OP_SILU_BACK) || (op == GGML_OP_LEAKY_RELU);
}

/**
 * @brief Returns a kernel name for @p tensor using @p op_name as the operations name if it is not
 *        empty.
 */
static std::string ggml_hsa_create_kernel_name(const ggml_tensor & tensor,
                                               std::string op_name = "") {
    // Accept both the upstream ggml ops [GGML_OP_NONE, GGML_OP_COUNT) and the HSA-only ops
    // (GGML_OP_COUNT, GGML_HSA_OP_COUNT); the latter always supply an explicit op_name.
    const bool is_ggml_op = (tensor.op >= GGML_OP_NONE) && (tensor.op < GGML_OP_COUNT);
    if (!is_ggml_op && !ggml_hsa_is_hsa_op(tensor.op)) {
        throw std::runtime_error{std::string("Tensor \"")
                                     .append(ggml_get_name(&tensor))
                                     .append("\" operation index out of bounds: ")
                                     .append(std::to_string(static_cast<int>(tensor.op)))
                                     .append(" not in [0, GGML_OP_COUNT) or (GGML_OP_COUNT, "
                                             "GGML_HSA_OP_COUNT)")};
    }

    // no operation name supplied - use the tensor operation name
    if (op_name.empty()) {
        op_name = ggml_hsa_tensor_op_desc(tensor);
    }

    std::ostringstream oss;

    // convert name in lowercase
    std::transform(op_name.begin(), op_name.end(), std::ostreambuf_iterator(oss), [&](char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    });

    // output tensor
    oss << '-';
    ggml_hsa_output_tensor(tensor, oss);

    // input tensors
    const auto nsrcs = ggml_hsa_nsrcs(tensor);
    for (std::int32_t i = 0; i < nsrcs; ++i) {
        oss << '-';
        if (tensor.src[i] == nullptr) {
            // a source may be nullptr, e.g., for SOFT_MAX with src[0]=input, src[1]=nullptr,
            // src[2]=sinks
            oss << "null";
        } else {
            ggml_hsa_output_tensor(*(tensor.src[i]), oss);
        }
    }

    // determine if op_params need to be encoded in the kernel name
    if (!ggml_hsa_is_unary_op(tensor.op) && !ggml_hsa_op_params_all_zero(tensor)) {
        oss << '-';
        ggml_hsa_encode_op_params(tensor, oss);
    }

    return oss.str();
}

/**
 * @brief Returns if @p op is an element-wise operation.
 */
constexpr bool ggml_hsa_is_elementwise_op(ggml_op op) {
    return (op == GGML_OP_ADD) || (op == GGML_OP_SUB) || (op == GGML_OP_MUL) ||
           (op == GGML_OP_DIV) || (op == GGML_OP_SCALE);
}

/**
 * @brief Returns if @p op can be flattened.
 *
 * An operation can be flattened if its result is independent of how elements are laid out across
 * dimensions: unary operations always qualify; element-wise operations qualify when all input and
 * output tensors have the same shape.
 */
static bool ggml_hsa_can_flatten(const ggml_tensor & op) {
    // operations with non-contiguously allocated tensors cannot be flattened
    if (!ggml_is_contiguously_allocated(&op)) {
        return false;
    }
    for (auto src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
        if (op.src[src_idx] == nullptr) {
            break;
        }
        if (!ggml_is_contiguously_allocated(op.src[src_idx])) {
            return false;
        }
    }

    if (ggml_hsa_is_unary_op(op.op)) {
        // unary operations can be flattened independently of the tensors' shape
        return true;
    }

    if (ggml_hsa_is_elementwise_op(op.op)) {
        // element-wise operations can be flattened only if the shapes match
        for (auto src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            if (op.src[src_idx] == nullptr) {
                break;
            }
            if (!ggml_are_same_shape(op.src[src_idx], &op)) {
                return false;
            }
        }

        return true;
    }

    return false;
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
    char agent_name[ggml_hsa_agent_name_size] = {};
    GGML_HSA_CHECK_THROW(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name));
    return std::string{agent_name};
}

/**
 * @brief Returns the minimum queue size.
 */
static std::uint32_t ggml_hsa_get_agent_min_queue_size(hsa_agent_t agent) {
    std::uint32_t min_queue_size = 0;
    GGML_HSA_CHECK_THROW(hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size));
    return min_queue_size;
}

/**
 * @brief Populates the information in @p info from @p pool.
 */
static hsa_status_t ggml_hsa_get_memory_pool_info(hsa_amd_memory_pool_t pool,
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
 * @brief Memory pool discovery information.
 */
struct ggml_hsa_find_memory_pool_data_t {
    /// Expected memory pool flags.
    hsa_amd_memory_pool_global_flag_t expected_flags =
        HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    /// @c true if allocation is expected from the pool.
    bool expected_allocatable = true;
    /// Retrieved memory pool information.
    ggml_hsa_device_info::memory_pool_info mem_info;
};

/**
 * @brief Find a pool with the required flags.
 */
static hsa_status_t ggml_hsa_find_memory_pool(hsa_amd_memory_pool_t pool, void * data) {
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

    // check if flags satisfied
    auto & mem_pool_data = *static_cast<ggml_hsa_find_memory_pool_data_t *>(data);
    if ((pool_flags & mem_pool_data.expected_flags) == 0x0) {
        return HSA_STATUS_SUCCESS;
    }

    // check if allocation satisfied
    std::size_t alloc_rec_granule = 0;
    if (auto status = hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &alloc_rec_granule);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }
    const bool allocable = (alloc_rec_granule != 0);
    if (mem_pool_data.expected_allocatable != allocable) {
        return HSA_STATUS_SUCCESS;
    }

    if (auto status = ggml_hsa_get_memory_pool_info(pool, mem_pool_data.mem_info);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }
    return HSA_STATUS_INFO_BREAK;
}

/**
 * @brief Find a memory pool with the required flags and populate @p mem_info on success.
 *
 * @return @c HSA_STATUS_INFO_BREAK if a matching pool was found (with @p mem_info populated),
 *         @c HSA_STATUS_SUCCESS if iteration finished without finding a match, or an error status.
 */
static hsa_status_t ggml_hsa_find_pool(hsa_agent_t agent,
                                       hsa_amd_memory_pool_global_flag_t flags,
                                       bool allocatable,
                                       ggml_hsa_device_info::memory_pool_info & mem_info) {
    ggml_hsa_find_memory_pool_data_t mem_pool_data = {};
    mem_pool_data.expected_flags = flags;
    mem_pool_data.expected_allocatable = allocatable;
    auto status =
        hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_memory_pool, &mem_pool_data);
    if (status == HSA_STATUS_INFO_BREAK) {
        mem_info = mem_pool_data.mem_info;
    }
    return status;
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

    switch (type) {
        case HSA_DEVICE_TYPE_AIE:
            break;
        default:
            // only consider AIE agents for now
            return HSA_STATUS_SUCCESS;
    }

    auto & info = *static_cast<ggml_hsa_device_info *>(data);
    if (info.device_count >= GGML_HSA_MAX_DEVICES) {
        GGML_ABORT("%s: exceeded GGML_HSA_MAX_DEVICES limit (%d)", __func__, GGML_HSA_MAX_DEVICES);
    }

    // populate device information (agent, type, name, etc.)
    auto & dev_info = info.devices[info.device_count];
    dev_info.device = info.device_count;
    dev_info.agent = agent;
    dev_info.type = type;

    char name[ggml_hsa_agent_name_size] = {};
    if (auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
        status != HSA_STATUS_SUCCESS) {
        return status;
    }
    dev_info.name = std::string(name);

    if (dev_info.name == "aie2" || dev_info.name == "aie2p") {
        dev_info.substitute_fp16_bf16 = true;
        GGML_ASSERT(dev_info.alignment % 4 == 0);
    } else {
        GGML_ABORT("%s: unknown agent \"%s\"\n", __func__, dev_info.name.c_str());
    }

    // find dev memory pool (only for AIE agents)
    if (type == HSA_DEVICE_TYPE_AIE) {
        // XDNA dev heap is coarse-grained with alloc_rec_granule == 0
        auto status = ggml_hsa_find_pool(agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                         /*allocatable=*/false, dev_info.dev_memory);
        if (status == HSA_STATUS_SUCCESS) {
            // iteration finished with no errors, but no pool found
            return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
        }
        if (status != HSA_STATUS_INFO_BREAK) {
            // iteration aborted with errors
            return status;
        }
    }

    // find data pool
    {
        auto status = ggml_hsa_find_pool(agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
                                         /*allocatable=*/true, dev_info.data_memory);
        if (status == HSA_STATUS_SUCCESS) {
            // iteration finished with no errors, but no pool found
            return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
        }
        if (status != HSA_STATUS_INFO_BREAK) {
            // iteration aborted with errors
            return status;
        }
    }

    // find kernarg pool
    {
        auto status = ggml_hsa_find_pool(agent, HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT,
                                         /*allocatable=*/true, dev_info.kernarg_memory);
        if (status == HSA_STATUS_SUCCESS) {
            // iteration finished with no errors, but no pool found; use data pool
            dev_info.kernarg_memory = dev_info.data_memory;
        } else if (status != HSA_STATUS_INFO_BREAK) {
            // iteration aborted with errors
            return status;
        }
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
    GGML_HSA_CHECK_THROW(hsa_init());

    ggml_hsa_device_info info = {};
    GGML_HSA_CHECK_THROW(hsa_iterate_agents(ggml_hsa_find_hsa_agents, &info));

    return info;
}

/**
 * @brief Returns a mutable reference to the HSA device information singleton.
 */
static ggml_hsa_device_info & ggml_hsa_info_mut() {
    static ggml_hsa_device_info info = ggml_hsa_init();
    return info;
}

const ggml_hsa_device_info & ggml_hsa_info() { return ggml_hsa_info_mut(); }

const ggml_hsa_device_info::device_info & ggml_hsa_get_device_info(std::int32_t device_id) {
    const auto & info = ggml_hsa_info();
    const auto & dev_info = info.devices[device_id];
    return dev_info;
}

/**
 * @brief Caches the @p new_kernel for the tensor @p tensor.name on the device @p device_id.
 */
static void ggml_hsa_cache_kernel(std::string kernel_name,
                                  std::int32_t device_id,
                                  std::shared_ptr<ggml_hsa_kernel> kernel) {
    auto & info = ggml_hsa_info_mut();
    auto & dev_info = info.devices[device_id];
    auto & kernels = dev_info.kernels;
    auto result = kernels.emplace(std::move(kernel_name), std::move(kernel));
    if (!result.second) {
        GGML_ABORT("%s: kernel %s already exists on device %d\n", __func__, kernel_name.c_str(),
                   device_id);
    }
}

/**
 * @brief Returns the cached kernel for @p kernel_name for the device @p device_id if it exists.
 */
static std::shared_ptr<ggml_hsa_kernel>
ggml_hsa_get_cached_kernel(const std::string & kernel_name,
                           const ggml_hsa_device_info::device_info & dev_info) {
    const auto & kernels = dev_info.kernels;
    auto it = kernels.find(kernel_name);
    if (it != kernels.end()) {
        return it->second;
    }
    return nullptr;
}

/**
 * @brief Builds (or fetches from cache) an internal single-input transform kernel.
 *
 * Used for the HSA-only transform operators, which are not GGML ops. A
 * carrier tensor is synthesized from @p out (shape/dtype/strides of the transform's destination)
 * with its single source set to @p in, then compiled under @p op (e.g. @ref GGML_HSA_OP_CONVERT_PAD
 * or @ref GGML_HSA_OP_DEPAD). The (in, out) shapes/dtypes flow into the kernel name so each
 * distinct padded/unpadded combination caches its own PDI.
 *
 * @param[in] dev_info device information
 * @param[in] op HSA-only operator selecting the kernel source
 * @param[in] in transform input tensor (metadata only; data pointer not required here)
 * @param[in] out transform output tensor (metadata only)
 * @return the compiled/cached kernel, or nullptr on failure (caller falls back to the host path)
 */
static std::shared_ptr<ggml_hsa_kernel>
ggml_hsa_build_transform_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                ggml_hsa_op op,
                                const ggml_tensor & in,
                                const ggml_tensor & out) {
    const std::string op_name = ggml_hsa_op_name(op);

    // carrier tensor: destination metadata, single source = transform input. Both objects are
    // function-local and die together; kernel creation below reads their metadata synchronously and
    // retains neither, so carrier.src[0] never outlives source. Anything that starts deferring or
    // storing the carrier has to copy it first.
    ggml_tensor carrier = out;
    ggml_tensor source = in;
    for (auto & s : carrier.src) {
        s = nullptr;
    }
    carrier.src[0] = &source;

    // The carrier describes the transform, not whatever produced @p out. @p out is sometimes an
    // internal source node, i.e. a shallow copy of a producer graph tensor, which would otherwise
    // drag that producer's op and op_params in: ggml_hsa_create_kernel_name appends an op_params
    // hash gated on tensor.op, so two identical transforms would land under different cache keys
    // (and recompile) purely because of what produced their input. Transform tensors built by
    // ggml_hsa_new_transform carry no op_params, so this is a no-op for those.
    carrier.op = static_cast<ggml_op>(op);
    std::fill(std::begin(carrier.op_params), std::end(carrier.op_params), 0);

    auto kernel_name = ggml_hsa_create_kernel_name(carrier, op_name);
    auto kernel = ggml_hsa_get_cached_kernel(kernel_name, dev_info);
    if (kernel != nullptr) {
        return kernel;
    }
    if (ggml_hsa_create_kernel(dev_info, carrier, op_name, kernel_name, kernel) !=
        GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_hsa_cache_kernel(kernel_name, dev_info.device, kernel);
    return kernel;
}

/**
 * @brief Deletes all unused cached kernels.
 */
static void ggml_hsa_purge_unused_cached_kernels(std::int32_t device_id) {
    auto & info = ggml_hsa_info_mut();
    auto & dev_info = info.devices[device_id];
    auto & kernels = dev_info.kernels;
    for (auto it = kernels.begin(); it != kernels.end();) {
        if (it->second.use_count() == 1) {
            it = kernels.erase(it);
        } else {
            ++it;
        }
    }
}

/**
 * @brief Makes a shallow copy of @p src tensor in @p dst.
 *
 * @param src source tensor
 * @param dst destination tensor
 */
static void ggml_hsa_shallow_copy(const ggml_tensor & src, ggml_tensor & dst) { dst = src; }

/**
 * @brief Returns if @p tensor has a trivial layout.
 *
 * A tensor with a trivial layout is contiguously allocated and is not permuted.
 */
static bool ggml_hsa_has_trivial_layout(const ggml_tensor & tensor) {
    return ggml_is_contiguously_allocated(&tensor) && !ggml_is_permuted(&tensor);
}

/**
 * @brief Recomputes the strides of @p tensor from its shape for a contiguous, unpermuted layout.
 *
 * After this function is called, the @p tensor has a trivial layout.
 */
static void ggml_hsa_set_contiguous_strides(ggml_tensor & tensor) {
    tensor.nb[0] = ggml_type_size(tensor.type);
    tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(tensor.type));
    for (std::int32_t i = 2; i < GGML_MAX_DIMS; ++i) {
        tensor.nb[i] = tensor.nb[i - 1] * tensor.ne[i - 1];
    }
}

/**
 * @brief Flattens @p tensor.
 */
static void ggml_hsa_flatten_tensor(ggml_tensor & tensor) {
    const auto nelements = ggml_nelements(&tensor);
    tensor.ne[0] = nelements;
    std::fill_n(std::next(tensor.ne), GGML_MAX_DIMS - 1, 1);
    ggml_hsa_set_contiguous_strides(tensor);
}

ggml_backend_hsa_tensor_extra::ggml_backend_hsa_tensor_extra(
    const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor & parent_tensor) {

    sources.count = ggml_hsa_nsrcs(parent_tensor);

    // View tensors are generally not supported, but some operations like GGML_OP_CLAMP
    // are created as views in GGML even though they can be treated as non-in-place.
    // We allow these specific operations to proceed.
    if (ggml_is_view(&parent_tensor) && parent_tensor.op != GGML_OP_CLAMP) {
        throw std::runtime_error{"View tensor is not supported."};
    }

    // initialize internal nodes
    ggml_hsa_shallow_copy(parent_tensor, node.tensor);
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        if (parent_tensor.src[src_idx] == nullptr) {
            throw std::runtime_error{std::string("Source tensor ") + std::to_string(src_idx) +
                                     " is null. Holes are not supported."};
        }
        ggml_hsa_shallow_copy(*parent_tensor.src[src_idx], sources[src_idx].tensor);
        node.tensor.src[src_idx] = &sources[src_idx].tensor;
    }
    assert(ggml_hsa_nsrcs(node.tensor) == sources.count);

    // early exit if operation does not require a kernel
    if (ggml_op_is_empty(node.tensor.op)) {
        return;
    }

    // HSA-only operators (single-input transforms: convert/pad, de-pad, dtype cast). The kernel
    // maps the sole parent source into this node's shape/dtype; no generic layout/flatten handling.
    if (ggml_hsa_is_hsa_op(node.tensor.op)) {
        const auto hsa_op = static_cast<ggml_hsa_op>(node.tensor.op);
        if (parent_tensor.src[0] == nullptr) {
            throw std::runtime_error{std::string{"HSA transform \""}
                                         .append(ggml_get_name(&parent_tensor))
                                         .append("\" (")
                                         .append(ggml_hsa_op_name(hsa_op))
                                         .append(") has no source tensor")};
        }
        kernel =
            ggml_hsa_build_transform_kernel(dev_info, hsa_op, *parent_tensor.src[0], parent_tensor);
        if (kernel == nullptr) {
            throw std::runtime_error{std::string{"Could not build HSA transform kernel for \""}
                                         .append(ggml_get_name(&parent_tensor))
                                         .append("\" (")
                                         .append(ggml_hsa_op_name(hsa_op))
                                         .append(")")};
        }
        return;
    }

    switch (node.tensor.op) {
        // implemented as host kernels; nothing to be done
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_GET_ROWS:
            return;
        default:
            break;
    }

    std::array<bool, GGML_MAX_SRC> update_src_buffer_size = {};
    // Tracks *why* a source needs a temporary buffer. A source whose only difference from its
    // parent is the element type can be pre-processed on the device by the element-wise
    // GGML_HSA_OP_CONVERT kernel; one that also needs its layout rewritten cannot, because that
    // kernel only streams contiguous runs. The flatten step below erases the distinction from the
    // tensors themselves, so it has to be recorded here.
    std::array<bool, GGML_MAX_SRC> src_dtype_only = {};

    // convert tensor data types if needed
    if (dev_info.substitute_fp16_bf16) {
        // output tensor can be converted in-place
        if (node.tensor.type == GGML_TYPE_F16) {
            node.tensor.type = GGML_TYPE_BF16;
            node.convert_dtype = true;
        }

        // inputs require temporary storage as they may be shared among tensors
        for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
            auto & src_node = sources[src_idx];
            if (src_node.tensor.type == GGML_TYPE_F16) {
                update_src_buffer_size[src_idx] = true;
                src_dtype_only[src_idx] = true;
                src_node.tensor.type = GGML_TYPE_BF16;
            }
        }
    }

    // make tensor layouts trivial; tensors that do not have a trivial layout will need
    // temporary storage
    if (!ggml_hsa_has_trivial_layout(node.tensor)) {
        throw std::runtime_error{"Output tensor does not have trivial layout."};
    }
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        auto & src_node = sources[src_idx];
        if (!ggml_hsa_has_trivial_layout(src_node.tensor)) {
            update_src_buffer_size[src_idx] = true;
            src_dtype_only[src_idx] = false;
            ggml_hsa_set_contiguous_strides(src_node.tensor);
        }
    }

    // flatten tensors to reuse kernels
    if (ggml_hsa_can_flatten(node.tensor)) {
        ggml_hsa_flatten_tensor(node.tensor);
        for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
            ggml_hsa_flatten_tensor(sources[src_idx].tensor);
        }
    }

    // update required tensor sizes
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        if (update_src_buffer_size[src_idx]) {
            auto & src_node = sources[src_idx];
            src_node.tensor.data = nullptr;
            src_node.buffer_size = GGML_PAD(ggml_nbytes(&src_node.tensor), dev_info.alignment);
        }
    }

    // Build on-device pre-processing kernels for the sources that only change element type. These
    // run the parent -> internal conversion on the device queue in place of the host copy in
    // graph_compute, removing the queue drain that copy would otherwise force. A null kernel (the
    // dtype pair or element count is not streamable) simply leaves the source on the host path.
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        if (!src_dtype_only[src_idx] || !ggml_is_contiguous(parent_tensor.src[src_idx])) {
            continue;
        }
        sources[src_idx].preprocess_kernel = ggml_hsa_build_transform_kernel(
            dev_info, GGML_HSA_OP_CONVERT, *parent_tensor.src[src_idx], sources[src_idx].tensor);
    }

    // Decide how each group (sources, output) synchronizes its parent<->internal transformations
    // independently. On-device transformations run on the same in-order queue as the main kernel,
    // so no host queue drain is needed and the packets batch with surrounding work; the host
    // fallback must drain (before the dispatch for sources, after it for the output, since the host
    // may not touch a buffer the device is still using). Device is preferred; the groups do not
    // have to agree.

    // Sources: a source needs pre-processing when it has a transformed internal buffer. If every
    // such source has a pre-processing kernel the whole group runs on-device; otherwise a host
    // transformation on any source drains before all of them.
    const bool sources_need_sync = std::any_of(
        sources.begin(), sources.end(), [](const source_node_t & s) { return s.buffer_size != 0; });
    const bool sources_device_capable =
        std::all_of(sources.begin(), sources.end(), [](const source_node_t & s) {
            return s.buffer_size == 0 || s.preprocess_kernel != nullptr;
        });
    if (!sources_need_sync) {
        sources.sync_mode = sync_mode_t::none;
    } else if (sources_device_capable) {
        sources.sync_mode = sync_mode_t::device;
    } else {
        sources.sync_mode = sync_mode_t::host;
    }

    // Output: needs post-processing when the result is dtype-converted back into the parent. There
    // is no on-device kernel for that direction yet (the convert kernel has no f32 -> f16 path), so
    // it always drains after the dispatch and converts on the host.
    node.sync_mode = node.convert_dtype ? sync_mode_t::host : sync_mode_t::none;

    // create a kernel for the operation
    auto kernel_name = ggml_hsa_create_kernel_name(node.tensor);
    kernel = ggml_hsa_get_cached_kernel(kernel_name, dev_info);
    if (kernel == nullptr) {
        // kernel not in cache; create a new one and store it in the cache
        if (ggml_hsa_create_kernel(dev_info, node.tensor, std::nullopt, kernel_name, kernel) !=
            GGML_STATUS_SUCCESS) {
            throw std::runtime_error{std::string{"Could not create kernel for tensor \""}
                                         .append(node.tensor.name)
                                         .append("\" (")
                                         .append(ggml_hsa_tensor_op_desc(node.tensor))
                                         .append(")")};
        }
        ggml_hsa_cache_kernel(std::move(kernel_name), dev_info.device, kernel);
    }
}

ggml_status ggml_backend_hsa_tensor_extra::allocate_internal_storage(
    const ggml_hsa_device_info::device_info & dev_info) {
    if (buffer != nullptr) {
        // already allocated
        return GGML_STATUS_ABORTED;
    }

    std::size_t buffer_size = 0;
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        buffer_size += sources[src_idx].buffer_size;
    }

    if (buffer_size == 0) {
        // no temporary storage needed
        return GGML_STATUS_SUCCESS;
    }

    // allocate storage for all tensors
    void * ptr = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.data_memory.memory_pool, buffer_size,
                                                   /* flags = */ 0, &ptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate %.2f MiB on device %s (%s)", __func__,
                           (buffer_size / 1024.0 / 1024.0), dev_info.name.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_ALLOC_FAILED;
    }
    buffer.reset(static_cast<std::byte *>(ptr));

    auto buffer_ptr = buffer.get();
    for (auto src_idx = 0; src_idx < sources.count; ++src_idx) {
        auto & src_node = sources[src_idx];
        if (src_node.buffer_size > 0) {
            assert(src_node.tensor.data == nullptr);
            src_node.tensor.data = buffer_ptr;
            buffer_ptr += src_node.buffer_size;
        }
    }

    GGML_HSA_LOG_INFO("%s: created temporary storage for tensor %s (%s)", __func__,
                      node.tensor.name, ggml_hsa_tensor_op_desc(node.tensor));

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Records an asynchronous queue error on the owning context.
 *
 * The runtime calls this synchronously, on the thread that rang the doorbell, when it cannot
 * submit a packet; it then suspends the queue. Only the first error is kept: that is the one that
 * suspended the queue, and every later failure is a consequence of it.
 *
 * @param[in] status error reported by the runtime
 * @param[in] source queue the error occurred on (unused; the context identifies it)
 * @param[in] data   owning @ref ggml_backend_hsa_context, passed at queue creation
 */
static void ggml_hsa_queue_error_callback(hsa_status_t status, hsa_queue_t * source, void * data) {
    GGML_UNUSED(source);

    auto & ctx = *static_cast<ggml_backend_hsa_context *>(data);
    hsa_status_t expected = HSA_STATUS_SUCCESS;
    if (ctx.queue_error.compare_exchange_strong(expected, status)) {
        GGML_HSA_LOG_ERROR("%s: queue suspended by the runtime: %s", __func__,
                           ggml_hsa_get_status_string(status));
    }
}

ggml_backend_hsa_context::ggml_backend_hsa_context(
    const ggml_hsa_device_info::device_info & dev_info) :
    device{dev_info.device}, name{ggml_hsa_format_name(device)} {
    hsa_agent_t agent = dev_info.agent;

    // create queue
    const std::uint32_t min_queue_size = ggml_hsa_get_agent_min_queue_size(agent);
    // The callback receives `this` while the object is still under construction. It only touches
    // queue_error, which its default member initializer has already set, and the runtime cannot
    // invoke it before a packet is submitted -- which cannot happen until construction finishes.
    if (auto status =
            hsa_queue_create(agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE,
                             ggml_hsa_queue_error_callback, this, 0, 0, &queue);
        status != HSA_STATUS_SUCCESS) {
        throw std::runtime_error{std::string("Could not create hsa_queue (")
                                     .append(ggml_hsa_get_status_string(status))
                                     .append(")")};
    }

    // create signal to wait for packets
    if (auto status = hsa_signal_create(0, 0, nullptr, &dispatch_signal);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_CHECK_ABORT(hsa_queue_destroy(queue));
        throw std::runtime_error{std::string("Could not create hsa_signal (")
                                     .append(ggml_hsa_get_status_string(status))
                                     .append(")")};
    }

    // One kernarg slot per queue ring slot, reused only once the device consumes the packet that
    // referenced it. Sized for the worst-case kernarg region (all sources + destination, pointer +
    // size each), so a free ring slot always has a free kernarg slot.
    const std::size_t kernarg_alignment =
        std::max<std::size_t>(dev_info.kernarg_memory.alignment, alignof(std::uint64_t));
    constexpr std::size_t max_kernarg_entries =
        (GGML_MAX_SRC + 1 /* destination */) * 2 /* pointer + size */;
    const std::size_t slot_size = max_kernarg_entries * sizeof(std::uint64_t);
    try {
        kernargs = ggml_hsa_kernarg_pool{dev_info.kernarg_memory.memory_pool, queue->size,
                                         slot_size, kernarg_alignment};
    } catch (...) {
        // release the HSA resources acquired above before propagating the failure
        GGML_HSA_CHECK_ABORT(hsa_signal_destroy(dispatch_signal));
        GGML_HSA_CHECK_ABORT(hsa_queue_destroy(queue));
        throw;
    }

    // Packets to accumulate before ringing the doorbell. Defaults to min(32, queue depth); a
    // larger value would never be reached before the queue-full drain, so it is clamped with a
    // warning.
    constexpr std::size_t default_batch_size = 32;
    std::size_t batch_size = g_ggml_hsa_dispatch_batch_size != 0
                                 ? g_ggml_hsa_dispatch_batch_size
                                 : std::min<std::size_t>(default_batch_size, queue->size);
    if (batch_size > queue->size) {
        GGML_HSA_LOG_WARN("%s: GGML_HSA_DISPATCH_BATCH_SIZE (%zu) exceeds queue size (%u); "
                          "clamping to queue size",
                          __func__, batch_size, queue->size);
        batch_size = queue->size;
    }
    dispatch_batch_size = batch_size;
}

ggml_backend_hsa_context::~ggml_backend_hsa_context() {
    // Drain outstanding work before tearing down the queue and signal; otherwise batched-but-
    // unrung packets would be lost and in-flight packets would reference a destroyed signal.
    ggml_hsa_wait_dispatches(*this);
    ggml_hsa_purge_unused_cached_kernels(device);
    GGML_HSA_CHECK_ABORT(hsa_signal_destroy(dispatch_signal));
    GGML_HSA_CHECK_ABORT(hsa_queue_destroy(queue));
}

void ggml_hsa_flush_dispatches(ggml_backend_hsa_context & ctx) {
    if (ctx.n_batched == 0) {
        return;
    }
    // Ring the doorbell for the last written packet (write_index - 1, since write_index points one
    // past the last reserved slot); the device processes every packet up to it, and the release
    // fence publishes all pending packet writes. Safe because submission is single-producer
    // (HSA_QUEUE_TYPE_SINGLE, one thread): no other thread reserves a slot between a packet write
    // and this ring. Concurrent submission would need to track the last fully-published index.
    const std::uint64_t last_packet = hsa_queue_load_write_index_relaxed(ctx.queue) - 1;
    hsa_signal_store_screlease(ctx.queue->doorbell_signal, last_packet);
    ctx.n_batched = 0;
}

void ggml_hsa_wait_dispatches(ggml_backend_hsa_context & ctx) {
    // Flush pending packets first, otherwise the wait below could block forever on packets that
    // were written but never rung. The flush is also what surfaces a submission failure: the
    // runtime reports it through the error callback from inside the doorbell ring.
    ggml_hsa_flush_dispatches(ctx);

    // A suspended queue never releases the completion signals of the packets it did not run, so
    // the drain below would never return. graph_compute turns the recorded error into a failed
    // graph; there is nothing left to wait for here.
    if (ctx.queue_error.load(std::memory_order_relaxed) != HSA_STATUS_SUCCESS) {
        return;
    }

    if (auto val = hsa_signal_wait_scacquire(ctx.dispatch_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                             UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        val != 0) {
        GGML_ABORT("%s: unexpected signal value (%ld)\n", __func__, val);
    }
}

// HSA buffer

/**
 * @brief Context for managing a HSA buffer associated with a specific device.
 */
struct ggml_backend_hsa_buffer_context {
    std::int32_t device{};             ///< Device ID associated with this buffer context.
    ggml_hsa_unique_ptr<void> dev_ptr; ///< Pointer to the device memory.
    std::vector<std::unique_ptr<ggml_backend_hsa_tensor_extra>> tensor_extras;

    ggml_backend_hsa_buffer_context(std::int32_t device, ggml_hsa_unique_ptr<void> dev_ptr) :
        device{device}, dev_ptr{std::move(dev_ptr)} {}
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
    return buf_ctx.dev_ptr.get();
}

/**
 * @brief Initializes the tensor.
 */
static enum ggml_status ggml_backend_hsa_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor * tensor) {
    // View tensors generally don't need initialization, but some operations like CLAMP
    // are created as views in GGML even though they have actual compute work.
    // These need tensor_extra for kernel dispatch.
    if (ggml_is_view(tensor) && tensor->op != GGML_OP_CLAMP) {
        // no further initialization needed for views
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    assert(tensor->extra == nullptr);

    auto & buf_ctx = *static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    const auto & dev_info = ggml_hsa_get_device_info(buf_ctx.device);

    try {
        // initialize tensor extra
        auto tensor_extra = std::make_unique<ggml_backend_hsa_tensor_extra>(dev_info, *tensor);
        if (auto status = tensor_extra->allocate_internal_storage(dev_info);
            status != GGML_STATUS_SUCCESS) {
            return status;
        }
        // register tensor extra with the buffer context and the tensor
        buf_ctx.tensor_extras.push_back(std::move(tensor_extra));
        tensor->extra = buf_ctx.tensor_extras.back().get();
    } catch (const std::exception & ex) {
        GGML_HSA_LOG_ERROR("%s: exception caught: %s", __func__, ex.what());
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
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
    if (ggml_backend_buffer_is_hsa(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

/**
 * @brief Clear buffer @p buffer by setting all its memory to @p value.
 */
static void ggml_backend_hsa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto & buf_ctx = *static_cast<ggml_backend_hsa_buffer_context *>(buffer->context);
    std::memset(buf_ctx.dev_ptr.get(), value, buffer->size);
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
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
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
ggml_backend_hsa_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & dev_info = ggml_hsa_get_device_info(buft_ctx.device);

    void * buffer = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(dev_info.data_memory.memory_pool, size,
                                                   /* flags = */ 0, &buffer);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to allocate %.2f MiB on device %s (%s)", __func__,
                           (size / 1024.0 / 1024.0), dev_info.name.c_str(),
                           ggml_hsa_get_status_string(status));
        return nullptr;
    }

    try {
        auto * buf_ctx =
            new ggml_backend_hsa_buffer_context(buft_ctx.device, ggml_hsa_unique_ptr<void>{buffer});
        return ggml_backend_buffer_init(buft, ggml_backend_hsa_buffer_interface, buf_ctx, size);
    } catch (const std::exception & ex) {
        GGML_HSA_LOG_ERROR("%s: exception caught: %s", __func__, ex.what());
        return nullptr;
    }
}

/**
 * @brief Returns the memory alignment requirement for buffer type @p buft in bytes.
 */
static size_t ggml_backend_hsa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & dev_info = ggml_hsa_get_device_info(buft_ctx.device);
    return dev_info.alignment;
}

/**
 * @brief Returns the maximum allocation size for buffer type @p buft in bytes.
 */
static size_t ggml_backend_hsa_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const auto & buft_ctx = *static_cast<ggml_backend_hsa_buffer_type_context *>(buft->context);
    const auto & dev_info = ggml_hsa_get_device_info(buft_ctx.device);
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
 * @brief Interface for managing HSA buffer types.
 */
static const ggml_backend_buffer_type_i ggml_backend_hsa_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_hsa_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hsa_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hsa_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hsa_buffer_type_get_max_size,
    /* .get_alloc_size = */ ggml_backend_hsa_buffer_type_get_alloc_size,
    /* .is_host        = */ nullptr,
};

/**
 * @brief HSA buffer types.
 */
static struct {
    ggml_backend_buffer_type type[GGML_HSA_MAX_DEVICES];
    std::once_flag flag;
} ggml_backend_hsa_buffer_type_metadata;

ggml_backend_buffer_type_t ggml_backend_hsa_buffer_type(std::int32_t device) {
    const auto device_count = ggml_backend_hsa_get_device_count();

    if (device < 0 || device >= device_count) {
        return nullptr;
    }

    try {
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
        GGML_HSA_LOG_ERROR("%s: exception caught: %s", __func__, ex.what());
        return nullptr;
    }
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

[[noreturn]] static void
ggml_backend_hsa_host_buffer_free_buffer(ggml_backend_buffer_t /* buffer */) {
    // TODO free buffer
    NOT_IMPLEMENTED();
}

static void * ggml_hsa_host_malloc(size_t /* size */) {
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
    return (ggml_is_view(tensor) ? tensor->view_src->buffer : tensor->buffer)->buft;
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
    std::memcpy(data, static_cast<std::byte *>(tensor->data) + offset, size);
    GGML_UNUSED(backend);
}

/**
 * @brief Copy tensor data between buffers if possible.
 *
 * Both tensors must be contiguous; the number of bytes copied is @c ggml_nbytes(dst).
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
    ggml_hsa_wait_dispatches(ctx);
}

/**
 * @brief Pre-processes a node's sources into their internal buffers before the main kernel
 * dispatch.
 *
 * The path is selected by @c tensor_extra.sources.sync_mode. On the device path
 * (@c sync_mode_t::device) each source that needs a transformed buffer is dispatched on-queue via
 * its `preprocess_kernel` (an element-wise dtype conversion), with no queue drain. On the host path
 * (@c sync_mode_t::host) the queue is drained first (the host may not touch a buffer the device is
 * still using), then each source is copied into its internal buffer. No-op when
 * @c sources.sync_mode is @c none.
 *
 * @param[in,out] ctx HSA backend context (queue used for drains and on-device dispatches)
 * @param[in,out] tensor_extra node metadata holding the internal source nodes and sync mode
 * @param[in] node parent graph node whose sources are pre-processed
 * @return @c GGML_STATUS_SUCCESS, or the failing status of the first source that could not be
 *         prepared
 */
static ggml_status ggml_hsa_dispatch_preprocess(ggml_backend_hsa_context & ctx,
                                                ggml_backend_hsa_tensor_extra & tensor_extra,
                                                ggml_tensor * node) {
    using sync_mode_t = ggml_backend_hsa_tensor_extra::sync_mode_t;

    if (tensor_extra.sources.sync_mode == sync_mode_t::none) {
        return GGML_STATUS_SUCCESS;
    }

    const bool use_device_transforms = tensor_extra.sources.sync_mode == sync_mode_t::device;
    ggml_tensor & internal_node = tensor_extra.node.tensor;

    if (!use_device_transforms) {
        ggml_hsa_wait_dispatches(ctx);
    }
    for (auto src_idx = 0; src_idx < tensor_extra.sources.count; ++src_idx) {
        if (tensor_extra.sources[src_idx].buffer_size == 0) {
            continue;
        }
        ggml_status status = GGML_STATUS_SUCCESS;
        if (use_device_transforms) {
            // on-device source pre-processing: convert the parent source into its internal buffer
            // on-queue, no drain
            ggml_tensor * preprocess_src = node->src[src_idx];
            status = tensor_extra.sources[src_idx].preprocess_kernel->dispatch(
                ctx, &preprocess_src, 1, *internal_node.src[src_idx]);
        } else {
            // change layout and/or convert datatypes
            status = ggml_hsa_copy_tensor(node->src[src_idx], internal_node.src[src_idx]);
        }
        if (status != GGML_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to prepare source %i for tensor \"%s (%s)\"", __func__,
                               src_idx, node->name, ggml_hsa_tensor_op_desc(*node));
            return status;
        }
    }
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Post-processes a node's internal output buffer back into the parent tensor after dispatch.
 *
 * The path is selected by @c tensor_extra.node.sync_mode. Only @c sync_mode_t::host is currently
 * reachable: the queue is drained and the result is converted back into the parent on the host.
 * No-op when @c node.sync_mode is @c none.
 *
 * @param[in,out] ctx HSA backend context (queue used for the drain)
 * @param[in,out] tensor_extra node metadata holding the internal output node and sync mode
 * @param[in,out] node parent graph node that receives the post-processed result
 * @return @c GGML_STATUS_SUCCESS, or the failing status of the post-processing step
 */
static ggml_status ggml_hsa_dispatch_postprocess(ggml_backend_hsa_context & ctx,
                                                 ggml_backend_hsa_tensor_extra & tensor_extra,
                                                 ggml_tensor * node) {
    using sync_mode_t = ggml_backend_hsa_tensor_extra::sync_mode_t;

    if (tensor_extra.node.sync_mode != sync_mode_t::host) {
        return GGML_STATUS_SUCCESS;
    }

    // change layout and/or convert datatypes
    ggml_hsa_wait_dispatches(ctx);
    ggml_status status = ggml_hsa_copy_tensor(&tensor_extra.node.tensor, node);
    if (status != GGML_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: failed to copy back for tensor \"%s\" (%s)", __func__, node->name,
                           ggml_hsa_tensor_op_desc(*node));
    }
    return status;
}

static enum ggml_status ggml_backend_hsa_graph_compute(ggml_backend_t backend,
                                                       ggml_cgraph * cgraph) {
    auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    ggml_status status = GGML_STATUS_SUCCESS;

    const std::int32_t node_count = ggml_graph_n_nodes(cgraph);

    // shallow copies may not have been fully initialized when the graph was created, so we need to
    // make sure all nodes have their source tensor pointers set before we can start dispatching
    // kernels
    for (std::int32_t i = 0; i < node_count; ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        if (ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
            continue;
        }

        auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(node->extra);
        for (auto src_idx = 0; src_idx < tensor_extra.sources.count; ++src_idx) {
            if (tensor_extra.sources[src_idx].tensor.data == nullptr) {
                tensor_extra.sources[src_idx].tensor.data = node->src[src_idx]->data;
            }
        }
    }

    for (std::int32_t i = 0; (i < node_count) && (status == GGML_STATUS_SUCCESS); ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        // early exit if operation does not require a dispatch
        if (ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
            continue;
        }

        switch (node->op) {
            // implemented as host kernels, so no dispatch required
            case GGML_OP_DUP:
                status = ggml_hsa_compute_dup(ctx, node);
                continue;
            case GGML_OP_CPY:
                status = ggml_hsa_compute_cpy(ctx, node);
                continue;
            case GGML_OP_CONT:
                status = ggml_hsa_compute_cont(ctx, node);
                continue;
            case GGML_OP_GET_ROWS:
                status = ggml_hsa_compute_get_rows(ctx, node);
                continue;
            default:
                break;
        }

        auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(node->extra);
        ggml_tensor & internal_node = tensor_extra.node.tensor;

        // break out of the node loop on failure so the trailing flush still runs
        if (status = ggml_hsa_dispatch_preprocess(ctx, tensor_extra, node);
            status != GGML_STATUS_SUCCESS) {
            break;
        }

        if (status = tensor_extra.kernel->dispatch(ctx, internal_node.src,
                                                   tensor_extra.sources.count, internal_node);
            status != GGML_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to dispatch kernel for tensor \"%s\" (%s)", __func__,
                               node->name, ggml_hsa_tensor_op_desc(*node));
            break;
        }

        if (status = ggml_hsa_dispatch_postprocess(ctx, tensor_extra, node);
            status != GGML_STATUS_SUCCESS) {
            break;
        }
    }

    // Flush unconditionally, including on the error paths above: packets written for
    // successfully-dispatched earlier nodes must be rung so their work reaches the device and
    // callers reading via the buffer get/copy paths (which don't synchronize) see complete results.
    // This is also where a submission failure surfaces: the runtime runs the queue error callback
    // from inside the doorbell ring.
    ggml_hsa_flush_dispatches(ctx);

    // A suspended queue cannot have produced this graph's results, and it stays suspended, so
    // every later graph on this backend fails here too rather than returning wrong data.
    if (ctx.queue_error.load(std::memory_order_relaxed) != HSA_STATUS_SUCCESS) {
        return GGML_STATUS_FAILED;
    }

    return status;
}

// HSA-only graph operators (see ggml-hsa.h)

/**
 * @brief Builds a single-node result carrying an HSA-only transform op with @p a as its sole
 * source.
 */
static ggml_tensor * ggml_hsa_new_transform(ggml_context * ctx,
                                            ggml_tensor * a,
                                            ggml_hsa_op op,
                                            ggml_type type,
                                            const int64_t ne[GGML_MAX_DIMS]) {
    ggml_tensor * result = ggml_new_tensor(ctx, type, GGML_MAX_DIMS, ne);
    result->op = static_cast<ggml_op>(op);
    result->src[0] = a;
    return result;
}

ggml_tensor * ggml_hsa_convert_pad(
    ggml_context * ctx, ggml_tensor * a, ggml_type type, int64_t ne0, int64_t ne1) {
    const int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, 1, 1};
    return ggml_hsa_new_transform(ctx, a, GGML_HSA_OP_CONVERT_PAD, type, ne);
}

ggml_tensor *
ggml_hsa_depad(ggml_context * ctx, ggml_tensor * a, ggml_type type, int64_t ne0, int64_t ne1) {
    const int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, 1, 1};
    return ggml_hsa_new_transform(ctx, a, GGML_HSA_OP_DEPAD, type, ne);
}

ggml_tensor * ggml_hsa_convert(ggml_context * ctx, ggml_tensor * a, ggml_type type) {
    return ggml_hsa_new_transform(ctx, a, GGML_HSA_OP_CONVERT, type, a->ne);
}

// event

/**
 * @brief Per-event data captured at record time.
 *
 * Because the dispatch signal is a counting signal (incremented per submission, decremented per
 * completion), there is no cheap point-in-time fence.  We instead record whether any work was
 * in-flight at record time; if so, the wait drains the entire queue (conservative but correct).
 */
struct ggml_hsa_event_context {
    /// Backend context the event was recorded on, used to flush pending packets before waiting.
    /// Only valid when @ref snapshot is non-zero. Not owned; the caller must ensure the recording
    /// backend outlives every wait on this event.
    ggml_backend_hsa_context * ctx{};
    hsa_signal_value_t snapshot{}; ///< Non-zero means work was in flight; wait must drain fully.
    hsa_queue_t * queue{};         ///< Queue the event was recorded on (for same-queue detection).

    /// @brief Dispatch signal at record time. Only valid when @ref snapshot is non-zero.
    hsa_signal_t signal() const { return ctx->dispatch_signal; }
};

/**
 * @brief Blocks until all work submitted before the corresponding @c event_record call completes.
 *
 * Because @c dispatch_signal is a counting signal (incremented on each submission, decremented on
 * each completion), waiting for it to drop below the recorded snapshot would only guarantee that
 * at least one dispatch completed, not all pre-record work.  When work was in flight at record
 * time (@p ec.snapshot != 0), this function conservatively drains the entire queue by waiting for
 * the signal to reach zero.
 *
 * @param[in] ec Event context populated by @c ggml_backend_hsa_event_record.
 */
static void ggml_hsa_event_wait_for_snapshot(const ggml_hsa_event_context & ec) {
    if (ec.snapshot == 0) {
        return;
    }
    // Non-zero snapshot is only ever set together with ec.ctx, so ctx is valid here.
    assert(ec.ctx != nullptr);
    // Flush pending packets on the recording context; the drain below waits for the dispatch
    // signal to reach zero, which never happens if written packets were never rung.
    ggml_hsa_flush_dispatches(*ec.ctx);
    // Same reasoning as in ggml_hsa_wait_dispatches: a suspended queue never drains.
    if (ec.ctx->queue_error.load(std::memory_order_relaxed) != HSA_STATUS_SUCCESS) {
        return;
    }
    hsa_signal_wait_scacquire(ec.signal(), HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
}

/**
 * @brief Records a fence point on @p backend into @p event.
 *
 * Snapshots the current value of the backend's dispatch signal.  A non-zero snapshot indicates
 * that work was in flight at record time; a later @c ggml_backend_hsa_device_event_synchronize or
 * cross-queue @c ggml_backend_hsa_event_wait will conservatively drain the entire queue to ensure
 * all pre-record work has completed.
 *
 * @param[in]  backend Backend whose in-flight work the event should fence.
 * @param[out] event   Event to record into; must have been created with
 *                     @c ggml_backend_hsa_device_event_new.
 */
static void ggml_backend_hsa_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    auto & ec = *static_cast<ggml_hsa_event_context *>(event->context);
    // Flush pending packets so the work fenced by this event is actually in flight; otherwise a
    // later wait on the snapshot could block on packets that were never rung.
    ggml_hsa_flush_dispatches(ctx);
    ec.ctx = &ctx;
    ec.snapshot = hsa_signal_load_scacquire(ctx.dispatch_signal);
    ec.queue = ctx.queue;
}

/**
 * @brief Inserts a device-side dependency on @p event into @p backend's command stream.
 *
 * The AIE queue only accepts kernel dispatch packets, so a true GPU-side barrier cannot be
 * enqueued.  Work on the same in-order queue is implicitly ordered, so when @p backend uses the
 * same queue as the recording backend this is a no-op.  For a different queue the CPU blocks until
 * the recorded work completes (conservative but correct).
 *
 * @param[in] backend Backend that should wait for @p event before proceeding.
 * @param[in] event   Event previously populated by @c ggml_backend_hsa_event_record.
 */
static void ggml_backend_hsa_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    const auto & ec = *static_cast<ggml_hsa_event_context *>(event->context);
    const auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    if (ec.queue != ctx.queue) {
        ggml_hsa_event_wait_for_snapshot(ec);
    }
}

/**
 * @brief Interface for managing HSA backends.
 */
static const ggml_backend_i ggml_backend_hsa_interface = {
    /* .get_name            = */ ggml_backend_hsa_get_name,
    /* .free                = */ ggml_backend_hsa_free,
    /* .set_tensor_async    = */ ggml_backend_hsa_set_tensor_async,
    /* .get_tensor_async    = */ ggml_backend_hsa_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ ggml_backend_hsa_cpy_tensor_async,
    /* .synchronize         = */ ggml_backend_hsa_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_hsa_graph_compute,
    /* .event_record        = */ ggml_backend_hsa_event_record,
    /* .event_wait          = */ ggml_backend_hsa_event_wait,
    /* .graph_optimize      = */ nullptr,
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
 * @brief Returns the number of devices (i.e., HSA agents) associated with the HSA backend.
 */
std::int32_t ggml_backend_hsa_get_device_count() { return ggml_hsa_info().device_count; }

/**
 * @brief Returns the device description of device @p device.
 */
void ggml_backend_hsa_get_device_description(std::int32_t device,
                                             char * description,
                                             size_t description_size) {
    const auto & dev_info = ggml_hsa_get_device_info(device);
    snprintf(description, description_size, "%s", dev_info.name.data());
}

/**
 * @brief Returns the free and total memory in @p free and @p total respectively for device
 *        @p dev.
 */
void ggml_backend_hsa_get_device_memory(std::int32_t device, size_t * free, size_t * total) {
    const auto & dev_info = ggml_hsa_get_device_info(device);
    *total = dev_info.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

bool ggml_backend_hsa_register_host_buffer(void * /* buffer */, size_t /* size */) {
    NOT_IMPLEMENTED();
    return false;
}

void ggml_backend_hsa_unregister_host_buffer(void * /* buffer */) { NOT_IMPLEMENTED(); }

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
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & dev_info = ggml_hsa_get_device_info(dev_ctx.device);
    *total = dev_info.data_memory.size;
    // HSA does not report free memory, set it to total
    *free = *total;
}

/**
 * @brief Returns the device type of @p dev.
 */
static enum ggml_backend_dev_type ggml_backend_hsa_device_get_type(ggml_backend_dev_t dev) {
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & dev_info = ggml_hsa_get_device_info(dev_ctx.device);
    switch (dev_info.type) {
        case HSA_DEVICE_TYPE_CPU:
            return GGML_BACKEND_DEVICE_TYPE_CPU;
        case HSA_DEVICE_TYPE_GPU:
            return GGML_BACKEND_DEVICE_TYPE_GPU;
        case HSA_DEVICE_TYPE_DSP:
        case HSA_DEVICE_TYPE_AIE:
            return GGML_BACKEND_DEVICE_TYPE_ACCEL;
        default:
            GGML_ABORT("%s: unknown HSA device type %d", __func__, dev_info.type);
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
        /* .events               = */ true,
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
static bool ggml_backend_hsa_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // early exit if operation does not require a kernel
    if (ggml_op_is_empty(op->op)) {
        return true;
    }

    switch (op->op) {
        // implemented as host kernels
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return true;
        case GGML_OP_GET_ROWS:
            // host gather: int32 indices, float table -> f32 output (covers the dtype
            // combinations ggml_hsa_assign supports for this op).
            return (op->src[0] != nullptr) && (op->src[1] != nullptr) &&
                   (op->src[1]->type == GGML_TYPE_I32) &&
                   ((op->src[0]->type == GGML_TYPE_F32) || (op->src[0]->type == GGML_TYPE_F16) ||
                    (op->src[0]->type == GGML_TYPE_BF16)) &&
                   (op->type == GGML_TYPE_F32);
        // GPT-2 attention block (KQ -> SCALE -> DIAG_MASK_INF -> SOFT_MAX -> KQV, over the 3D KQ
        // tensor [n_kv, N, n_head]). Each of these IRON kernels is correct in isolation -- the
        // standalone device tests pass, including on GPT-2-shaped inputs -- but when they run back
        // to back inside the full attention graph they fault the hardware AIE queue after a number
        // of tokens, which ROCr turns into an abort() at the next doorbell ring (unrecoverable, not
        // a status we can catch). Until the queue fault is root-caused, route all three to the CPU
        // fallback so graphs containing them run to completion.
        //
        // This alone does not keep the surrounding attention block on the CPU. Declining these
        // three says nothing about the neighbouring KQ/KQV MUL_MATs, and where those are supported
        // the scheduler leaves them on this device and their tensors cross the CPU/HSA boundary
        // around each declined node. It happens that for GPT-2 they are not supported either --
        // the GEMM kernel requires tile-aligned shapes and its build asserts "A/B must be tileable
        // into (m * n_aie_rows, k)-sized blocks" -- so that block does end up entirely on the CPU,
        // but that is a property of the MUL_MAT shapes, not something this switch arranges.
        //
        // The kernels themselves are still built and still correct, so their standalone device
        // tests must keep running: reporting "unsupported" to those would make them skip every
        // case and pass vacuously, leaving the kernels unguarded against regressions. They set
        // GGML_HSA_ENABLE_FAULTING_OPS to opt back in. Do not set it for whole-graph workloads.
        case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
            if (!g_ggml_hsa_enable_faulting_ops) {
                return false;
            }
            break;
        default:
            break;
    }

    // check if tensor is already initialized with a valid kernel
    if ((op->extra != nullptr) &&
        (static_cast<ggml_backend_hsa_tensor_extra *>(op->extra)->kernel != nullptr)) {
        return true;
    }

    // check if compilation artifacts exist or if the kernel can be compiled
    const auto & dev_ctx = *static_cast<ggml_backend_hsa_device_context *>(dev->context);
    const auto & dev_info = ggml_hsa_get_device_info(dev_ctx.device);
    try {
        ggml_backend_hsa_tensor_extra tensor_extra{dev_info, *op};
        return (tensor_extra.kernel != nullptr);
    } catch (const std::exception & ex) {
        // exception is not fatal, it means that the op is not supported
        GGML_HSA_LOG_WARN("%s: exception caught: %s", __func__, ex.what());
        return false;
    }
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

/**
 * @brief Allocates a new event associated with @p dev.
 *
 * @param[in] dev Device the event belongs to.
 * @return Newly allocated event, or @c nullptr on allocation failure.
 */
static ggml_backend_event_t ggml_backend_hsa_device_event_new(ggml_backend_dev_t dev) {
    return new ggml_backend_event{
        /* .device  = */ dev,
        /* .context = */ new ggml_hsa_event_context,
    };
}

/**
 * @brief Frees an event previously created by @c ggml_backend_hsa_device_event_new.
 *
 * @param[in] event Event to destroy.
 */
static void ggml_backend_hsa_device_event_free(ggml_backend_dev_t /* dev */,
                                               ggml_backend_event_t event) {
    delete static_cast<ggml_hsa_event_context *>(event->context);
    delete event;
}

/**
 * @brief Blocks the calling thread until the work recorded into @p event has completed.
 *
 * @param[in] event Event previously populated by @c ggml_backend_hsa_event_record.
 */
static void ggml_backend_hsa_device_event_synchronize(ggml_backend_dev_t /* dev */,
                                                      ggml_backend_event_t event) {
    ggml_hsa_event_wait_for_snapshot(*static_cast<ggml_hsa_event_context *>(event->context));
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
    std::array<ggml_backend_feature, 1> features = {{{nullptr, nullptr}}};
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
    GGML_HSA_LOG_ERROR("%s: exception caught: %s", __func__, ex.what());
    return nullptr;
}

ggml_backend_t ggml_backend_hsa_init(std::int32_t device) {
    const auto & info = ggml_hsa_info();

    if (device < 0 || device >= info.device_count) {
        GGML_HSA_LOG_ERROR("%s: invalid device ID %d", __func__, device);
        return nullptr;
    }

    try {
        auto * ctx = new ggml_backend_hsa_context{info.devices[device]};

        ggml_backend_t hsa_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_hsa_guid(),
            /* .interface = */ ggml_backend_hsa_interface,
            /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_hsa_reg(), device),
            /* .context   = */ ctx,
        };

        return hsa_backend;
    } catch (const std::exception & ex) {
        GGML_HSA_LOG_ERROR("%s: exception caught: %s", __func__, ex.what());
        return nullptr;
    }
}

GGML_BACKEND_DL_IMPL(ggml_backend_hsa_reg)
