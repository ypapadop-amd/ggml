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
    if ((tensor.op < GGML_OP_NONE) || (tensor.op >= GGML_OP_COUNT)) {
        throw std::runtime_error{std::string("Tensor \"")
                                     .append(ggml_get_name(&tensor))
                                     .append("\" operation index out of bounds: ")
                                     .append(std::to_string(tensor.op))
                                     .append(" not in [0, GGML_OP_COUNT)")};
    }

    // no operation name supplied - use the tensor operation name
    if (op_name.empty()) {
        op_name = ggml_op_desc(&tensor);
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
        ggml_hsa_find_memory_pool_data_t mem_pool_data = {};
        mem_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
        mem_pool_data.expected_allocatable = false;
        auto status =
            hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_memory_pool, &mem_pool_data);
        switch (status) {
            case HSA_STATUS_INFO_BREAK:
                dev_info.dev_memory = mem_pool_data.mem_info;
                break;
            case HSA_STATUS_SUCCESS:
                // iteration finished with no errors, but no pool found
                return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
            default:
                // iteration aborted with errors
                return status;
        }
    }

    // find data pool
    {
        ggml_hsa_find_memory_pool_data_t mem_pool_data = {};
        mem_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
        mem_pool_data.expected_allocatable = true;
        auto status =
            hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_memory_pool, &mem_pool_data);
        switch (status) {
            case HSA_STATUS_INFO_BREAK:
                dev_info.data_memory = mem_pool_data.mem_info;
                break;
            case HSA_STATUS_SUCCESS:
                // iteration finished with no errors, but no pool found
                return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
            default:
                // iteration aborted with errors
                return status;
        }
    }

    // find kernarg pool
    {
        ggml_hsa_find_memory_pool_data_t mem_pool_data = {};
        mem_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
        mem_pool_data.expected_allocatable = true;
        auto status =
            hsa_amd_agent_iterate_memory_pools(agent, ggml_hsa_find_memory_pool, &mem_pool_data);
        switch (status) {
            case HSA_STATUS_INFO_BREAK:
                dev_info.kernarg_memory = mem_pool_data.mem_info;
                break;
            case HSA_STATUS_SUCCESS:
                // iteration finished with no errors, but no pool found; use data pool
                dev_info.kernarg_memory = dev_info.data_memory;
                break;
            default:
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
 * Used for the MUL_MAT convert/pad pre-amble and de-pad post-amble, which are not GGML ops. A
 * carrier tensor is synthesized from @p out (shape/dtype/strides of the transform's destination)
 * with its single source set to @p in, then compiled under @p op_name (e.g. "HSA_CONVERT_PAD" or
 * "HSA_DEPAD"). The (in, out) shapes/dtypes flow into the kernel name so each distinct
 * padded/unpadded combination caches its own PDI.
 *
 * @param[in] dev_info device information
 * @param[in] op_name internal operation name selecting the kernel source
 * @param[in] in transform input tensor (metadata only; data pointer not required here)
 * @param[in] out transform output tensor (metadata only)
 * @return the compiled/cached kernel, or nullptr on failure (caller falls back to the host path)
 */
static std::shared_ptr<ggml_hsa_kernel>
ggml_hsa_build_transform_kernel(const ggml_hsa_device_info::device_info & dev_info,
                                const std::string & op_name,
                                const ggml_tensor & in,
                                const ggml_tensor & out) {
    // carrier tensor: destination metadata, single source = transform input
    ggml_tensor carrier = out;
    ggml_tensor source = in;
    for (auto & s : carrier.src) {
        s = nullptr;
    }
    carrier.src[0] = &source;

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
 * @brief Returns whether a CPY/DUP node is a pure element-wise dtype conversion.
 *
 * True when the copy just changes dtype: source and destination are both contiguous, have the same
 * element count, and differ only in type. Such a copy can run on-device via the HSA_CONVERT kernel
 * (batched on the queue) instead of the host copy path (which drains the queue). Strided copies,
 * reshapes-via-cont, and same-dtype copies are excluded (the last has nothing to convert).
 */
static bool ggml_hsa_is_convert_copy(const ggml_tensor & tensor) {
    if (tensor.op != GGML_OP_CPY && tensor.op != GGML_OP_DUP) {
        return false;
    }
    const ggml_tensor * src = tensor.src[0];
    if (src == nullptr) {
        return false;
    }
    return src->type != tensor.type && ggml_nelements(src) == ggml_nelements(&tensor) &&
           ggml_is_contiguous(src) && ggml_is_contiguous(&tensor);
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

/**
 * @brief Eligibility for the padded bf16 GEMM path, over a raw tensor.
 *
 * Shared by @c ggml_hsa_prepare_mul_mat_f32 (which additionally constrains the destination dtype)
 * and @c ggml_backend_hsa_graph_optimize (which runs before tensor_extra exists, so it only has the
 * raw graph tensor). Deliberately does NOT check @c mm.type: the two callers disagree on the
 * destination dtype, so each applies its own rule.
 *
 * @param[in] mm candidate MUL_MAT node.
 * @return @c true if @p mm is a 2-source MUL_MAT with f32/bf16 operands, trivial layout on both
 *         operands and the destination, and no batch/broadcast.
 */
static bool ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor & mm) {
    if (mm.op != GGML_OP_MUL_MAT || mm.src[0] == nullptr || mm.src[1] == nullptr ||
        mm.src[2] != nullptr) {
        return false;
    }
    const ggml_tensor & a = *mm.src[0]; // [K, M]
    const ggml_tensor & b = *mm.src[1]; // [K, N]

    const bool a_ok = a.type == GGML_TYPE_F32 || a.type == GGML_TYPE_BF16;
    const bool b_ok = b.type == GGML_TYPE_F32 || b.type == GGML_TYPE_BF16;
    if (!a_ok || !b_ok) {
        return false;
    }
    if (!ggml_hsa_has_trivial_layout(a) || !ggml_hsa_has_trivial_layout(b) ||
        !ggml_hsa_has_trivial_layout(mm)) {
        return false;
    }
    // no batching / broadcasting
    if (a.ne[2] != 1 || a.ne[3] != 1 || b.ne[2] != 1 || b.ne[3] != 1) {
        return false;
    }
    return true;
}

/**
 * @brief Prepares an F32 @c MUL_MAT node for the AIE whole-array GEMM kernel.
 *
 * The GEMM microkernel has no native f32 path and the whole-array tiling requires the matrix
 * dimensions to be multiples of the per-architecture tile factors. This rewrites the internal node
 * (and its sources) so the kernel sees bf16 operands zero-padded up to those tile multiples:
 *   - src0 A = [K, M], src1 B = [K, N], dst C = [M, N] (GGML MUL_MAT layout);
 *   - each dimension is padded to K->Kpad, M->Mpad, N->Npad and the dtype set to bf16;
 *   - the padded regions read as zero (pre-zeroed by @ref allocate_internal_storage), so the extra
 *     rows/cols and the interior K gap contribute nothing to the result.
 * The output node keeps f32 but needs its own padded temporary storage plus a de-pad copy back into
 * the (smaller) parent tensor, flagged via @c node_t::depad.
 *
 * Only the contiguous, non-batched, non-permuted f32 x f32 case is handled (the shapes exercised by
 * MNIST). Returns @c false for anything else, leaving the node untouched so the caller falls back
 * to the generic path.
 *
 * @param[in] dev_info device information (selects the tile factors)
 * @param[in,out] node internal output node
 * @param[in,out] src_nodes internal source nodes
 * @param[in] nsrcs number of sources
 * @return @c true if the node was rewritten for the padded bf16 GEMM path
 */
static bool ggml_hsa_prepare_mul_mat_f32(const ggml_hsa_device_info::device_info & dev_info,
                                         ggml_backend_hsa_tensor_extra::node_t & node,
                                         ggml_backend_hsa_tensor_extra::node_t * src_nodes,
                                         std::int32_t nsrcs) {
    ggml_tensor & dst = node.tensor;

    // The GEMM microkernel runs in bf16, so both operands must be f32 (converted to bf16 below) or
    // already bf16 (converted in the graph, e.g. the bf16 MNIST variant). Shared eligibility is
    // checked over the internal source tensors via the raw-tensor predicate; here we additionally
    // require an f32 destination (ggml's native MUL_MAT output) or a bf16 destination (retyped by
    // graph_optimize so the de-pad narrows f32->bf16 in one pass).
    if (nsrcs != 2) {
        return false;
    }
    ggml_tensor & a = src_nodes[0].tensor; // [K, M]
    ggml_tensor & b = src_nodes[1].tensor; // [K, N]

    // Build a raw view of the node/sources for the shared predicate: node_t.tensor already mirrors
    // the parent's op/type/shape at this point in construction.
    ggml_tensor probe = dst;
    probe.src[0] = &a;
    probe.src[1] = &b;
    probe.src[2] = nullptr;
    if (!ggml_hsa_mul_mat_is_padded_gemm(probe)) {
        return false;
    }
    if (dst.type != GGML_TYPE_F32 && dst.type != GGML_TYPE_BF16) {
        return false;
    }

    // per-architecture GEMM tiling factors (see gemm.py: tile size and n_aie_rows/n_aie_cols).
    // n_aie_rows is 4 for both architectures.
    constexpr std::int64_t n_aie_rows = 4;
    std::int64_t tile = 0;
    std::int64_t n_aie_cols = 0;
    if (dev_info.name == "aie2p") {
        tile = 16;
        n_aie_cols = 8;
    } else { // aie2
        tile = 32;
        n_aie_cols = 4;
    }

    const std::int64_t K = a.ne[0];
    const std::int64_t M = a.ne[1];
    const std::int64_t N = b.ne[1];

    const std::int64_t Kpad = GGML_PAD(K, tile);
    const std::int64_t Mpad = GGML_PAD(M, tile * n_aie_rows);
    const std::int64_t Npad = GGML_PAD(N, tile * n_aie_cols);

    // rewrite sources to padded bf16; only sources that arrive as f32 need a dtype conversion,
    // bf16 sources are just zero-padded to the tile multiples (handled by the pre-processing
    // kernel below, which is selected from the parent tensor's own dtype).
    a.type = GGML_TYPE_BF16;
    a.ne[0] = Kpad;
    a.ne[1] = Mpad;
    ggml_hsa_set_contiguous_strides(a);
    src_nodes[0].tensor.data = nullptr;
    src_nodes[0].buffer_size = GGML_PAD(ggml_nbytes(&a), dev_info.alignment);

    b.type = GGML_TYPE_BF16;
    b.ne[0] = Kpad;
    b.ne[1] = Npad;
    ggml_hsa_set_contiguous_strides(b);
    src_nodes[1].tensor.data = nullptr;
    src_nodes[1].buffer_size = GGML_PAD(ggml_nbytes(&b), dev_info.alignment);

    // rewrite output to a padded f32 temporary; it must be de-padded back into the parent tensor
    dst.ne[0] = Mpad;
    dst.ne[1] = Npad;
    ggml_hsa_set_contiguous_strides(dst);
    node.tensor.data = nullptr;
    node.buffer_size = GGML_PAD(ggml_nbytes(&dst), dev_info.alignment);
    node.depad = true;

    return true;
}

ggml_backend_hsa_tensor_extra::ggml_backend_hsa_tensor_extra(
    const ggml_hsa_device_info::device_info & dev_info, const ggml_tensor & parent_tensor) :
    nsrcs{ggml_hsa_nsrcs(parent_tensor)} {

    // View tensors are generally not supported, but some operations like GGML_OP_CLAMP
    // are created as views in GGML even though they can be treated as non-in-place.
    // We allow these specific operations to proceed.
    if (ggml_is_view(&parent_tensor) && parent_tensor.op != GGML_OP_CLAMP) {
        throw std::runtime_error{"View tensor is not supported."};
    }

    // initialize internal nodes
    ggml_hsa_shallow_copy(parent_tensor, node.tensor);
    for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
        if (parent_tensor.src[src_idx] == nullptr) {
            throw std::runtime_error{std::string("Source tensor ") + std::to_string(src_idx) +
                                     " is null. Holes are not supported."};
        }
        ggml_hsa_shallow_copy(*parent_tensor.src[src_idx], src_nodes[src_idx].tensor);
        node.tensor.src[src_idx] = &src_nodes[src_idx].tensor;
    }
    assert(ggml_hsa_nsrcs(node.tensor) == nsrcs);

    // early exit if operation does not require a kernel
    if (ggml_op_is_empty(node.tensor.op)) {
        return;
    }

    switch (node.tensor.op) {
        case GGML_OP_DUP:
        case GGML_OP_CPY:
            // A pure dtype-conversion copy runs on-device via HSA_CONVERT (flattened element-wise
            // cast), so it batches on the queue instead of the host copy path that drains it. Any
            // other copy (strided, reshape) stays on the host.
            if (ggml_hsa_is_convert_copy(node.tensor)) {
                kernel = ggml_hsa_build_transform_kernel(dev_info, "HSA_CONVERT",
                                                         *parent_tensor.src[0], parent_tensor);
            }
            return;
        case GGML_OP_CONT:
            // implemented as a host kernel; nothing to be done
            return;
        default:
            break;
    }

    std::array<bool, GGML_MAX_SRC> update_src_buffer_size = {};
    bool requires_sync = false;

    // F32 MUL_MAT is handled specially: the operands are converted to bf16 and zero-padded to the
    // GEMM tile multiples. This fully sets up the internal nodes (dtype, shape, buffer sizes,
    // depad), so the generic dtype/layout/flatten handling below is skipped.
    if (ggml_hsa_prepare_mul_mat_f32(dev_info, node, src_nodes.data(), nsrcs)) {
        requires_sync = true;

        // A source that is a graph-constant leaf (a weight/bias: op == GGML_OP_NONE, not a graph
        // input) has contents that never change across dispatches, so its converted+padded (or
        // just padded) form can be produced once into the persistent internal buffer and reused.
        // graph_compute skips the pre-processing while the cached pointer matches
        // (converted_ptr).
        for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
            const ggml_tensor * src = parent_tensor.src[src_idx];
            if (src_nodes[src_idx].buffer_size != 0 && src->op == GGML_OP_NONE &&
                (src->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                src_nodes[src_idx].is_constant = true;
            }
        }
    } else {
        // convert tensor data types if needed
        if (dev_info.substitute_fp16_bf16) {
            // output tensor can be converted in-place
            if (node.tensor.type == GGML_TYPE_F16) {
                node.tensor.type = GGML_TYPE_BF16;
                node.convert_dtype = true;
            }

            // inputs require temporary storage as they may be shared among tensors
            for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
                auto & src_node = src_nodes[src_idx];
                if (src_node.tensor.type == GGML_TYPE_F16) {
                    update_src_buffer_size[src_idx] = true;
                    src_node.tensor.type = GGML_TYPE_BF16;
                }
            }
        }

        // make tensor layouts trivial; tensors that do not have a trivial layout will need
        // temporary storage
        if (!ggml_hsa_has_trivial_layout(node.tensor)) {
            throw std::runtime_error{"Output tensor does not have trivial layout."};
        }
        for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
            auto & src_node = src_nodes[src_idx];
            if (!ggml_hsa_has_trivial_layout(src_node.tensor)) {
                update_src_buffer_size[src_idx] = true;
                ggml_hsa_set_contiguous_strides(src_node.tensor);
            }
        }

        // flatten tensors to reuse kernels
        if (ggml_hsa_can_flatten(node.tensor)) {
            ggml_hsa_flatten_tensor(node.tensor);
            for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
                ggml_hsa_flatten_tensor(src_nodes[src_idx].tensor);
            }
        }

        // update required tensor sizes
        for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
            if (update_src_buffer_size[src_idx]) {
                auto & src_node = src_nodes[src_idx];
                src_node.tensor.data = nullptr;
                src_node.buffer_size = GGML_PAD(ggml_nbytes(&src_node.tensor), dev_info.alignment);
                requires_sync = true;
            }
        }
    }

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
                                         .append(ggml_op_desc(&node.tensor))
                                         .append(")")};
        }
        ggml_hsa_cache_kernel(std::move(kernel_name), dev_info.device, kernel);
    }

    // For the padded MUL_MAT path, build on-device pre-processing kernels (per source; convert
    // f32->bf16 and/or zero-pad) and a post-processing kernel (de-pad the result). These run on the
    // device queue in place of the host copies in graph_compute, removing the per-op queue drains
    // that would otherwise flush the packet batch. If any fails to compile the pointer stays null
    // and graph_compute falls back to the host copy path. A source whose operand is already bf16
    // needs no dtype conversion but still needs zero-padding to the tile multiples, so build the
    // pre-processing kernel for every source that has a padded internal buffer (HSA_CONVERT_PAD
    // selects convert+pad or pad-only from the source dtype).
    if (node.depad) {
        for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
            if (src_nodes[src_idx].buffer_size == 0) {
                continue;
            }
            src_nodes[src_idx].preprocess_kernel = ggml_hsa_build_transform_kernel(
                dev_info, "HSA_CONVERT_PAD", *parent_tensor.src[src_idx],
                src_nodes[src_idx].tensor);
        }
        postprocess_kernel =
            ggml_hsa_build_transform_kernel(dev_info, "HSA_DEPAD", node.tensor, parent_tensor);
    }

    // Prefer on-device pre/post-processing kernels when the node provides them: they run on the
    // same in-order queue as the main kernel, so no host queue drain is needed and the packets
    // batch with surrounding work. This requires the full chain (a pre-processing kernel for every
    // source that needs one, plus the post-processing kernel) to be available; otherwise fall back
    // to the host copy path (which must drain first, since the host may not touch a buffer the
    // device is still using).
    if (!requires_sync) {
        sync_mode = sync_mode_t::none;
    } else if (node.depad && postprocess_kernel != nullptr &&
               std::all_of(
                   src_nodes.begin(), src_nodes.begin() + nsrcs, [](const node_t & src_node) {
                       return src_node.buffer_size == 0 || src_node.preprocess_kernel != nullptr;
                   })) {
        sync_mode = sync_mode_t::device;
    } else {
        sync_mode = sync_mode_t::host;
    }
}

ggml_status ggml_backend_hsa_tensor_extra::allocate_internal_storage(
    const ggml_hsa_device_info::device_info & dev_info) {
    if (buffer != nullptr) {
        // already allocated
        return GGML_STATUS_ABORTED;
    }

    std::size_t buffer_size = node.buffer_size;
    for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
        buffer_size += src_nodes[src_idx].buffer_size;
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

    // pre-zero the whole block so padding gaps (trailing rows/cols and the interior K gap of a
    // padded MUL_MAT) read as zero without the per-dispatch sub-block copies having to touch them
    std::memset(buffer.get(), 0, buffer_size);

    auto buffer_ptr = buffer.get();
    for (auto src_idx = 0; src_idx < nsrcs; ++src_idx) {
        auto & src_node = src_nodes[src_idx];
        if (src_node.buffer_size > 0) {
            assert(src_node.tensor.data == nullptr);
            src_node.tensor.data = buffer_ptr;
            buffer_ptr += src_node.buffer_size;
        }
    }

    // the output node may need its own padded temporary storage (e.g. a de-padded MUL_MAT result)
    if (node.buffer_size > 0) {
        assert(node.tensor.data == nullptr);
        node.tensor.data = buffer_ptr;
        buffer_ptr += node.buffer_size;
    }

    GGML_HSA_LOG_INFO("%s: created temporary storage for tensor %s (%s)", __func__,
                      node.tensor.name, ggml_op_desc(&node.tensor));

    return GGML_STATUS_SUCCESS;
}

ggml_backend_hsa_context::ggml_backend_hsa_context(
    const ggml_hsa_device_info::device_info & dev_info) :
    device{dev_info.device}, name{ggml_hsa_format_name(device)} {
    hsa_agent_t agent = dev_info.agent;

    // create queue
    const std::uint32_t min_queue_size = ggml_hsa_get_agent_min_queue_size(agent);
    if (auto status = hsa_queue_create(agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                                       nullptr, 0, 0, &queue);
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
    // were written but never rung.
    ggml_hsa_flush_dispatches(ctx);

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
 * @brief Fuses each padded MUL_MAT's de-pad with a following f32->bf16 convert-CPY.
 *
 * Whole-graph pass (tensor_extra is built per node and cannot see a node's consumers, so this must
 * run once all extras exist). For every MUL_MAT on the padded-GEMM device-transform path whose only
 * consumer is a pure f32->bf16 convert-CPY, builds a de-pad kernel that also narrows to bf16,
 * points it at the CPY's output buffer, and marks the CPY skipped. Opportunistic: if the fused
 * kernel fails to build nothing changes and the graph runs the original de-pad + separate cast.
 * Latched per MUL_MAT via @c fusion.analyzed so the scan runs once, not on every forward pass.
 */
static void ggml_hsa_fuse_mul_mat_narrow(const ggml_backend_hsa_context & ctx,
                                         ggml_cgraph * cgraph,
                                         std::int32_t node_count) {
    const auto & dev_info = ggml_hsa_get_device_info(ctx.device);

    for (std::int32_t i = 0; i < node_count; ++i) {
        ggml_tensor * mm = ggml_graph_node(cgraph, i);
        if (mm->op != GGML_OP_MUL_MAT || mm->extra == nullptr) {
            continue;
        }
        auto & mm_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(mm->extra);
        if (mm_extra.fusion.analyzed) {
            continue;
        }
        mm_extra.fusion.analyzed = true;

        // Only a padded MUL_MAT on the on-device de-pad path is a candidate. Require the full
        // device-transform chain (post-processing + a pre-processing kernel for every buffered
        // source) so use_device_transforms is guaranteed true at dispatch: otherwise the host
        // fallback would write f32 to the parent while the fused (skipped) cast leaves the bf16
        // consumer unwritten. Kernel pointers are fixed after construction, so this holds per pass.
        if (!mm_extra.node.depad || mm_extra.postprocess_kernel == nullptr) {
            continue;
        }
        bool device_transforms = true;
        for (auto src_idx = 0; src_idx < mm_extra.nsrcs; ++src_idx) {
            if (mm_extra.src_nodes[src_idx].buffer_size != 0 &&
                mm_extra.src_nodes[src_idx].preprocess_kernel == nullptr) {
                device_transforms = false;
                break;
            }
        }
        if (!device_transforms) {
            continue;
        }
        // The f32 parent must not be read by anything but the fused consumer.
        if ((mm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
            continue;
        }

        // Find the sole consumer of the MUL_MAT in the graph (graphs here are small).
        ggml_tensor * consumer = nullptr;
        bool multiple = false;
        for (std::int32_t j = 0; j < node_count && !multiple; ++j) {
            ggml_tensor * c = ggml_graph_node(cgraph, j);
            for (auto s = 0; s < GGML_MAX_SRC; ++s) {
                if (c->src[s] == mm) {
                    if (consumer != nullptr && consumer != c) {
                        multiple = true;
                    }
                    consumer = c;
                    break;
                }
            }
        }
        if (multiple || consumer == nullptr || consumer->extra == nullptr) {
            continue;
        }
        // The consumer must be a pure f32->bf16 dtype-convert copy.
        if (!ggml_hsa_is_convert_copy(*consumer) || consumer->src[0]->type != GGML_TYPE_F32 ||
            consumer->type != GGML_TYPE_BF16) {
            continue;
        }

        // Build a de-pad kernel that narrows the padded f32 result to bf16 straight into the
        // consumer's buffer. The distinct (f32 in, bf16 out) dtype pair caches its own PDI.
        auto fused =
            ggml_hsa_build_transform_kernel(dev_info, "HSA_DEPAD", mm_extra.node.tensor, *consumer);
        if (fused == nullptr) {
            continue; // fall back to the original de-pad + separate cast
        }
        mm_extra.postprocess_kernel = std::move(fused);
        mm_extra.fusion.narrow_dst = consumer;
        static_cast<ggml_backend_hsa_tensor_extra *>(consumer->extra)->fusion.skip_dispatch = true;
    }
}

/**
 * @brief Host-path copy from @p src to @p dst, choosing sub-block or plain copy based on @p depad.
 *
 * A padded tensor (@p depad) has a different shape from its counterpart, so the logical sub-block
 * must be scattered/gathered; otherwise the shapes match and a plain layout/dtype copy suffices.
 */
static ggml_status
ggml_hsa_copy_padded_or_plain(const ggml_tensor * src, ggml_tensor * dst, bool depad) {
    return depad ? ggml_hsa_copy_subblock(src, dst) : ggml_hsa_copy_tensor(src, dst);
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
        for (auto src_idx = 0; src_idx < tensor_extra.nsrcs; ++src_idx) {
            if (tensor_extra.src_nodes[src_idx].tensor.data == nullptr) {
                tensor_extra.src_nodes[src_idx].tensor.data = node->src[src_idx]->data;
            }
        }
    }

    // Fuse the per-layer f32->bf16 cast into the MUL_MAT de-pad post-amble. A padded-GEMM MUL_MAT
    // produces an f32 result that HSA_DEPAD strips into the f32 parent, which a following
    // convert-CPY then re-reads to write a bf16 copy for the next op. When the MUL_MAT's sole
    // consumer is exactly that cast, redirect the de-pad to narrow-and-write bf16 straight into the
    // cast's output buffer and skip the cast dispatch, removing a kernel launch and a full [M, N]
    // f32 memory round trip. The ggml graph is left untouched (no tensor type mutated); this is a
    // device-side rewrite only.
    ggml_hsa_fuse_mul_mat_narrow(ctx, cgraph, node_count);

    for (std::int32_t i = 0; (i < node_count) && (status == GGML_STATUS_SUCCESS); ++i) {
        ggml_tensor * node = ggml_graph_node(cgraph, i);

        // early exit if operation does not require a dispatch
        if (ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
            continue;
        }

        auto & tensor_extra = *static_cast<ggml_backend_hsa_tensor_extra *>(node->extra);

        // This node was fused into its producer's de-pad (only possible for a convert-CPY/DUP
        // consumer of a MUL_MAT), which already wrote the narrowed bf16 result here. Running it
        // would read the never-written f32 parent.
        if (tensor_extra.fusion.skip_dispatch) {
            continue;
        }

        switch (node->op) {
            // implemented as host kernels, so no dispatch required
            case GGML_OP_DUP:
            case GGML_OP_CPY:
                // A pure dtype-conversion copy dispatches on-device (no queue drain); other
                // copies fall back to the host path.
                if (tensor_extra.kernel != nullptr) {
                    status = tensor_extra.kernel->dispatch(ctx, node->src, 1, *node);
                } else if (node->op == GGML_OP_DUP) {
                    status = ggml_hsa_compute_dup(ctx, node);
                } else {
                    status = ggml_hsa_compute_cpy(ctx, node);
                }
                continue;
            case GGML_OP_CONT:
                status = ggml_hsa_compute_cont(ctx, node);
                continue;
            default:
                break;
        }

        ggml_tensor & internal_node = tensor_extra.node.tensor;

        // Prefer on-device pre/post-processing kernels when the node provides them: they run on the
        // same in-order queue as the main kernel, so no host queue drain is needed and the packets
        // batch with surrounding work. Otherwise fall back to the host copy path (which must drain
        // first, since the host may not touch a buffer the device is still using). Fixed once at
        // tensor-extra construction time.
        const bool use_device_transforms =
            tensor_extra.sync_mode == ggml_backend_hsa_tensor_extra::sync_mode_t::device;

        if (tensor_extra.sync_mode != ggml_backend_hsa_tensor_extra::sync_mode_t::none) {
            if (!use_device_transforms) {
                ggml_hsa_wait_dispatches(ctx);
            }
            for (auto src_idx = 0; src_idx < tensor_extra.nsrcs; ++src_idx) {
                if (tensor_extra.src_nodes[src_idx].buffer_size == 0) {
                    continue;
                }
                // A constant source (weight/bias) is converted+padded into the persistent internal
                // buffer only once; once cached, its contents are identical on every later
                // dispatch, so skip the pre-processing entirely. Keyed on the parent data pointer
                // so a moved buffer forces a re-conversion.
                if (tensor_extra.src_nodes[src_idx].is_constant) {
                    if (tensor_extra.src_nodes[src_idx].converted_ptr == node->src[src_idx]->data) {
                        continue;
                    }
                    tensor_extra.src_nodes[src_idx].converted_ptr = node->src[src_idx]->data;
                }
                if (use_device_transforms) {
                    // on-device source pre-processing: transform the parent source into its
                    // internal buffer on-queue (e.g. convert+pad), no drain
                    ggml_tensor * preprocess_src = node->src[src_idx];
                    status = tensor_extra.src_nodes[src_idx].preprocess_kernel->dispatch(
                        ctx, &preprocess_src, 1, *internal_node.src[src_idx]);
                } else {
                    // A padded source has a different shape from its parent, so scatter the logical
                    // sub-block into the (pre-zeroed) padded buffer; otherwise the shapes match and
                    // a plain layout/dtype copy suffices.
                    status = ggml_hsa_copy_padded_or_plain(
                        node->src[src_idx], internal_node.src[src_idx], tensor_extra.node.depad);
                }
                if (status != GGML_STATUS_SUCCESS) {
                    GGML_HSA_LOG_ERROR("%s: failed to prepare source %i for tensor \"%s (%s)\"",
                                       __func__, src_idx, node->name, ggml_op_desc(node));
                    break;
                }
            }
            // break out of the node loop on failure so the trailing flush still runs
            if (status != GGML_STATUS_SUCCESS) {
                break;
            }
        }

        if (status = tensor_extra.kernel->dispatch(ctx, internal_node.src, tensor_extra.nsrcs,
                                                   internal_node);
            status != GGML_STATUS_SUCCESS) {
            GGML_HSA_LOG_ERROR("%s: failed to dispatch kernel for tensor \"%s\" (%s)", __func__,
                               node->name, ggml_op_desc(node));
            break;
        }

        if (use_device_transforms) {
            // on-device result post-processing: transform the internal output back into the parent
            // tensor on-queue (e.g. de-pad), no drain. Reads the internal node, writes the parent —
            // or, when the de-pad was fused with a following f32->bf16 cast, narrows straight into
            // that consumer's buffer (skipping the separate cast dispatch).
            ggml_tensor * postprocess_src = &internal_node;
            ggml_tensor & post_dst =
                tensor_extra.fusion.narrow_dst ? *tensor_extra.fusion.narrow_dst : *node;
            status = tensor_extra.postprocess_kernel->dispatch(ctx, &postprocess_src, 1, post_dst);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_HSA_LOG_ERROR("%s: failed to de-pad result for tensor \"%s\" (%s)", __func__,
                                   node->name, ggml_op_desc(node));
                break;
            }
        } else if (tensor_extra.node.convert_dtype || tensor_extra.node.depad) {
            // gather the padded result sub-block back into the parent tensor (de-pad), or convert
            // the datatype in place for the same-shape case
            ggml_hsa_wait_dispatches(ctx);
            status = ggml_hsa_copy_padded_or_plain(&internal_node, node, tensor_extra.node.depad);
            if (status != GGML_STATUS_SUCCESS) {
                GGML_HSA_LOG_ERROR("%s: failed to copy back for tensor \"%s\" (%s)", __func__,
                                   node->name, ggml_op_desc(node));
                break;
            }
        }
    }

    // Flush unconditionally, including on the error paths above: packets written for
    // successfully-dispatched earlier nodes must be rung so their work reaches the device and
    // callers reading via the buffer get/copy paths (which don't synchronize) see complete results.
    ggml_hsa_flush_dispatches(ctx);

    return status;
}

// test support (see ggml-hsa.h)

ggml_status ggml_hsa_test_dispatch_transform(ggml_backend_t backend,
                                             const char * op_name,
                                             const ggml_tensor * src,
                                             ggml_tensor * dst) {
    auto & ctx = *static_cast<ggml_backend_hsa_context *>(backend->context);
    const auto & dev_info = ggml_hsa_get_device_info(ctx.device);

    auto kernel = ggml_hsa_build_transform_kernel(dev_info, op_name, *src, *dst);
    if (kernel == nullptr) {
        GGML_HSA_LOG_ERROR("%s: failed to build transform kernel %s", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    ggml_tensor * dispatch_src = const_cast<ggml_tensor *>(src);
    if (auto status = kernel->dispatch(ctx, &dispatch_src, 1, *dst);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }
    ggml_hsa_wait_dispatches(ctx);
    return GGML_STATUS_SUCCESS;
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
