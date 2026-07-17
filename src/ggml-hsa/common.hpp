// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa.h"
#include "ggml.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "ggml-common.h"

#if defined(__clang__) || defined(__GNUC__)
// Optimize for the execution path that is more or less likely than the alternative.
#define LIKELY(ex) __builtin_expect(!!(ex), 1)
#define UNLIKELY(ex) __builtin_expect(!!(ex), 0)
#else
#define LIKELY(ex) (ex)
#define UNLIKELY(ex) (ex)
#endif

/// @brief @c true if logging is enabled.
extern bool g_ggml_hsa_verbose;

/**
 * @brief Logs an error when verbose logging is enabled (@ref g_ggml_hsa_verbose).
 */
#define GGML_HSA_LOG_ERROR(MSG, ...)                                                               \
    do {                                                                                           \
        if (UNLIKELY(g_ggml_hsa_verbose))                                                          \
            GGML_LOG_ERROR(MSG "\n", __VA_ARGS__);                                                 \
    } while (false)

/**
 * @brief Logs a warning when verbose logging is enabled (@ref g_ggml_hsa_verbose).
 */
#define GGML_HSA_LOG_WARN(MSG, ...)                                                                \
    do {                                                                                           \
        if (UNLIKELY(g_ggml_hsa_verbose))                                                          \
            GGML_LOG_WARN(MSG "\n", __VA_ARGS__);                                                  \
    } while (false)

/**
 * @brief Logs an informational message when verbose logging is enabled (@ref g_ggml_hsa_verbose).
 */
#define GGML_HSA_LOG_INFO(MSG, ...)                                                                \
    do {                                                                                           \
        if (UNLIKELY(g_ggml_hsa_verbose))                                                          \
            GGML_LOG_INFO(MSG "\n", __VA_ARGS__);                                                  \
    } while (false)

/**
 * @brief Returns if @p s evaluates to `true` or `false`.
 */
bool ggml_hsa_string_to_bool(std::string_view s);

/**
 * @brief Returns the description of @p status as a string.
 */
const char * ggml_hsa_get_status_string(hsa_status_t status);

/**
 * @brief Prints an error message based on the status and aborts.
 *
 * @param[in] stmt statement that caused the error
 * @param[in] func function in which the error occurred
 * @param[in] file file in which the error occurred
 * @param[in] line line number where the error occurred
 * @param[in] status error code
 */
[[noreturn]]
void ggml_hsa_error(
    const char * stmt, const char * func, const char * file, int line, hsa_status_t status);

/**
 * @brief Checks if @p status is an error code, prints an error message and aborts.
 */
#define GGML_HSA_CHECK_ABORT(status)                                                               \
    do {                                                                                           \
        auto status_ = (status);                                                                   \
        if (status_ != HSA_STATUS_SUCCESS)                                                         \
            ggml_hsa_error(#status, __func__, __FILE__, __LINE__, status_);                        \
    } while (false)

/**
 * @brief Checks if @p status is an error code and throws an exception.
 */
#define GGML_HSA_CHECK_THROW(status)                                                               \
    do {                                                                                           \
        auto status_ = (status);                                                                   \
        if (status_ != HSA_STATUS_SUCCESS)                                                         \
            throw std::runtime_error{ggml_hsa_get_status_string(status_)};                         \
    } while (false)

/**
 * @brief Returns the number of sources of @p tensor including holes (null sources).
 */
std::int32_t ggml_hsa_nsrcs(const ggml_tensor & tensor);

/**
 * @brief Creates a string representation of the tensor shape.
 *
 * For a 3D tensor with dimensions `[3,3,4,1]`, the default representation is of the form `3x3x4`.
 *
 * @param[in] tensor tensor to output shape for
 * @param[out] os output stream
 * @param[in] delim delimiter
 */
template <typename OutputStream>
void ggml_hsa_output_tensor_shape(const ggml_tensor & tensor, OutputStream & os, char delim = 'x') {
    const auto ndims = ggml_n_dims(&tensor);
    os << tensor.ne[0];
    for (std::int32_t i = 1; i < ndims; ++i) {
        os << delim << tensor.ne[i];
    }
}

/**
 * @brief Creates a string representation of the tensor stride.
 *
 * For a 3D tensor with dimensions `[3,3,4,1]`, the default representation is of the form `X,Y,Z`,
 * where X, Y, Z are the stride in bytes in the first, second, and third dimensions, respectively.
 *
 * @param[in] tensor tensor to output stride for
 * @param[out] os output stream
 * @param[in] delim delimiter
 */
template <typename OutputStream>
void ggml_hsa_output_tensor_stride(const ggml_tensor & tensor,
                                   OutputStream & os,
                                   char delim = ',') {
    const auto ndims = ggml_n_dims(&tensor);
    os << tensor.nb[0];
    for (std::int32_t i = 1; i < ndims; ++i) {
        os << delim << tensor.nb[i];
    }
}

/**
 * @brief Creates a string representation of the tensor.
 *
 * The representation is of the form `DimsDatatypeModifiers`, e.g., `3x3x4f32` for a contiguous 3D
 * tensor with dimensions `[3,3,4]`.
 *
 * @param[in] tensor tensor to output
 * @param[out] os output stream
 */
template <typename OutputStream>
void ggml_hsa_output_tensor(const ggml_tensor & tensor, OutputStream & os) {
    ggml_hsa_output_tensor_shape(tensor, os);
    os << ggml_type_name(tensor.type);
    if (!ggml_is_contiguous(&tensor)) {
        os << 'n';
    }
}

/**
 * @brief Creates a string representation of the tensor's op_params using a hash.
 *
 * @param[in] tensor tensor to output
 * @param[out] os output stream
 */
template <typename OutputStream>
void ggml_hsa_encode_op_params(const ggml_tensor & tensor, OutputStream & os) {
    std::string_view bytes(reinterpret_cast<const char *>(tensor.op_params), GGML_MAX_OP_PARAMS);
    std::size_t hash_value = std::hash<std::string_view>{}(bytes);
    os << std::hex << hash_value;
}

/**
 * @brief Frees memory allocated using HSA.
 */
template <typename T>
struct ggml_hsa_delete {
    static_assert(!std::is_array_v<T>, "ggml_hsa_delete does not support arrays");

    void operator()(T * ptr) const {
        if (ptr) {
            if constexpr (!std::is_void_v<T>) {
                std::destroy_at(ptr);
            }
            GGML_HSA_CHECK_ABORT(hsa_amd_memory_pool_free(ptr));
        }
    }
};

/// @brief HSA allocated managed memory.
template <typename T>
using ggml_hsa_unique_ptr = std::unique_ptr<T, ggml_hsa_delete<T>>;

/**
 * @brief Fixed-slot kernarg pool over a single HSA memory-pool buffer.
 *
 * The buffer is carved into @p slot_count equally sized, aligned slots at construction. @ref slot
 * returns the slot address with no HSA call on the access path. Slot lifetime is managed by the
 * caller: each slot maps to the queue ring slot of the same index, reused only once the device has
 * consumed that ring slot. Sized for one worst-case kernarg region per ring slot.
 */
class ggml_hsa_kernarg_pool {
  public:
    ggml_hsa_kernarg_pool() = default;

    /**
     * @brief Constructs a pool of @p slot_count slots, each at least @p slot_size bytes.
     *
     * @param[in] memory_pool HSA memory pool to allocate the backing buffer from
     * @param[in] slot_count number of fixed slots (one per HSA queue ring slot)
     * @param[in] slot_size minimum size of each slot in bytes
     * @param[in] alignment alignment applied to the start of every slot; must be a power of two
     * @throws std::invalid_argument if @p alignment is not a power of two
     * @throws std::runtime_error if the backing buffer cannot be allocated
     */
    ggml_hsa_kernarg_pool(hsa_amd_memory_pool_t memory_pool,
                          std::size_t slot_count,
                          std::size_t slot_size,
                          std::size_t alignment) :
        m_slot_count{slot_count} {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument{"Kernarg pool alignment must be a power of two"};
        }
        // Pad each slot to alignment so every slot start is aligned.
        m_slot_size = GGML_PAD(slot_size, alignment);
        void * buffer = nullptr;
        if (auto status =
                hsa_amd_memory_pool_allocate(memory_pool, m_slot_size * m_slot_count, 0, &buffer);
            status != HSA_STATUS_SUCCESS) {
            throw std::runtime_error{std::string("Could not allocate kernarg pool buffer (")
                                         .append(ggml_hsa_get_status_string(status))
                                         .append(")")};
        }
        m_buffer.reset(buffer);
    }

    /**
     * @brief Returns the address of slot @p index.
     *
     * @param[in] index slot index
     */
    void * slot(std::size_t index) const {
        assert(index < m_slot_count);
        return static_cast<std::byte *>(m_buffer.get()) + index * m_slot_size;
    }

  private:
    ggml_hsa_unique_ptr<void> m_buffer; ///< Backing storage.
    std::size_t m_slot_size{};          ///< Size of each slot in bytes (padded to alignment).
    std::size_t m_slot_count{};         ///< Number of slots.
};

struct ggml_backend_hsa_context;

/**
 * @brief Base class for HSA kernels.
 */
class ggml_hsa_kernel {
  public:
    virtual ~ggml_hsa_kernel() = default;

    /**
     * @brief Dispatches the kernel.
     *
     * @param[in] ctx backend context
     * @param[in] src_tensors source tensors
     * @param[in] num_src_tensors number of source tensors
     * @param[out] dst_tensor destination tensor
     */
    virtual ggml_status dispatch(ggml_backend_hsa_context & ctx,
                                 ggml_tensor * src_tensors[],
                                 std::size_t num_src_tensors,
                                 ggml_tensor & dst_tensor) const = 0;
};

/**
 * @brief Device information.
 */
struct ggml_hsa_device_info {
    std::int32_t device_count{}; ///< Number of devices, up to @ref GGML_HSA_MAX_DEVICES.

    /**
     * @brief Information about a single HSA memory pool.
     */
    struct memory_pool_info {
        hsa_amd_memory_pool_t memory_pool{}; ///< HSA memory pool object.
        std::size_t size{};                  ///< Memory available to the pool in bytes.
        std::size_t alignment{};             ///< Memory pool alignment.
        std::size_t max_alloc_size{};        ///< Memory pool maximum allocation.
    };

    /**
     * @brief Information about a single HSA device.
     */
    struct device_info {
        std::int32_t device{};             ///< Device ID.
        hsa_agent_t agent{};               ///< HSA agent associated with the device.
        hsa_device_type_t type{};          ///< Agent type.
        std::string name;                  ///< Agent name.
        memory_pool_info dev_memory{};     ///< Kernel memory pool.
        memory_pool_info kernarg_memory{}; ///< Kernel arguments memory pool.
        memory_pool_info data_memory{};    ///< Data memory pool.
        std::size_t alignment{64};         ///< Memory alignment requirement for buffers.
        bool substitute_fp16_bf16{false};  ///< Use BF16 when FP16 is requested.
        std::unordered_map<std::string, std::shared_ptr<ggml_hsa_kernel>>
            kernels; ///< Cached device kernels.
    };

    std::array<device_info, GGML_HSA_MAX_DEVICES> devices = {};
};

/**
 * @brief Returns the HSA device information, initialized once and reused on subsequent calls.
 */
const ggml_hsa_device_info & ggml_hsa_info();

/**
 * @brief Returns the device info associated with @p device_id.
 */
const ggml_hsa_device_info::device_info & ggml_hsa_get_device_info(std::int32_t device_id);

/**
 * @brief Tensor metadata.
 *
 * Holds metadata about a parent ggml_tensor used by the HSA backend to build an alternative graph
 * representation for run-time use. Copies are made of the parent and its source tensors' metadata,
 * with transformations applied (e.g., making them contiguous, flattening).
 */
struct ggml_backend_hsa_tensor_extra {
    /// @brief Internal graph node.
    struct node_t {
        ggml_tensor tensor{};      ///< Transformed tensor.
        std::size_t buffer_size{}; ///< Temporary storage size in bytes.
        bool convert_dtype{};      ///< True if data conversion is necessary.
        bool depad{};              ///< True if the transformed tensor is zero-padded and must be
                                   ///< copied to/from the (smaller) parent tensor sub-block.
    };

    /// @brief Number of source tensors.
    std::int32_t nsrcs{};
    /// @brief Internal graph node.
    node_t node{};
    /// @brief Internal graph node sources, including holes for null sources.
    std::array<node_t, GGML_MAX_SRC> src_nodes{};
    /// @brief Kernel associated with the tensor.
    std::shared_ptr<ggml_hsa_kernel> kernel;
    /// @brief Temporary storage for tensor data, allocated if the kernel requires an intermediate
    /// buffer.
    ggml_hsa_unique_ptr<std::byte> buffer;
    /// @brief True if synchronization before and after the kernel is required, e.g., if host-based
    /// transformations are necessary.
    bool requires_sync{false};
    /// @brief Optional on-device pre-processing kernel per source: transforms a parent source
    /// tensor into its internal buffer (e.g., dtype conversion and/or zero-padding) on the device
    /// queue instead of on the host. Entry is null when a source needs no on-device pre-processing.
    std::array<std::shared_ptr<ggml_hsa_kernel>, GGML_MAX_SRC> src_preprocess_kernels{};
    /// @brief Optional on-device post-processing kernel for the result: transforms the internal
    /// output buffer back into the parent tensor (e.g., de-padding and/or dtype conversion) on the
    /// device queue. Null when the output needs no on-device post-processing.
    std::shared_ptr<ggml_hsa_kernel> postprocess_kernel;
    /// @brief Optional on-device kernel for a pure dtype-conversion CPY/DUP node: casts the single
    /// source into this tensor on the device queue (no host drain). Null for copies handled on the
    /// host (strided, reshape, or same-dtype).
    std::shared_ptr<ggml_hsa_kernel> convert_copy_kernel;
    /// @brief Per source: true if the source is a graph-constant leaf (e.g. a weight or bias) whose
    /// converted/padded contents can be cached in the (persistent) internal buffer and reused across
    /// dispatches instead of re-running the pre-processing every time.
    std::array<bool, GGML_MAX_SRC> src_is_constant{};
    /// @brief Per source: the parent data pointer whose converted contents currently sit in the
    /// internal buffer. Null until the first conversion. The pre-processing is skipped while this
    /// matches the parent's data pointer (constant sources only), guarding against a moved buffer.
    std::array<const void *, GGML_MAX_SRC> src_converted_ptr{};

    ggml_backend_hsa_tensor_extra(const ggml_hsa_device_info::device_info & dev_info,
                                  const ggml_tensor & parent_tensor);
    ggml_backend_hsa_tensor_extra(const ggml_backend_hsa_tensor_extra &) = delete;
    ggml_backend_hsa_tensor_extra(ggml_backend_hsa_tensor_extra &&) = delete;

    ~ggml_backend_hsa_tensor_extra() = default;

    ggml_backend_hsa_tensor_extra & operator=(const ggml_backend_hsa_tensor_extra &) = delete;
    ggml_backend_hsa_tensor_extra & operator=(ggml_backend_hsa_tensor_extra &&) = delete;

    /**
     * @brief Allocates storage for the internal tensor.
     */
    ggml_status allocate_internal_storage(const ggml_hsa_device_info::device_info & dev_info);
};

/**
 * @brief Context for HSA backend operations.
 */
struct ggml_backend_hsa_context {
    std::int32_t device{};              ///< Device ID.
    std::string name;                   ///< Device name.
    hsa_queue_t * queue{};              ///< HSA queue.
    hsa_signal_t dispatch_signal{};     ///< Signal for packet completion.
    ggml_hsa_kernarg_pool kernargs;     ///< Per-ring-slot kernarg buffers for in-flight packets.
    std::size_t dispatch_batch_size{1}; ///< Packets accumulated before the doorbell is rung.
    std::size_t n_batched{};            ///< Packets written since the last doorbell ring.

    explicit ggml_backend_hsa_context(const ggml_hsa_device_info::device_info & dev_info);

    ggml_backend_hsa_context(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context(ggml_backend_hsa_context &&) = delete;

    ~ggml_backend_hsa_context();

    ggml_backend_hsa_context & operator=(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context & operator=(ggml_backend_hsa_context &&) = delete;
};

/**
 * @brief Waits for all dispatched kernels to finish.
 *
 * @param[in] ctx backend context
 */
void ggml_hsa_wait_dispatches(ggml_backend_hsa_context & ctx);

/**
 * @brief Rings the doorbell for any packets accumulated since the last ring.
 *
 * Packets are written eagerly but the doorbell only rings once
 * @ref ggml_backend_hsa_context::dispatch_batch_size packets accumulate, or a synchronization
 * point forces a flush. Submits pending packets for processing; does @e not wait for completion.
 *
 * @param[in] ctx backend context
 */
void ggml_hsa_flush_dispatches(ggml_backend_hsa_context & ctx);
