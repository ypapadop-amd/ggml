// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa.h"
#include "ggml.h"

#include <array>
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
 * @brief Returns the number of sources of @p tensor.
 */
std::int64_t ggml_hsa_nsrcs(const ggml_tensor & tensor);

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
 * @brief Returns a kernel name for @p tensor.
 */
std::string ggml_hsa_create_kernel_name(const ggml_tensor & tensor);

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
                                 ggml_tensor * dst_tensor) const = 0;
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
        std::int32_t device{};                  ///< Device ID.
        hsa_agent_t agent{};                    ///< HSA agent associated with the device.
        hsa_device_type_t type{};               ///< Agent type.
        std::string name;                       ///< Agent name.
        std::vector<ggml_type> supported_types; ///< Device natively supported tensor types.
        memory_pool_info dev_memory{};          ///< Kernel memory pool.
        memory_pool_info kernarg_memory{};      ///< Kernel arguments memory pool.
        memory_pool_info data_memory{};         ///< Data memory pool.
        std::size_t alignment{256};             ///< Memory alignment requirement for buffers.
        std::unordered_map<std::string, std::shared_ptr<ggml_hsa_kernel>>
            kernels; ///< Cached device kernels.
    };

    std::array<device_info, GGML_HSA_MAX_DEVICES> devices = {};
};

/**
 * @brief Returns the HSA device information.
 *
 * This function returns a reference to a structure containing the HSA device
 * information. HSA and the information is initialized once and reused on all
 * subsequent calls.
 *
 * @return structure with device information
 */
const ggml_hsa_device_info & ggml_hsa_info();

/**
 * @brief Returns the device info associated with @p device_id.
 */
const ggml_hsa_device_info::device_info & ggml_hsa_get_device_info(std::int32_t device_id);

/**
 * @brief Tensor extra information.
 */
struct ggml_backend_hsa_tensor_extra {
    std::shared_ptr<ggml_hsa_kernel> kernel;     ///< Kernel associated with the tensor.
    std::int64_t nsrcs{};                        ///< Number of source tensors.
    ggml_tensor tensor{};                        ///< Transformed operation tensor.
    std::array<ggml_tensor, GGML_MAX_SRC> src{}; ///< Source tensors for the operation.
    std::array<std::size_t, GGML_MAX_SRC>
        src_sizes{};                       ///< Sizes of the source tensors in bytes. 0 for no copy.
    std::size_t total_src_size{};          ///< Total size of the source tensors in bytes.
    ggml_hsa_unique_ptr<std::byte> buffer; ///< Temporary buffer for the tensor data.

    ggml_backend_hsa_tensor_extra(const ggml_hsa_device_info::device_info & dev_info,
                                  const ggml_tensor * parent_tensor);
    ggml_backend_hsa_tensor_extra(const ggml_backend_hsa_tensor_extra &) = delete;
    ggml_backend_hsa_tensor_extra(ggml_backend_hsa_tensor_extra &&) = delete;

    ~ggml_backend_hsa_tensor_extra() = default;

    ggml_backend_hsa_tensor_extra & operator=(const ggml_backend_hsa_tensor_extra &) = delete;
    ggml_backend_hsa_tensor_extra & operator=(ggml_backend_hsa_tensor_extra &&) = delete;

    /**
     * @brief Allocates storage for the internal tensor.
     */
    ggml_status allocate_internal_storage(const ggml_hsa_device_info::device_info & dev_info);

    /**
     * @brief Returns if one or more input tensors need to be copied..
     */
    bool has_input_tensor_copies() const { return total_src_size > 0; }
};

/**
 * @brief Context for HSA backend operations.
 */
struct ggml_backend_hsa_context {
    std::int32_t device{};          ///< Device ID.
    std::string name;               ///< Device name.
    hsa_queue_t * queue{};          ///< HSA queue.
    hsa_signal_t dispatch_signal{}; ///< Signal for packet completion.
    std::vector<ggml_hsa_unique_ptr<void>>
        pending_payloads; ///< Packet payloads since last synchronization.

    explicit ggml_backend_hsa_context(const ggml_hsa_device_info::device_info & dev_info);

    ggml_backend_hsa_context(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context(ggml_backend_hsa_context &&) = delete;

    ~ggml_backend_hsa_context();

    ggml_backend_hsa_context & operator=(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context & operator=(ggml_backend_hsa_context &&) = delete;

    /**
     * @brief Frees all memory associated with pending packets.
     *
     * @warning This function assumes that packets have been processed.
     */
    void free_pending_payloads();
};

/**
 * @brief Waits for all dispatched kernels to finish.
 *
 * @param[in] ctx backend context
 */
void ggml_hsa_wait_dispatches(ggml_backend_hsa_context & ctx);
