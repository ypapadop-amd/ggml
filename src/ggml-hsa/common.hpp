// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

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
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "ggml-common.h"

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
#define HSA_CHECK_ABORT(status)                                                                    \
    do {                                                                                           \
        auto status_ = (status);                                                                   \
        if (status_ != HSA_STATUS_SUCCESS)                                                         \
            ggml_hsa_error(#status, __func__, __FILE__, __LINE__, status_);                        \
    } while (false)

/**
 * @brief Checks if @p status is an error code and throws an exception.
 */
#define HSA_CHECK_THROW(status)                                                                    \
    do {                                                                                           \
        auto status_ = (status);                                                                   \
        if (status_ != HSA_STATUS_SUCCESS)                                                         \
            throw std::runtime_error{ggml_hsa_get_status_string(status_)};                         \
    } while (false)

/**
 * @brief Decomposes a 64-bit address to two @c std::uint32_t.
 */
inline std::tuple<std::uint32_t, std::uint32_t> ggml_hsa_addr_to_hilo(void * address) {
    static_assert(sizeof(void *) == 2 * sizeof(std::uint32_t));
    return {reinterpret_cast<uint64_t>(address) >> 32,
            reinterpret_cast<uint64_t>(address) & 0xFFFFFFFF};
}

/**
 * @brief Returns the number of sources of the tensor.
 *
 * @param[in] tensor tensor to find number of sources for
 */
int64_t ggml_hsa_nsrcs(const ggml_tensor * tensor);

/**
 * @brief Returns the full path to this library.
 */
const std::filesystem::path & ggml_hsa_library_path();

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
        hsa_agent_t agent{};               ///< HSA agent associated with the device.
        hsa_device_type_t type{};          ///< Agent type.
        std::string name;                  ///< Agent name.
        memory_pool_info dev_memory{};     ///< Pool for kernels.
        memory_pool_info kernarg_memory{}; ///< Pool for kernel arguments.
        memory_pool_info data_memory{};    ///< Pool for data.
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
 * @brief PDI buffer.
 */
struct ggml_hsa_pdi_buffer {
    std::uint64_t * data{};
    std::size_t size{};

    bool is_valid() const { return data != nullptr; }
};

/**
 * @brief Instructions buffer.
 */
struct ggml_hsa_insts_buffer {
    std::uint32_t * data{};
    std::size_t size{};

    bool is_valid() const { return data != nullptr; }
};

/**
 * @brief AIE agent kernel.
 */
struct ggml_hsa_aie_kernel {
    ggml_hsa_pdi_buffer pdi;
    ggml_hsa_insts_buffer insts;

    bool is_valid() const {
        assert(pdi.is_valid() == insts.is_valid());
        return pdi.is_valid();
    }
};

#ifdef GGML_HSA_CPU_FALLBACK
struct ggml_backend_hsa_emulated_tensor;
#endif

/**
 * @brief Tensor extra information.
 */
struct ggml_backend_hsa_tensor_extra {
    ggml_hsa_aie_kernel kernel; ///< Kernel associated with this tensor.
#ifdef GGML_HSA_CPU_FALLBACK
    std::unique_ptr<ggml_backend_hsa_emulated_tensor> emulated_tensor;
#endif
};

/**
 * @brief Context for HSA backend operations.
 */
struct ggml_backend_hsa_context {
    std::int32_t device{};          ///< Device ID.
    std::string name;               ///< Device name.
    hsa_queue_t * queue{};          ///< HSA queue.
    hsa_signal_t dispatch_signal{}; ///< Signal to wait dispatches.
    std::unordered_map<std::string, ggml_hsa_aie_kernel> aie_kernels; ///< AIE agent kernels.
    std::unordered_set<std::string> blocked_aie_kernels; ///< Blocked AIE agent kernels.
    std::vector<void *> pending_payloads; ///< Packet payloads since last synchronization.
#ifdef GGML_HSA_CPU_FALLBACK
    ggml_backend_t fallback_backend{}; ///< Fallback backend for operations not supported by HSA.
    ggml_gallocr_t fallback_galloc{};  ///< Fallback graph allocator.
#endif

    ggml_backend_hsa_context(std::int32_t device,
                             const ggml_hsa_device_info::device_info & dev_info);

    ggml_backend_hsa_context(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context(ggml_backend_hsa_context &&) = delete;

    ~ggml_backend_hsa_context();

    ggml_backend_hsa_context & operator=(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context & operator=(ggml_backend_hsa_context &&) = delete;

    /**
     * @brief Destroys all loaded kernels and frees the used memory.
     */
    void destroy_kernels();

    /**
     * @brief Frees all memory associated with pending packets.
     *
     * @warning This function assumes that packets have been processed.
     */
    void free_pending_payloads();
};

/**
 * @brief Dispatches a kernel that implements for the tensor operation.
 *
 * @param[in] ctx backend context
 * @param[in] tensor tensor with the operation to dispatch for
 */
ggml_status ggml_hsa_dispatch_kernel(ggml_backend_hsa_context & ctx, ggml_tensor * tensor);

/**
 * @brief Creates a string representation of the tensor shape.
 *
 * The representation is of the form `3x3x4` for a 3D tensor with dimensions `[3,3,4]`.
 *
 * @param[in] tensor tensor to output shape for
 * @param[out] os output stream
 * @param[in] delim delimiter
 */
template <typename OutputStream>
void ggml_hsa_output_tensor_shape(const ggml_tensor * tensor, OutputStream & os, char delim = 'x') {
    const auto ndims = ggml_n_dims(tensor);
    os << tensor->ne[0];
    for (int i = 1; i < ndims; ++i) {
        os << delim << tensor->ne[i];
    }
}

/**
 * @brief Creates a string representation of the tensor.
 *
 * The representation is of the form `DimsDatatypeModifiers`, e.g., `3x3x4f32npt` for a 3D tensor
 * with dimensions `[3,3,4]` that is non-contiguous, is permuted, and transposed.
 *
 * @param[in] tensor tensor to output
 * @param[out] os output stream
 */
template <typename OutputStream>
void ggml_hsa_output_tensor(const ggml_tensor * tensor, OutputStream & os) {
    ggml_hsa_output_tensor_shape(tensor, os);

    os << ggml_type_name(tensor->type);

    // modifiers
    if (!ggml_is_contiguous(tensor)) {
        os << 'n';
    }
    if (ggml_is_permuted(tensor)) {
        os << 'p';
    }
    if (ggml_is_transposed(tensor)) {
        os << 't';
    }
}
