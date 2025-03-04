#pragma once

#include "ggml.h"
#include "ggml-hsa.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "ggml-common.h"

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

/**
 * @brief Returns the description of @p status as a string.
 */
const char* ggml_hsa_get_status_string(hsa_status_t status);

/**
 * @brief Prints an error message based on the status and aborts.
 *
 * @param stmt statement that caused the error
 * @param func function in which the error occurred
 * @param file file in which the error occurred
 * @param line line number where the error occurred
 * @param status error code
 */
[[noreturn]]
void ggml_hsa_error(const char * stmt, const char * func, const char * file, int line, hsa_status_t status);

#define HSA_CHECK(status)              \
  do {                                 \
    auto status_ = (status);           \
    if (status_ != HSA_STATUS_SUCCESS) \
      ggml_hsa_error(                  \
          #status,                     \
          __func__,                    \
          __FILE__,                    \
          __LINE__,                    \
          status_);                    \
  } while (false)

#define HSA_CHECK_THROW(status)                                      \
  do {                                                               \
    auto status_ = (status);                                         \
    if (status_ != HSA_STATUS_SUCCESS)                               \
      throw std::runtime_error{ggml_hsa_get_status_string(status_)}; \
  } while (false)

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
};

/**
 * @brief Instructions buffer.
 */
struct ggml_hsa_insts_buffer {
    std::uint32_t * data{};
    std::size_t size{};
};

/**
 * @brief AIE agent kernel.
 */
struct ggml_hsa_aie_kernel {
    ggml_hsa_pdi_buffer pdi_buffer;
    ggml_hsa_insts_buffer insts_buffer;
};

/**
 * @brief Context for HSA backend operations.
 */
struct ggml_backend_hsa_context {
    std::int32_t device{};                                            ///< Device ID.
    std::string name;                                                 ///< Device name.
    hsa_queue_t* queue{};                                             ///< HSA queue.
    hsa_signal_t dispatch_signal{};                                   ///< Signal to wait for dispatches.
    std::unordered_map<std::string, ggml_hsa_aie_kernel> aie_kernels; ///< AIE agent kernels.
#ifdef GGML_HSA_CPU_FALLBACK
    ggml_backend_t fallback_backend{}; ///< Fallback backend for operations not supported by HSA.
    ggml_gallocr_t fallback_galloc{};  ///< Fallback graph allocator.
#endif

    ggml_backend_hsa_context(std::int32_t device, const ggml_hsa_device_info::device_info& dev_info);

    ggml_backend_hsa_context(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context(ggml_backend_hsa_context &&) = delete;

    ~ggml_backend_hsa_context();

    ggml_backend_hsa_context& operator=(const ggml_backend_hsa_context &) = delete;
    ggml_backend_hsa_context& operator=(ggml_backend_hsa_context &&) = delete;

    /**
     * @brief Destroys all stored AIE kernels.
     */
    void destroy_aie_kernels();
};
