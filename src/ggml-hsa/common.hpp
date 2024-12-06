#pragma once

#include "ggml.h"
#include "ggml-hsa.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "ggml-common.h"

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

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

#define HSA_CHECK(err)              \
  do {                              \
    auto err_ = (err);              \
    if (err_ != HSA_STATUS_SUCCESS) \
      ggml_hsa_error(               \
          #err,                     \
          __func__,                 \
          __FILE__,                 \
          __LINE__,                 \
          err_);                    \
  } while (0)

struct ggml_hsa_device_info {
    std::int32_t device_count{}; ///< Number of devices, up to @ref GGML_HSA_MAX_DEVICES.

    /**
     * @brief Information about a single HSA memory pool.
     */
    struct hsa_memory_pool_info {
        hsa_amd_memory_pool_t memory_pool{}; ///< HSA memory pool object.
        std::size_t alignment{};             ///< Memory pool alignment.
        std::size_t max_size{};              ///< Memory pool maximum allocation.
    };

    /**
     * @brief Information about a single HSA device.
     */
    struct hsa_device_info {
        hsa_agent_t agent{};                  ///< HSA agent associated with the device.
        hsa_device_type_t type{};             ///< Agent type.
        hsa_memory_pool_info data_memory;     ///< Pool for data.
        hsa_memory_pool_info kernarg_memory;  ///< Pool for kernel arguments.
    };

    std::array<hsa_device_info, GGML_HSA_MAX_DEVICES> devices = {};
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
 * @brief Context for HSA backend operations.
 */
struct ggml_backend_hsa_context {
    std::int32_t device; ///< Device ID.
    hsa_agent_t agent;   ///< HSA agent associated with the device.
    std::string name;    ///< Device name.

    ggml_backend_hsa_context(std::int32_t device, hsa_agent_t agent);
};
