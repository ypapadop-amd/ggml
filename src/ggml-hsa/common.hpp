#pragma once

#include "ggml.h"
#include "ggml-hsa.h"

#include <string>

#include <hsa/hsa.h>

#include "ggml-common.h"

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

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
    int device_count;
};

const ggml_hsa_device_info & ggml_hsa_info();

struct ggml_backend_hsa_context {
    int device;
    std::string name;

    explicit ggml_backend_hsa_context(int device) :
        device(device),
        name(GGML_HSA_NAME + std::to_string(device)) {
    }
};
