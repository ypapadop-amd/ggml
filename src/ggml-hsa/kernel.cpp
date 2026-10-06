// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel.hpp"

#include <fcntl.h>
#include <unistd.h>

#include "ggml-impl.h"

ggml_hsa_kernel::~ggml_hsa_kernel() {
    if (m_executable.handle != 0) {
        GGML_HSA_CHECK_WARN(hsa_executable_destroy(m_executable));
    }
    if (m_reader.handle != 0) {
        GGML_HSA_CHECK_WARN(hsa_code_object_reader_destroy(m_reader));
    }
}

ggml_status ggml_hsa_kernel::load(hsa_agent_t agent,
                                  const std::filesystem::path & path,
                                  const std::string & kernel_name) {
    // the reader maps the file, so the descriptor is not needed once the reader exists
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        GGML_HSA_LOG_ERROR("%s: could not open file %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    const hsa_status_t reader_status = hsa_code_object_reader_create_from_file(fd, &m_reader);
    close(fd);
    if (reader_status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not read code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(reader_status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_create_alt(
            HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &m_executable);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not create executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status =
            hsa_executable_load_agent_code_object(m_executable, agent, m_reader, nullptr, nullptr);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not load code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_freeze(m_executable, nullptr); status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not freeze executable for %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    hsa_executable_symbol_t symbol{};
    if (auto status =
            hsa_executable_get_symbol_by_name(m_executable, kernel_name.c_str(), &agent, &symbol);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: kernel %s not found in %s (%s)", __func__, kernel_name.c_str(),
                           path.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &m_kernel_object);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernel object of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_symbol_get_info(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &m_kernarg_size);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not get kernarg size of %s (%s)", __func__,
                           kernel_name.c_str(), ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}
