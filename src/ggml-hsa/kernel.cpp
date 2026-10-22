// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/kernel.hpp"

#include <cerrno>
#include <cstring>
#include <fstream>

#include "ggml-impl.h"

ggml_hsa_kernel::~ggml_hsa_kernel() {
    if (m_executable.handle != 0) {
        GGML_HSA_CHECK_WARN(hsa_executable_destroy(m_executable));
    }
}

ggml_status ggml_hsa_kernel::load_executable(hsa_agent_t agent,
                                             const std::filesystem::path & path,
                                             const std::string & kernel_name,
                                             std::vector<char> & blob) {
    // single-use: a second call would leak the first executable rather than replace it
    if (m_executable.handle != 0) {
        GGML_HSA_LOG_ERROR("%s: kernel %s is already loaded", __func__, kernel_name.c_str());
        return GGML_STATUS_FAILED;
    }

    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) {
        GGML_HSA_LOG_ERROR("%s: could not open file %s (%s)", __func__, path.c_str(),
                           std::strerror(errno));
        return GGML_STATUS_FAILED;
    }
    const std::streamoff file_size = is.tellg();
    if ((file_size <= 0) || !is.seekg(0, std::ios::beg)) {
        GGML_HSA_LOG_ERROR("%s: could not get file size for %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }
    blob.resize(static_cast<std::size_t>(file_size));
    if (!is.read(blob.data(), file_size)) {
        GGML_HSA_LOG_ERROR("%s: failed to read %s", __func__, path.c_str());
        return GGML_STATUS_FAILED;
    }

    hsa_code_object_reader_t reader{};
    if (auto status = hsa_code_object_reader_create_from_memory(blob.data(), blob.size(), &reader);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not read code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_create_alt(
            HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &m_executable);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_LOG_ERROR("%s: could not create executable (%s)", __func__,
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status =
            hsa_executable_load_agent_code_object(m_executable, agent, reader, nullptr, nullptr);
        status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_CHECK_WARN(hsa_executable_destroy(m_executable));
        m_executable = {};
        GGML_HSA_LOG_ERROR("%s: could not load code object %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    if (auto status = hsa_executable_freeze(m_executable, nullptr); status != HSA_STATUS_SUCCESS) {
        hsa_code_object_reader_destroy(reader);
        GGML_HSA_CHECK_WARN(hsa_executable_destroy(m_executable));
        m_executable = {};
        GGML_HSA_LOG_ERROR("%s: could not freeze executable for %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }

    // The reader is only needed to feed the loader above; once the executable is frozen, the code
    // object's contents have been fully ingested and the reader is no longer needed.
    hsa_code_object_reader_destroy(reader);

    return GGML_STATUS_SUCCESS;
}
