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

    // Each hsaco holds exactly one kernel, found by kind rather than by name: a kernel packed from
    // a full ELF is named by the ELF (e.g. "main:sequence"), not after the hsaco.
    struct kernel_search {
        hsa_executable_symbol_t symbol{};
        std::size_t count = 0;
    } search;
    if (auto status = hsa_executable_iterate_agent_symbols(
            m_executable, agent,
            [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t symbol, void * data) {
                hsa_symbol_kind_t kind{};
                if (auto status = hsa_executable_symbol_get_info(
                        symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
                    status != HSA_STATUS_SUCCESS) {
                    return status;
                }
                if (kind == HSA_SYMBOL_KIND_KERNEL) {
                    auto & search = *static_cast<kernel_search *>(data);
                    search.symbol = symbol;
                    ++search.count;
                }
                return HSA_STATUS_SUCCESS;
            },
            &search);
        status != HSA_STATUS_SUCCESS) {
        GGML_HSA_LOG_ERROR("%s: could not list the kernels in %s (%s)", __func__, path.c_str(),
                           ggml_hsa_get_status_string(status));
        return GGML_STATUS_FAILED;
    }
    if (search.count != 1) {
        GGML_HSA_LOG_ERROR("%s: %s holds %zu kernels, expected exactly one for kernel %s", __func__,
                           path.c_str(), search.count, kernel_name.c_str());
        return GGML_STATUS_FAILED;
    }
    const hsa_executable_symbol_t symbol = search.symbol;

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
