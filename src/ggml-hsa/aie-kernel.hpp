// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <utility>

/**
 * @brief Buffer for AIE agent kernels.
 *
 * This buffer is used to hold the PDI and instruction data for AIE kernels.
 */
class ggml_hsa_aie_buffer {
    ggml_hsa_unique_ptr<std::byte> m_data;
    std::size_t m_size{};

  public:
    constexpr ggml_hsa_aie_buffer() = default;
    ggml_hsa_aie_buffer(std::byte * data, std::size_t size) : m_data{data}, m_size{size} {}

    ggml_hsa_aie_buffer(ggml_hsa_aie_buffer && other) :
        m_data{std::exchange(other.m_data, nullptr)}, m_size{std::exchange(other.m_size, 0)} {}

    ~ggml_hsa_aie_buffer() = default;

    ggml_hsa_aie_buffer & operator=(ggml_hsa_aie_buffer && other) {
        m_data = std::exchange(other.m_data, nullptr);
        m_size = std::exchange(other.m_size, 0);
        return *this;
    }

    /**
     * @brief Returns the size of the buffer in bytes.
     */
    std::size_t size() const { return m_size; }

    /**
     * @brief Returns a pointer to the buffer data.
     */
    std::byte * data() const { return m_data.get(); }
};

/**
 * @brief Kernel for AIE agents.
 */
class ggml_hsa_aie_kernel : public ggml_hsa_kernel {
  public:
    ggml_hsa_aie_buffer pdi;
    ggml_hsa_aie_buffer insts;

    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
