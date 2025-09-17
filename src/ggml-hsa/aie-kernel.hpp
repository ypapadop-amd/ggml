// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "ggml-hsa/common.hpp"
#include "ggml.h"

#include <cstddef>
#include <utility>

/**
 * @brief PDI buffer for AIE agent kernels.
 */
class ggml_hsa_pdi_buffer {
    ggml_hsa_unique_ptr<std::uint64_t> m_data;

  public:
    constexpr ggml_hsa_pdi_buffer() = default;
    explicit ggml_hsa_pdi_buffer(std::uint64_t * data) : m_data{data} {}

    std::uint64_t * data() { return m_data.get(); }
    const std::uint64_t * data() const { return m_data.get(); }
};

/**
 * @brief Instructions buffer for AIE agent kernels.
 */
class ggml_hsa_insts_buffer {
    ggml_hsa_unique_ptr<std::uint32_t> m_data;
    std::size_t m_size{};

  public:
    constexpr ggml_hsa_insts_buffer() = default;
    ggml_hsa_insts_buffer(std::uint32_t * data, std::size_t size) : m_data{data}, m_size{size} {}

    ggml_hsa_insts_buffer(ggml_hsa_insts_buffer && other) :
        m_data{std::exchange(other.m_data, nullptr)}, m_size{std::exchange(other.m_size, 0)} {}

    ~ggml_hsa_insts_buffer() = default;

    ggml_hsa_insts_buffer & operator=(ggml_hsa_insts_buffer && other) {
        m_data = std::exchange(other.m_data, nullptr);
        m_size = std::exchange(other.m_size, 0);
        return *this;
    }

    std::size_t size() const { return m_size; }
    std::uint32_t * data() { return m_data.get(); }
    const std::uint32_t * data() const { return m_data.get(); }
};

/**
 * @brief Kernel for AIE agents.
 */
class ggml_hsa_aie_kernel : public ggml_hsa_kernel {
  public:
    ggml_hsa_pdi_buffer pdi;
    ggml_hsa_insts_buffer insts;

    ggml_status dispatch(ggml_backend_hsa_context & ctx,
                         ggml_tensor * src_tensors[],
                         std::size_t num_src_tensors,
                         ggml_tensor & dst_tensor) const override;
};
