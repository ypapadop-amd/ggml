#include "kernels.hpp"

#include "ggml-impl.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

/**
 * @brief Reads a PDI file from @p filename and returns its contents and size in bytes in @p buffer and @p buffer_size respectively.
 */
static ggml_status ggml_load_pdi(hsa_amd_memory_pool_t pool, const std::string & filename, std::uint64_t *& buffer, std::size_t & buffer_size) {
    std::ifstream is(filename, std::ios::binary | std::ios::ate | std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, filename.c_str());
        return GGML_STATUS_FAILED;
    }

    const std::size_t size = is.tellg();
    GGML_ASSERT(size > 0);
    if (!is.seekg(0, std::ios::beg)) {
        GGML_LOG_ERROR("%s: I/O error, could not get file size for %s\n", __func__, filename.c_str());
        return GGML_STATUS_FAILED;
    }
    if (auto status = hsa_amd_memory_pool_allocate(pool, size, 0, reinterpret_cast<void **>(&buffer));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, size);
        return GGML_STATUS_FAILED;
    }

    is.read(reinterpret_cast<char *>(buffer), size);
    buffer_size = size;

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Reads an instruction file from @p filename and returns its contents and number of instructions size in
 *        @p instr_buf and @p instr_count respectively.
 */
static ggml_status ggml_load_instr(hsa_amd_memory_pool_t pool, const std::string & filename, std::uint32_t *& buffer, std::size_t & instr_count) {
    std::ifstream is(filename, std::ios::in);
    if (is.fail()) {
        GGML_LOG_ERROR("%s: Could not open file %s\n", __func__, filename.c_str());
        return GGML_STATUS_FAILED;
    }

    std::string line;
    std::vector<std::uint32_t> instr_v;
    while (std::getline(is, line)) {
      std::istringstream iss(line);
      std::uint32_t a;
      if (!(iss >> std::hex >> a)) {
        GGML_LOG_ERROR("%s: I/O error, could not read file %s\n", __func__, filename.c_str());
          return GGML_STATUS_FAILED;
      }
      instr_v.push_back(a);
    }
    GGML_ASSERT(instr_v.empty() == false);

    const std::size_t required_memory_size = instr_v.size() * sizeof(std::uint32_t);
    if (auto status = hsa_amd_memory_pool_allocate(pool, required_memory_size, 0, reinterpret_cast<void **>(&buffer));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate %zu bytes\n", __func__, required_memory_size);
        return GGML_STATUS_FAILED;
    }

    std::copy(instr_v.begin(), instr_v.end(), buffer);
    instr_count = instr_v.size();

    return GGML_STATUS_SUCCESS;
}

bool ggml_hsa_supports_add(const ggml_tensor * tensor) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    const ggml_tensor * dst = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if ((src0->type != src1->type) || (src0->type != dst->type) || src0->type != GGML_TYPE_I32) {
      return false;
    }

    if (ggml_nelements(src0) != 256) {
      return false;
    }

    return true;
}

ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor) {
    auto & info = ggml_hsa_info();
    auto & device_info = info.devices[ctx.device];

    const std::string path = "/home/ypapadop/workspace-raiders/mlir-aie/programming_examples/basic/vector_vector_add/build/";
    const std::string pdi_path = path + "add.pdi";
    const std::string instr_path = path + "insts.txt";

    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * dst = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    const std::int64_t element_count = ggml_nelements(src0);

    GGML_ASSERT(element_count == 256);

    std::uint64_t * pdi_buf = nullptr;
    std::size_t pdi_size = 0;
    if (auto status = ggml_load_pdi(device_info.dev_memory.memory_pool, pdi_path, pdi_buf, pdi_size);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    std::uint32_t * instr_buf = nullptr;
    std::size_t instr_count = 0;
    if (auto status = ggml_load_instr(device_info.dev_memory.memory_pool, instr_path, instr_buf, instr_count);
        status != GGML_STATUS_SUCCESS) {
        return status;
    }

    hsa_amd_aie_ert_hw_ctx_cu_config_addr_t cu_config = {};
    cu_config.cu_config_addr = reinterpret_cast<std::uint64_t>(pdi_buf);
    cu_config.cu_size = pdi_size;

    hsa_amd_aie_ert_hw_ctx_config_cu_param_addr_t config_cu_args = {};
    config_cu_args.num_cus = 1;
    config_cu_args.cu_configs = &cu_config;

    if (auto status = hsa_amd_queue_hw_ctx_config(ctx.queue, HSA_AMD_QUEUE_AIE_ERT_HW_CXT_CONFIG_CU, &config_cu_args);
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not configure hardware context (%d)\n", __func__, status);
        return GGML_STATUS_FAILED;
    }

#define LOW_ADDR(addr) (reinterpret_cast<uint64_t>(addr) & 0xFFFFFFFF)
#define HIGH_ADDR(addr) (reinterpret_cast<uint64_t>(addr) >> 32)

    hsa_amd_aie_ert_start_kernel_data_t * cmd_payload = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(device_info.kernarg_memory.memory_pool, 64, 0, reinterpret_cast<void **>(&cmd_payload));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate hsa_amd_aie_ert_start_kernel_data_t (%d)\n", __func__, status);
        return GGML_STATUS_FAILED;
    }
    cmd_payload->cu_mask = 0x1; // PDI to use with this command
    cmd_payload->data[0] = 0x3; // Transaction opcode
    cmd_payload->data[1] = 0x0;
    cmd_payload->data[2] = LOW_ADDR(instr_buf);
    cmd_payload->data[3] = HIGH_ADDR(instr_buf);
    cmd_payload->data[4] = static_cast<std::uint32_t>(instr_count);
    cmd_payload->data[5] = LOW_ADDR(src0->data);
    cmd_payload->data[6] = HIGH_ADDR(src0->data);
    cmd_payload->data[7] = LOW_ADDR(src1->data);
    cmd_payload->data[8] = HIGH_ADDR(src1->data);
    cmd_payload->data[9] = LOW_ADDR(dst->data);
    cmd_payload->data[10] = HIGH_ADDR(dst->data);
    cmd_payload->data[11] = element_count * sizeof(std::uint32_t);
    cmd_payload->data[12] = element_count * sizeof(std::uint32_t);
    cmd_payload->data[13] = element_count * sizeof(std::uint32_t);

    std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(ctx.queue, 1);
    std::uint64_t packet_id = wr_idx % ctx.queue->size;
    auto * cmd_pkt = static_cast<hsa_amd_aie_ert_packet_t *>(ctx.queue->base_address) + packet_id;
    cmd_pkt->state = HSA_AMD_AIE_ERT_STATE_NEW;
    cmd_pkt->count = 0xC; // # of arguments to put in command
    cmd_pkt->opcode = HSA_AMD_AIE_ERT_START_CU;
    cmd_pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
    cmd_pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    cmd_pkt->payload_data = reinterpret_cast<std::uint64_t>(cmd_payload);
    // TODO add cmd_pkt->completion_signal = ctx.dispatch_signal

    hsa_signal_store_screlease(ctx.queue->doorbell_signal, wr_idx);

    return GGML_STATUS_SUCCESS;
}
