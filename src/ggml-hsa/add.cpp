#include "kernels.hpp"

#include "ggml-impl.h"

bool ggml_hsa_supports_add(const ggml_tensor * tensor) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    const ggml_tensor * dst = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if ((src0->type != src1->type) || (src0->type != dst->type) || (src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_F32)) {
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

    std::string path = "/home/ypapadop/workspace-raiders/mlir-aie/programming_examples/basic/vector_vector_add/build/";

    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * dst = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    // Note: Assuming src0 and src1 are of the same type
    std::string pdi_path;
    std::string instr_path;
    if (tensor->src[0]->type == GGML_TYPE_I32) {
      pdi_path = path + "vector_vector_add_i32.pdi";
      instr_path = path + "vector_vector_add_insts_i32.txt";
    }
    else if (tensor->src[0]->type == GGML_TYPE_F32) {
      pdi_path = path + "vector_vector_add_f32.pdi";
      instr_path = path + "vector_vector_add_insts_f32.txt";
    }
    else {
      // We check type in ggml_hsa_supports_add so shouldn't hit this
      return GGML_STATUS_FAILED;
    }

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

#define LOW_ADDR(addr) (reinterpret_cast<uint64_t>(addr) & 0xFFFFFFFF)
#define HIGH_ADDR(addr) (reinterpret_cast<uint64_t>(addr) >> 32)

    hsa_amd_aie_ert_start_kernel_data_t * cmd_payload = nullptr;
    if (auto status = hsa_amd_memory_pool_allocate(device_info.kernarg_memory.memory_pool, 64, 0, reinterpret_cast<void **>(&cmd_payload));
        status != HSA_STATUS_SUCCESS) {
        GGML_LOG_ERROR("%s: Could not allocate hsa_amd_aie_ert_start_kernel_data_t (%d)\n", __func__, status);
        return GGML_STATUS_FAILED;
    }
    cmd_payload->pdi_addr = pdi_buf; // PDI to use with this command
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
