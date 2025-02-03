#include "kernels.hpp"

#include "ggml-impl.h"

#include <fstream>
#include <sstream>
#include <string>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

static std::vector<uint32_t> load_instr_sequence(const std::string& instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

bool ggml_hsa_supports_add(const ggml_tensor * tensor) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if ((src0->type != src1->type) || (src0->type != dst->type) || src0->type != GGML_TYPE_I32) {
      return false;
    }

    if (ggml_nelements(src0) > 1024) {
      return false;
    }

    return true;
}

ggml_status ggml_hsa_add(ggml_backend_hsa_context & ctx, ggml_tensor * tensor) {
    const std::string path = "/home/ypapadop/workspace-raiders/mlir-aie/programming_examples/basic/vector_vector_add/build/";
    const std::string xclbin_path = path + "final.xclbin";
    const std::string instr_path = path + "insts.txt";
    const std::string node = "MLIR_AIE";
    const auto device_index = 0u;

    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * dst  = tensor;

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    const uint64_t element_count = ggml_nelements(src0);

    GGML_ASSERT(element_count <= 1024);

    auto device = xrt::device(device_index);
    auto xclbin = xrt::xclbin(xclbin_path);
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [&node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 return name.rfind(node, 0) == 0;
                               });
    auto kernelName = xkernel.get_name();
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    //xrt::hw_context context2(device, xclbin.get_uuid());

    auto kernel = xrt::kernel(context, kernelName);

    // load instructions
    const auto instr_v = load_instr_sequence(instr_path);
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto * bufInstr = bo_instr.map<uint32_t *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

    // load inputs
    auto bo_inA = xrt::bo(device, element_count * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto * bufInA = bo_inA.map<int32_t *>();
    memcpy(bufInA, src0->data, (element_count * sizeof(int32_t)));

    auto bo_inB = xrt::bo(device, element_count * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto * bufInB = bo_inB.map<int32_t *>();
    memcpy(bufInB, src1->data, (element_count * sizeof(int32_t)));

    // create output
    auto bo_out = xrt::bo(device, element_count * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto * bufOut = bo_out.map<int32_t *>();
    memcpy(dst->data, bufOut, (element_count * sizeof(int32_t)));

    return GGML_STATUS_SUCCESS;
}
