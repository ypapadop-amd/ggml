#include "kernels.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include "ggml-impl.h"

namespace {

/**
 * @brief Maps an operation to the relevant kernel files.
 */
struct ggml_hsa_operation_kernel {
  ggml_op op;
  std::string_view pdi_file;
  std::string_view instr_file;

  constexpr bool valid() const noexcept { return !pdi_file.empty() && !instr_file.empty(); }
};

constexpr ggml_hsa_operation_kernel op_kernel_map[] = {
  { GGML_OP_NONE, "", "" },
  { GGML_OP_DUP, "", "" },
  { GGML_OP_ADD, "add.pdi", "insts.txt" },
  { GGML_OP_ADD1, "", "" },
  { GGML_OP_ACC, "", "" },
  { GGML_OP_SUB, "", "" },
  { GGML_OP_MUL, "", "" },
  { GGML_OP_DIV, "", "" },
  { GGML_OP_SQR, "", "" },
  { GGML_OP_SQRT, "", "" },
  { GGML_OP_LOG, "", "" },
  { GGML_OP_SIN, "", "" },
  { GGML_OP_COS, "", "" },
  { GGML_OP_SUM, "", "" },
  { GGML_OP_SUM_ROWS, "", "" },
  { GGML_OP_MEAN, "", "" },
  { GGML_OP_ARGMAX, "", "" },
  { GGML_OP_COUNT_EQUAL, "", "" },
  { GGML_OP_REPEAT, "", "" },
  { GGML_OP_REPEAT_BACK, "", "" },
  { GGML_OP_CONCAT, "", "" },
  { GGML_OP_SILU_BACK, "", "" },
  { GGML_OP_NORM, "", "" },
  { GGML_OP_RMS_NORM, "", "" },
  { GGML_OP_RMS_NORM_BACK, "", "" },
  { GGML_OP_GROUP_NORM, "", "" },
  { GGML_OP_MUL_MAT, "", "" },
  { GGML_OP_MUL_MAT_ID, "", "" },
  { GGML_OP_OUT_PROD, "", "" },
  { GGML_OP_SCALE, "", "" },
  { GGML_OP_SET, "", "" },
  { GGML_OP_CPY, "", "" },
  { GGML_OP_CONT, "", "" },
  { GGML_OP_RESHAPE, "", "" },
  { GGML_OP_VIEW, "", "" },
  { GGML_OP_PERMUTE, "", "" },
  { GGML_OP_TRANSPOSE, "", "" },
  { GGML_OP_GET_ROWS, "", "" },
  { GGML_OP_GET_ROWS_BACK, "", "" },
  { GGML_OP_DIAG, "", "" },
  { GGML_OP_DIAG_MASK_INF, "", "" },
  { GGML_OP_DIAG_MASK_ZERO, "", "" },
  { GGML_OP_SOFT_MAX, "", "" },
  { GGML_OP_SOFT_MAX_BACK, "", "" },
  { GGML_OP_ROPE, "", "" },
  { GGML_OP_ROPE_BACK, "", "" },
  { GGML_OP_CLAMP, "", "" },
  { GGML_OP_CONV_TRANSPOSE_1D, "", "" },
  { GGML_OP_IM2COL, "", "" },
  { GGML_OP_IM2COL_BACK, "", "" },
  { GGML_OP_CONV_TRANSPOSE_2D, "", "" },
  { GGML_OP_POOL_1D, "", "" },
  { GGML_OP_POOL_2D, "", "" },
  { GGML_OP_POOL_2D_BACK, "", "" },
  { GGML_OP_UPSCALE, "", "" },
  { GGML_OP_PAD, "", "" },
  { GGML_OP_PAD_REFLECT_1D, "", "" },
  { GGML_OP_ARANGE, "", "" },
  { GGML_OP_TIMESTEP_EMBEDDING, "", "" },
  { GGML_OP_ARGSORT, "", "" },
  { GGML_OP_LEAKY_RELU, "", "" },
  { GGML_OP_FLASH_ATTN_EXT, "", "" },
  { GGML_OP_FLASH_ATTN_BACK, "", "" },
  { GGML_OP_SSM_CONV, "", "" },
  { GGML_OP_SSM_SCAN, "", "" },
  { GGML_OP_WIN_PART, "", "" },
  { GGML_OP_WIN_UNPART, "", "" },
  { GGML_OP_GET_REL_POS, "", "" },
  { GGML_OP_ADD_REL_POS, "", "" },
  { GGML_OP_RWKV_WKV6, "", "" },
  { GGML_OP_GATED_LINEAR_ATTN, "", "" },
  { GGML_OP_UNARY, "", "" },
  { GGML_OP_MAP_UNARY, "", "" },
  { GGML_OP_MAP_BINARY, "", "" },
  { GGML_OP_MAP_CUSTOM1_F32, "", "" },
  { GGML_OP_MAP_CUSTOM2_F32, "", "" },
  { GGML_OP_MAP_CUSTOM3_F32, "", "" },
  { GGML_OP_MAP_CUSTOM1, "", "" },
  { GGML_OP_MAP_CUSTOM2, "", "" },
  { GGML_OP_MAP_CUSTOM3, "", "" },
  { GGML_OP_CROSS_ENTROPY_LOSS, "", "" },
  { GGML_OP_CROSS_ENTROPY_LOSS_BACK, "", "" },
  { GGML_OP_OPT_STEP_ADAMW, "", "" },
  { GGML_OP_COUNT, "", "" },
};
static_assert((sizeof(op_kernel_map) / sizeof(op_kernel_map[0])) - 1 == GGML_OP_COUNT,
              "Incorrect operation mapping");

const std::filesystem::path kernel_base_path = "/home/ypapadop/workspace-raiders/mlir-aie/programming_examples/basic/vector_vector_add/build/";

/**
 * @brief Returns if @p p is a file.
 */
bool ggml_hsa_is_file(const std::filesystem::path& p) {
  return std::filesystem::is_regular_file(p) || std::filesystem::is_symlink(p);
}

/**
 * @brief Reads a PDI file from @p filename and returns its contents and size in bytes in @p buffer and @p buffer_size respectively.
 */
ggml_status ggml_hsa_load_pdi(hsa_amd_memory_pool_t pool, const std::string & filename, std::uint64_t *& buffer, std::size_t & buffer_size) {
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
ggml_status ggml_hsa_load_instr(hsa_amd_memory_pool_t pool, const std::string & filename, std::uint32_t *& buffer, std::size_t & instr_count) {
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

} // namespace

ggml_status ggml_hsa_load_kernel(ggml_backend_hsa_context & ctx, const ggml_tensor * tensor,  ggml_hsa_pdi_buffer & pdi_buf, ggml_hsa_instr_buffer & instr_buf) {
  if ((tensor->op < GGML_OP_NONE) || (tensor->op >= GGML_OP_COUNT)) {
    GGML_LOG_ERROR("%s: Tensor operation index out of bounds (%d >= GGML_OP_COUNT)\n", __func__, tensor->op);
    return GGML_STATUS_FAILED;
  }

  const auto & op_kernel_mapping = op_kernel_map[tensor->op];
  if (tensor->op != op_kernel_mapping.op) {
    GGML_ABORT("%s: Inconsistent index in kernel/operation map for operation %s\n", __func__, ggml_op_name(tensor->op));
  }

  if (!op_kernel_mapping.valid()) {
    GGML_LOG_WARN("%s: No kernel found for operation %s\n", __func__, ggml_op_name(tensor->op));
    return GGML_STATUS_FAILED;
  }

  const auto pdi_path = kernel_base_path / op_kernel_mapping.pdi_file;
  if (!ggml_hsa_is_file(pdi_path)) {
    GGML_LOG_WARN("%s: No PDI file found for operation %s in %s\n", __func__, ggml_op_name(tensor->op), pdi_path.c_str());
    return GGML_STATUS_FAILED;
  }
  const auto instr_path = kernel_base_path / op_kernel_mapping.instr_file;
  if (!ggml_hsa_is_file(instr_path)) {
    GGML_LOG_WARN("%s: No instr file found for operation %s in %s\n", __func__, ggml_op_name(tensor->op), instr_path.c_str());
    return GGML_STATUS_FAILED;
  }

  auto & info = ggml_hsa_info();
  auto & device_info = info.devices[ctx.device];

  if (auto status = ggml_hsa_load_pdi(device_info.dev_memory.memory_pool, pdi_path, pdi_buf.data, pdi_buf.size);
      status != GGML_STATUS_SUCCESS) {
    return status;
  }

  if (auto status = ggml_hsa_load_instr(device_info.dev_memory.memory_pool, instr_path, instr_buf.data, instr_buf.size);
      status != GGML_STATUS_SUCCESS) {
    return status;
  }

  return GGML_STATUS_SUCCESS;
}

