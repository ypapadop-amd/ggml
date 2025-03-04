#include "kernels.hpp"

#include "ggml-impl.h"

#include <Eigen/Dense>

template<typename T>
ggml_status ggml_hsa_mul_mat_impl(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    auto mat_src0 = matrix_type::Map(static_cast<T*>(src0->data), src0->ne[0], src0->ne[1]);
    auto mat_src1 = matrix_type::Map(static_cast<T*>(src1->data), src1->ne[0], src1->ne[1]);
    auto mat_dst = matrix_type::Map(static_cast<T*>(dst->data), dst->ne[0], dst->ne[1]);
    mat_dst = mat_src0.transpose() * mat_src1;
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_hsa_mul_mat_f32(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("%s: Unsupported type for src1 %s", __func__, ggml_type_name(src1->type));
        return GGML_STATUS_FAILED;
    }

    if (dst->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("%s: Unsupported type for dst %s", __func__, ggml_type_name(dst->type));
        return GGML_STATUS_FAILED;
    }

    using matrix_type = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

    matrix_type mat_src0;
    switch (src0->type) {
        case GGML_TYPE_F16: {
            auto tmp = Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic>::Map(static_cast<Eigen::half*>(src0->data), src0->ne[0], src0->ne[1]);
            mat_src0 = tmp.cast<float>();
            break;
        }
        case GGML_TYPE_F32:
            mat_src0 = matrix_type::Map(static_cast<float*>(src0->data), src0->ne[0], src0->ne[1]);
            break;
        default:
            return GGML_STATUS_FAILED;
    }
    auto mat_src1 = matrix_type::Map(static_cast<float*>(src1->data), src1->ne[0], src1->ne[1]);
    auto mat_dst = matrix_type::Map(static_cast<float*>(dst->data), dst->ne[0], dst->ne[1]);
    mat_dst = mat_src0.transpose() * mat_src1;

    return GGML_STATUS_SUCCESS;
}

bool ggml_hsa_supports_mul_mat(const ggml_hsa_device_info::device_info & /*dev_info*/, const ggml_tensor * tensor) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    const ggml_tensor * dst = tensor;

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return false;
    }

    if ((src0->type) == (src1->type)
        && (src0->type == dst->type)
        && ((src0->type == GGML_TYPE_F16) || (src0->type == GGML_TYPE_F32) || (src0->type == GGML_TYPE_F64))) {
        return true;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (((src0->type == GGML_TYPE_F16) || (src0->type == GGML_TYPE_F32))
        && (src1->type == GGML_TYPE_F32)
        && (dst->type == GGML_TYPE_F32)) {
        return true;
    }

    return false;
}


ggml_status ggml_hsa_mul_mat(ggml_backend_hsa_context & /*ctx*/, ggml_tensor * tensor) {
    const ggml_tensor * src0 = tensor->src[0];
    const ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * dst = tensor;

    assert(ggml_is_contiguous(src0));
    assert(ggml_is_contiguous(src1));
    assert(ggml_is_contiguous(dst));


    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        GGML_LOG_ERROR("%s: %s: matmul on tranposed tensor not supported", __func__, ggml_op_name(dst->op));
        return GGML_STATUS_FAILED;
    }

    ggml_status status = GGML_STATUS_FAILED;
    if ((src0->type) == (src1->type) && (src0->type == dst->type)) {
        switch (src0->type) {
            case GGML_TYPE_F16:
                status = ggml_hsa_mul_mat_impl<Eigen::half>(src0, src1, dst);
                break;
            case GGML_TYPE_F32:
                status = ggml_hsa_mul_mat_impl<float>(src0, src1, dst);
                break;
            case GGML_TYPE_F64: {
                status = ggml_hsa_mul_mat_impl<double>(src0, src1, dst);
                break;
            default:
                GGML_LOG_ERROR("%s: Unsupported type %s", __func__, ggml_type_name(dst->type));
                status = GGML_STATUS_FAILED;
                break;
            }
        }
    }
    else {
        status = ggml_hsa_mul_mat_f32(src0, src1, dst);
    }

    return status;
}
