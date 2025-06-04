// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/host-ops.hpp"

#include "ggml-hsa/common.hpp"
#include "ggml-impl.h"

#include <cassert>

template <typename T, typename U>
void copy_tensor_impl(const ggml_tensor * src, ggml_tensor * dst) {
    auto dst_ptr = static_cast<U *>(dst->data);

    std::size_t id = 0;
    for (std::int64_t i03 = 0; i03 < src->ne[3]; ++i03) {
        for (std::int64_t i02 = 0; i02 < src->ne[2]; ++i02) {
            for (std::int64_t i01 = 0; i01 < src->ne[1]; ++i01) {
                for (std::int64_t i00 = 0; i00 < src->ne[0]; ++i00) {
                    auto src_ptr = reinterpret_cast<const T *>(static_cast<char *>(src->data) +
                                                               i00 * src->nb[0] + i01 * src->nb[1] +
                                                               i02 * src->nb[2] + i03 * src->nb[3]);
                    dst_ptr[id] = *static_cast<const T *>(src_ptr);
                    ++id;
                }
            }
        }
    }
}

ggml_status copy_tensor(const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != dst->type) {
        GGML_LOG_ERROR("%s: type mismatch %s != %s (%s)\n", __func__, src->name, dst->name,
                       ggml_type_name(src->type));
        return GGML_STATUS_FAILED;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            copy_tensor_impl<float, float>(src, dst);
            break;
        case GGML_TYPE_F16:
            copy_tensor_impl<ggml_fp16_t, ggml_fp16_t>(src, dst);
            break;
        case GGML_TYPE_I16:
            copy_tensor_impl<std::int16_t, std::int16_t>(src, dst);
            break;
        case GGML_TYPE_I32:
            copy_tensor_impl<std::int32_t, std::int32_t>(src, dst);
            break;
        case GGML_TYPE_BF16:
            copy_tensor_impl<ggml_bf16_t, ggml_bf16_t>(src, dst);
            break;
        default:
            GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, src->name,
                           ggml_type_name(src->type));
            return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

ggml_status ggml_hsa_compute_dup(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 1) && (t->type == t->src[0]->type) &&
           ggml_are_same_shape(t, t->src[0]));

    auto * src = t->src[0];
    auto * dst = t;

    if (dst->view_src == src) {
        // destination tensor is a view of the source tensor
        return GGML_STATUS_SUCCESS;
    }

    ggml_hsa_wait_dispatches(ctx);

    return copy_tensor(src, dst);
}

ggml_status ggml_hsa_compute_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 2) && ggml_are_same_shape(t->src[0], t->src[1]) &&
           ggml_are_same_stride(t->src[0], t->src[1]));

    auto * src = t->src[0];
    auto * dst = t->src[1];

    ggml_hsa_wait_dispatches(ctx);

    return copy_tensor(src, dst);
}

ggml_status ggml_hsa_compute_cont(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 1) && (t->type == t->src[0]->type) &&
           (ggml_nelements(t) == ggml_nelements(t->src[0])) && ggml_is_contiguous(t));

    auto * src = t->src[0];
    auto * dst = t;

    ggml_hsa_wait_dispatches(ctx);

    return copy_tensor(src, dst);
}
