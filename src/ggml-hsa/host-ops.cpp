// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/host-ops.hpp"

#include "ggml-hsa/common.hpp"
#include "ggml-impl.h"

#include <cassert>
#include <cstddef>

/**
 * @brief @ref ggml_typetype traits.
 */
template <ggml_type T>
struct ggml_hsa_type_traits;

template <>
struct ggml_hsa_type_traits<GGML_TYPE_F32> {
    static constexpr ggml_type ggml_type = GGML_TYPE_F32;
    using type = float;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_F16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_F16;
    using type = ggml_fp16_t;
    static constexpr bool is_fundamental = false;
    static constexpr auto to_fp32 = [](ggml_fp16_t v) -> float { return GGML_FP16_TO_FP32(v); };
    static constexpr auto from_fp32 = [](float v) -> ggml_fp16_t { return GGML_FP32_TO_FP16(v); };
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I8> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I8;
    using type = std::int8_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I16;
    using type = std::int16_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_I32> {
    static constexpr ggml_type ggml_type = GGML_TYPE_I32;
    using type = std::int32_t;
    static constexpr bool is_fundamental = true;
};

template <>
struct ggml_hsa_type_traits<GGML_TYPE_BF16> {
    static constexpr ggml_type ggml_type = GGML_TYPE_BF16;
    using type = ggml_bf16_t;
    static constexpr bool is_fundamental = false;
    static constexpr auto to_fp32 = [](ggml_bf16_t v) -> float { return GGML_BF16_TO_FP32(v); };
    static constexpr auto from_fp32 = [](float v) -> ggml_bf16_t { return GGML_FP32_TO_BF16(v); };
};

/**
 * @brief Copies the data from the source tensor to a destination tensor with the same shape.
 *
 * This function handles different types of tensors and performs necessary conversions
 * based on the type traits defined for each tensor type.
 */
template <ggml_type SrcT, ggml_type DstT>
void ggml_hsa_copy_same_shape_tensors(const ggml_tensor * src, ggml_tensor * dst) {
    assert(ggml_are_same_shape(src, dst));

    using src_traits = ggml_hsa_type_traits<SrcT>;
    using dst_traits = ggml_hsa_type_traits<DstT>;

    using src_type = typename src_traits::type;
    using dst_type = typename dst_traits::type;

    for (std::int64_t i03 = 0; i03 < src->ne[3]; ++i03) {
        for (std::int64_t i02 = 0; i02 < src->ne[2]; ++i02) {
            for (std::int64_t i01 = 0; i01 < src->ne[1]; ++i01) {
                for (std::int64_t i00 = 0; i00 < src->ne[0]; ++i00) {
                    auto src_ptr = reinterpret_cast<const src_type *>(
                        static_cast<const std::byte *>(src->data) +
                        (i00 * src->nb[0] + i01 * src->nb[1] + i02 * src->nb[2] +
                         i03 * src->nb[3]));
                    auto dst_ptr =
                        reinterpret_cast<dst_type *>(static_cast<std::byte *>(dst->data) +
                                                     (i00 * dst->nb[0] + i01 * dst->nb[1] +
                                                      i02 * dst->nb[2] + i03 * dst->nb[3]));

                    if constexpr (SrcT == DstT) {
                        // no conversion needed
                        *dst_ptr = *src_ptr;
                    } else if constexpr (src_traits::is_fundamental && dst_traits::is_fundamental) {
                        // trivial conversion based on fundamental types
                        *dst_ptr = static_cast<dst_type>(*src_ptr);
                    } else if constexpr (src_traits::is_fundamental) {
                        // conversion using promotion of source type to fp32
                        auto src_v = static_cast<float>(*src_ptr);
                        *dst_ptr = dst_traits::from_fp32(src_v);
                    } else if constexpr (dst_traits::is_fundamental) {
                        // conversion using promotion of destination type to fp32
                        auto src_v = src_traits::to_fp32(*src_ptr);
                        *dst_ptr = static_cast<dst_type>(src_v);
                    } else {
                        // conversion using promotion of source and destination types to fp32
                        auto src_v = src_traits::to_fp32(*src_ptr);
                        *dst_ptr = dst_traits::from_fp32(src_v);
                    }
                }
            }
        }
    }
}

/**
 * @brief Copies the data from the source tensor to a contiguous destination tensor.
 *
 * This function handles different types of tensors and performs necessary conversions
 * based on the type traits defined for each tensor type.
 */
template <ggml_type SrcT, ggml_type DstT>
void ggml_hsa_copy_tensor_to_cont_tensor(const ggml_tensor * src, ggml_tensor * dst) {
    assert((ggml_nelements(src) == ggml_nelements(dst)) && ggml_is_contiguous(dst));

    using src_traits = ggml_hsa_type_traits<SrcT>;
    using dst_traits = ggml_hsa_type_traits<DstT>;

    using src_type = typename src_traits::type;
    using dst_type = typename dst_traits::type;

    auto dst_ptr = static_cast<dst_type *>(dst->data);

    std::int64_t id = 0;
    for (std::int64_t i03 = 0; i03 < src->ne[3]; ++i03) {
        for (std::int64_t i02 = 0; i02 < src->ne[2]; ++i02) {
            for (std::int64_t i01 = 0; i01 < src->ne[1]; ++i01) {
                for (std::int64_t i00 = 0; i00 < src->ne[0]; ++i00) {
                    auto src_ptr = reinterpret_cast<const src_type *>(
                        static_cast<const std::byte *>(src->data) +
                        (i00 * src->nb[0] + i01 * src->nb[1] + i02 * src->nb[2] +
                         i03 * src->nb[3]));
                    if constexpr (SrcT == DstT) {
                        // no conversion needed
                        dst_ptr[id] = *src_ptr;
                    } else if constexpr (src_traits::is_fundamental && dst_traits::is_fundamental) {
                        // trivial conversion based on fundamental types
                        dst_ptr[id] = static_cast<dst_type>(*src_ptr);
                    } else if constexpr (src_traits::is_fundamental) {
                        // conversion using promotion of source type to fp32
                        auto src_v = static_cast<float>(*src_ptr);
                        dst_ptr[id] = dst_traits::from_fp32(src_v);
                    } else if constexpr (dst_traits::is_fundamental) {
                        // conversion using promotion of destination type to fp32
                        auto src_v = src_traits::to_fp32(*src_ptr);
                        dst_ptr[id] = static_cast<dst_type>(src_v);
                    } else {
                        // conversion using promotion of source and destination types to fp32
                        auto src_v = src_traits::to_fp32(*src_ptr);
                        dst_ptr[id] = dst_traits::from_fp32(src_v);
                    }
                    ++id;
                }
            }
        }
    }
}

/**
 * @brief Function object of @ref ggml_hsa_copy_same_shape_tensors.
 */
struct gggml_hsa_copy_same_shape_tensors_f {
    template <ggml_type T, ggml_type U = T>
    void operator()(const ggml_tensor * src, ggml_tensor * dst) {
        ggml_hsa_copy_same_shape_tensors<T, U>(src, dst);
    }
};

/**
 * @brief Function object of @ref ggml_hsa_copy_tensor_to_cont_tensor.
 */
struct ggml_hsa_copy_tensor_to_cont_tensor_f {
    template <ggml_type T, ggml_type U = T>
    void operator()(const ggml_tensor * src, ggml_tensor * dst) {
        ggml_hsa_copy_tensor_to_cont_tensor<T, U>(src, dst);
    }
};

/**
 * @brief Dispatches the operation based on the tensor type.
 */
template <typename F>
ggml_status ggml_hsa_dispatch(F && f, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type == dst->type) {
        switch (src->type) {
            case GGML_TYPE_F32:
                std::forward<F>(f).template operator()<GGML_TYPE_F32>(src, dst);
                return GGML_STATUS_SUCCESS;
            case GGML_TYPE_F16:
                std::forward<F>(f).template operator()<GGML_TYPE_F16>(src, dst);
                return GGML_STATUS_SUCCESS;
            case GGML_TYPE_I8:
                std::forward<F>(f).template operator()<GGML_TYPE_I8>(src, dst);
                return GGML_STATUS_SUCCESS;
            case GGML_TYPE_I16:
                std::forward<F>(f).template operator()<GGML_TYPE_I16>(src, dst);
                return GGML_STATUS_SUCCESS;
            case GGML_TYPE_I32:
                std::forward<F>(f).template operator()<GGML_TYPE_I32>(src, dst);
                return GGML_STATUS_SUCCESS;
            case GGML_TYPE_BF16:
                std::forward<F>(f).template operator()<GGML_TYPE_BF16>(src, dst);
                return GGML_STATUS_SUCCESS;
            default:
                GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, src->name,
                               ggml_type_name(src->type));
                return GGML_STATUS_FAILED;
        }
    } else {
        switch (src->type) {
            case GGML_TYPE_F32:
                switch (dst->type) {
                    case GGML_TYPE_F16:
                        std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_F16>(src,
                                                                                             dst);
                        return GGML_STATUS_SUCCESS;
                    case GGML_TYPE_I8:
                        std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_I8>(src,
                                                                                            dst);
                        return GGML_STATUS_SUCCESS;
                    case GGML_TYPE_I16:
                        std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_I16>(src,
                                                                                             dst);
                        return GGML_STATUS_SUCCESS;
                    case GGML_TYPE_I32:
                        std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_I32>(src,
                                                                                             dst);
                        return GGML_STATUS_SUCCESS;
                    case GGML_TYPE_BF16:
                        std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_BF16>(src,
                                                                                              dst);
                        return GGML_STATUS_SUCCESS;
                    default:
                        GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, dst->name,
                                       ggml_type_name(dst->type));
                        return GGML_STATUS_FAILED;
                }
            case GGML_TYPE_F16:
                switch (dst->type) {
                    case GGML_TYPE_F32:
                        std::forward<F>(f).template operator()<GGML_TYPE_F16, GGML_TYPE_F32>(src,
                                                                                             dst);
                        return GGML_STATUS_SUCCESS;
                    case GGML_TYPE_BF16:
                        std::forward<F>(f).template operator()<GGML_TYPE_F16, GGML_TYPE_BF16>(src,
                                                                                              dst);
                        return GGML_STATUS_SUCCESS;
                    default:
                        GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, dst->name,
                                       ggml_type_name(dst->type));
                        return GGML_STATUS_FAILED;
                }
            case GGML_TYPE_BF16:
                switch (dst->type) {
                    case GGML_TYPE_F32:
                        std::forward<F>(f).template operator()<GGML_TYPE_BF16, GGML_TYPE_F32>(src,
                                                                                              dst);
                        return GGML_STATUS_SUCCESS;
                    default:
                        GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, dst->name,
                                       ggml_type_name(dst->type));
                        return GGML_STATUS_FAILED;
                }
            default:
                GGML_LOG_ERROR("%s: type not supported %s (%s)\n", __func__, src->name,
                               ggml_type_name(src->type));
                return GGML_STATUS_FAILED;
        }
    }
}

ggml_status ggml_hsa_copy_tensor(const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_is_contiguous(dst)) {
        return ggml_hsa_dispatch(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
    }

    if (ggml_are_same_shape(src, dst)) {
        return ggml_hsa_dispatch(gggml_hsa_copy_same_shape_tensors_f{}, src, dst);
    }

    GGML_LOG_ERROR("%s: unsupported tensor combination between source %s and destination %s\n",
                   __func__, src->name, dst->name);
    return GGML_STATUS_FAILED;
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

    if (ggml_is_contiguous(dst)) {
        return ggml_hsa_dispatch(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
    }

    return ggml_hsa_dispatch(gggml_hsa_copy_same_shape_tensors_f{}, src, dst);
}

ggml_status ggml_hsa_compute_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 2) && (ggml_nelements(t->src[0]) == ggml_nelements(t->src[1])));

    auto * src = t->src[0];
    auto * dst = t->src[1];

    ggml_hsa_wait_dispatches(ctx);

    return ggml_hsa_copy_tensor(src, dst);
}

ggml_status ggml_hsa_compute_cont(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 1) && (t->type == t->src[0]->type) &&
           (ggml_nelements(t) == ggml_nelements(t->src[0])) && ggml_is_contiguous(t));

    auto * src = t->src[0];
    auto * dst = t;

    ggml_hsa_wait_dispatches(ctx);

    return ggml_hsa_dispatch(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
}
