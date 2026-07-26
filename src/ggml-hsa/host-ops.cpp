// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/host-ops.hpp"

#include <algorithm>
#include <cassert>
#include <new>
#include <utility>

#include <cstring>

#include "ggml-hsa/common.hpp"
#include "ggml-hsa/type-traits.hpp"

/**
 * @brief Converts a single element from @c SrcT to @c DstT via the type traits.
 *
 * Fundamental types convert with a plain cast; non-fundamental types (e.g. f16/bf16) promote
 * through fp32. Shared by every host copy/gather functor so the conversion policy lives in one place.
 */
template <ggml_type SrcT, ggml_type DstT>
inline void ggml_hsa_convert_element(const typename ggml_hsa_type_traits<SrcT>::type * src,
                                     typename ggml_hsa_type_traits<DstT>::type * dst) {
    using src_traits = ggml_hsa_type_traits<SrcT>;
    using dst_traits = ggml_hsa_type_traits<DstT>;
    using dst_type = typename dst_traits::type;

    if constexpr (SrcT == DstT) {
        // no conversion needed
        *dst = *src;
    } else if constexpr (src_traits::is_fundamental && dst_traits::is_fundamental) {
        // trivial conversion based on fundamental types
        *dst = static_cast<dst_type>(*src);
    } else if constexpr (src_traits::is_fundamental) {
        // conversion using promotion of source type to fp32
        *dst = dst_traits::from_fp32(static_cast<float>(*src));
    } else if constexpr (dst_traits::is_fundamental) {
        // conversion using promotion of destination type to fp32
        *dst = static_cast<dst_type>(src_traits::to_fp32(*src));
    } else {
        // conversion using promotion of source and destination types to fp32
        *dst = dst_traits::from_fp32(src_traits::to_fp32(*src));
    }
}

/**
 * @brief Copies data from a source tensor to a destination tensor with the same shape, converting
 * between types as needed based on their type traits.
 */
struct ggml_hsa_copy_same_shape_tensors_f {
    template <ggml_type SrcT, ggml_type DstT = SrcT>
    ggml_status operator()(const ggml_tensor * src, ggml_tensor * dst) {
        assert(ggml_are_same_shape(src, dst));

        using src_type = typename ggml_hsa_type_traits<SrcT>::type;
        using dst_type = typename ggml_hsa_type_traits<DstT>::type;

        for (std::int64_t i03 = 0; i03 < src->ne[3]; ++i03) {
            for (std::int64_t i02 = 0; i02 < src->ne[2]; ++i02) {
                for (std::int64_t i01 = 0; i01 < src->ne[1]; ++i01) {
                    for (std::int64_t i00 = 0; i00 < src->ne[0]; ++i00) {
                        auto src_ptr = std::launder(reinterpret_cast<const src_type *>(
                            static_cast<const std::byte *>(src->data) +
                            (i00 * src->nb[0] + i01 * src->nb[1] + i02 * src->nb[2] +
                             i03 * src->nb[3])));
                        auto dst_ptr = std::launder(
                            reinterpret_cast<dst_type *>(static_cast<std::byte *>(dst->data) +
                                                         (i00 * dst->nb[0] + i01 * dst->nb[1] +
                                                          i02 * dst->nb[2] + i03 * dst->nb[3])));

                        ggml_hsa_convert_element<SrcT, DstT>(src_ptr, dst_ptr);
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

/**
 * @brief Copies the overlapping sub-block between two differently-shaped tensors.
 *
 * Iterates over the per-dimension overlap and indexes both tensors through their own strides, so a
 * smaller logical tensor can be scattered into a larger zero-padded destination (or gathered back).
 * Padding gaps in the destination are never written and must be pre-zeroed by the caller. Datatype
 * conversion is handled identically to the same-shape copy.
 */
struct ggml_hsa_copy_subblock_f {
    template <ggml_type SrcT, ggml_type DstT = SrcT>
    ggml_status operator()(const ggml_tensor * src, ggml_tensor * dst) {
        using src_type = typename ggml_hsa_type_traits<SrcT>::type;
        using dst_type = typename ggml_hsa_type_traits<DstT>::type;

        const std::int64_t ne0 = std::min(src->ne[0], dst->ne[0]);
        const std::int64_t ne1 = std::min(src->ne[1], dst->ne[1]);
        const std::int64_t ne2 = std::min(src->ne[2], dst->ne[2]);
        const std::int64_t ne3 = std::min(src->ne[3], dst->ne[3]);

        for (std::int64_t i03 = 0; i03 < ne3; ++i03) {
            for (std::int64_t i02 = 0; i02 < ne2; ++i02) {
                for (std::int64_t i01 = 0; i01 < ne1; ++i01) {
                    for (std::int64_t i00 = 0; i00 < ne0; ++i00) {
                        auto src_ptr = std::launder(reinterpret_cast<const src_type *>(
                            static_cast<const std::byte *>(src->data) +
                            (i00 * src->nb[0] + i01 * src->nb[1] + i02 * src->nb[2] +
                             i03 * src->nb[3])));
                        auto dst_ptr = std::launder(
                            reinterpret_cast<dst_type *>(static_cast<std::byte *>(dst->data) +
                                                         (i00 * dst->nb[0] + i01 * dst->nb[1] +
                                                          i02 * dst->nb[2] + i03 * dst->nb[3])));

                        ggml_hsa_convert_element<SrcT, DstT>(src_ptr, dst_ptr);
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

/**
 * @brief Copies data from a source tensor to a contiguous destination tensor, converting between
 * types as needed based on their type traits.
 */
struct ggml_hsa_copy_tensor_to_cont_tensor_f {
    template <ggml_type SrcT, ggml_type DstT = SrcT>
    ggml_status operator()(const ggml_tensor * src, ggml_tensor * dst) {
        assert((ggml_nelements(src) == ggml_nelements(dst)) && ggml_is_contiguous(dst));

        using src_type = typename ggml_hsa_type_traits<SrcT>::type;
        using dst_type = typename ggml_hsa_type_traits<DstT>::type;

        auto dst_ptr = std::launder(static_cast<dst_type *>(dst->data));

        std::int64_t id = 0;
        for (std::int64_t i03 = 0; i03 < src->ne[3]; ++i03) {
            for (std::int64_t i02 = 0; i02 < src->ne[2]; ++i02) {
                for (std::int64_t i01 = 0; i01 < src->ne[1]; ++i01) {
                    for (std::int64_t i00 = 0; i00 < src->ne[0]; ++i00) {
                        auto src_ptr = std::launder(reinterpret_cast<const src_type *>(
                            static_cast<const std::byte *>(src->data) +
                            (i00 * src->nb[0] + i01 * src->nb[1] + i02 * src->nb[2] +
                             i03 * src->nb[3])));
                        ggml_hsa_convert_element<SrcT, DstT>(src_ptr, &dst_ptr[id]);
                        ++id;
                    }
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

/**
 * @brief Gathers rows of @p src (src0) selected by an index tensor into @p dst (GGML_OP_GET_ROWS).
 *
 * Implements ggml_compute_forward_get_rows: for each of the nr = nelements(indices) output rows,
 * reads the int32 row index i01 and copies the nc = ne00 elements of src0 row (i01, i11, i12) into
 * dst row (i10, i11, i12), converting the datatype if needed. The index tensor is held as a member
 * so the (src0, dst) operator() signature matches @ref ggml_hsa_assign.
 */
struct ggml_hsa_get_rows_f {
    const ggml_tensor * indices;

    template <ggml_type SrcT, ggml_type DstT = SrcT>
    ggml_status operator()(const ggml_tensor * src, ggml_tensor * dst) {
        using src_type = typename ggml_hsa_type_traits<SrcT>::type;
        using dst_type = typename ggml_hsa_type_traits<DstT>::type;

        const std::int64_t nc    = src->ne[0];
        const std::int64_t ne10  = indices->ne[0];
        const std::int64_t ne11  = indices->ne[1];
        const std::int64_t ne12  = indices->ne[2];
        const std::int64_t slice = ne11 * ne10;  // indices per ne12 slice
        const std::int64_t nr    = slice * ne12;

        assert(dst->ne[0] == nc);

        for (std::int64_t i = 0; i < nr; ++i) {
            const std::int64_t i12 = i / slice;
            const std::int64_t i11 = (i - i12 * slice) / ne10;
            const std::int64_t i10 = i - i12 * slice - i11 * ne10;

            const std::int64_t i01 = *std::launder(reinterpret_cast<const std::int32_t *>(
                static_cast<const std::byte *>(indices->data) +
                (i10 * indices->nb[0] + i11 * indices->nb[1] + i12 * indices->nb[2])));

            if (i01 < 0 || i01 >= src->ne[1]) {
                GGML_HSA_LOG_ERROR("%s: get_rows index %lld out of range [0, %lld)", __func__,
                                   static_cast<long long>(i01), static_cast<long long>(src->ne[1]));
                return GGML_STATUS_FAILED;
            }

            const auto * src_row = static_cast<const std::byte *>(src->data) +
                                   (i01 * src->nb[1] + i11 * src->nb[2] + i12 * src->nb[3]);
            auto * dst_row = static_cast<std::byte *>(dst->data) +
                             (i10 * dst->nb[1] + i11 * dst->nb[2] + i12 * dst->nb[3]);

            if constexpr (SrcT == DstT) {
                // Same dtype: the row is contiguous (nb[0] == element size), so copy it whole.
                std::memcpy(dst_row, src_row, static_cast<std::size_t>(nc) * sizeof(src_type));
            } else {
                for (std::int64_t i00 = 0; i00 < nc; ++i00) {
                    auto src_ptr = std::launder(
                        reinterpret_cast<const src_type *>(src_row + i00 * src->nb[0]));
                    auto dst_ptr =
                        std::launder(reinterpret_cast<dst_type *>(dst_row + i00 * dst->nb[0]));
                    ggml_hsa_convert_element<SrcT, DstT>(src_ptr, dst_ptr);
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

/**
 * @brief Assigns @p src to @p dst using @p f as the copy operation.
 */
template <typename F>
ggml_status ggml_hsa_assign(F && f, const ggml_tensor * src, ggml_tensor * dst) {
    switch (src->type) {
        case GGML_TYPE_F32:
            switch (dst->type) {
                case GGML_TYPE_F32:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F32>(src, dst);
                case GGML_TYPE_F16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_F16>(
                        src, dst);
                case GGML_TYPE_BF16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F32, GGML_TYPE_BF16>(
                        src, dst);
                default:
                    GGML_HSA_LOG_ERROR("%s: unsupported type for destination tensor \"%s\" (%s)",
                                       __func__, dst->name, ggml_type_name(dst->type));
                    return GGML_STATUS_FAILED;
            }
        case GGML_TYPE_F16:
            switch (dst->type) {
                case GGML_TYPE_F32:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F16, GGML_TYPE_F32>(
                        src, dst);
                case GGML_TYPE_F16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F16>(src, dst);
                case GGML_TYPE_BF16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_F16, GGML_TYPE_BF16>(
                        src, dst);
                default:
                    GGML_HSA_LOG_ERROR("%s: unsupported type for destination tensor \"%s\" (%s)",
                                       __func__, dst->name, ggml_type_name(dst->type));
                    return GGML_STATUS_FAILED;
            }
        case GGML_TYPE_I16:
            switch (dst->type) {
                case GGML_TYPE_I8:
                    return std::forward<F>(f).template operator()<GGML_TYPE_I16, GGML_TYPE_I8>(src,
                                                                                               dst);
                case GGML_TYPE_I16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_I16>(src, dst);
                case GGML_TYPE_I32:
                    return std::forward<F>(f).template operator()<GGML_TYPE_I16, GGML_TYPE_I32>(
                        src, dst);
                default:
                    GGML_HSA_LOG_ERROR("%s: unsupported type for destination tensor \"%s\" (%s)",
                                       __func__, dst->name, ggml_type_name(dst->type));
                    return GGML_STATUS_FAILED;
            }
        case GGML_TYPE_BF16:
            switch (dst->type) {
                case GGML_TYPE_F32:
                    return std::forward<F>(f).template operator()<GGML_TYPE_BF16, GGML_TYPE_F32>(
                        src, dst);
                case GGML_TYPE_F16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_BF16, GGML_TYPE_F16>(
                        src, dst);
                case GGML_TYPE_BF16:
                    return std::forward<F>(f).template operator()<GGML_TYPE_BF16>(src, dst);
                default:
                    GGML_HSA_LOG_ERROR("%s: unsupported type for destination tensor \"%s\" (%s)",
                                       __func__, dst->name, ggml_type_name(dst->type));
                    return GGML_STATUS_FAILED;
            }
        default:
            GGML_HSA_LOG_ERROR("%s: unsupported type for source tensor \"%s\" (%s)", __func__,
                               src->name, ggml_type_name(src->type));
            return GGML_STATUS_FAILED;
    }
}

ggml_status ggml_hsa_copy_tensor(const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_is_contiguous(dst)) {
        return ggml_hsa_assign(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
    }

    if (ggml_are_same_shape(src, dst)) {
        return ggml_hsa_assign(ggml_hsa_copy_same_shape_tensors_f{}, src, dst);
    }

    GGML_HSA_LOG_ERROR("%s: unsupported tensor combination between source \"%s\" (%s) and "
                       "destination tensors \"%s\" (%s)",
                       __func__, src->name, ggml_op_desc(src), dst->name, ggml_op_desc(dst));
    return GGML_STATUS_FAILED;
}

ggml_status ggml_hsa_copy_subblock(const ggml_tensor * src, ggml_tensor * dst) {
    return ggml_hsa_assign(ggml_hsa_copy_subblock_f{}, src, dst);
}

ggml_status ggml_hsa_compute_dup(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(*t) == 1) && (t->type == t->src[0]->type) &&
           ggml_are_same_shape(t, t->src[0]));

    auto * src = t->src[0];
    auto * dst = t;

    if (dst->view_src == src) {
        // destination tensor is a view of the source tensor
        return GGML_STATUS_SUCCESS;
    }

    ggml_hsa_wait_dispatches(ctx);

    if (ggml_is_contiguous(dst)) {
        return ggml_hsa_assign(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
    }

    return ggml_hsa_assign(ggml_hsa_copy_same_shape_tensors_f{}, src, dst);
}

ggml_status ggml_hsa_compute_cpy(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(*t) == 2) && (ggml_nelements(t->src[0]) == ggml_nelements(t->src[1])));

    auto * src = t->src[0];
    auto * dst = t->src[1];

    ggml_hsa_wait_dispatches(ctx);

    return ggml_hsa_copy_tensor(src, dst);
}

ggml_status ggml_hsa_compute_cont(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(*t) == 1) && (t->type == t->src[0]->type) &&
           (ggml_nelements(t) == ggml_nelements(t->src[0])) && ggml_is_contiguous(t));

    auto * src = t->src[0];
    auto * dst = t;

    ggml_hsa_wait_dispatches(ctx);

    return ggml_hsa_assign(ggml_hsa_copy_tensor_to_cont_tensor_f{}, src, dst);
}

ggml_status ggml_hsa_compute_get_rows(ggml_backend_hsa_context & ctx, ggml_tensor * t) {
    assert(ggml_hsa_nsrcs(*t) == 2);

    auto * src0 = t->src[0];  // data table
    auto * src1 = t->src[1];  // int32 row indices
    auto * dst  = t;

    assert(src1->type == GGML_TYPE_I32);

    // Indices may be produced by a preceding device op, so drain before reading them on the host.
    ggml_hsa_wait_dispatches(ctx);

    return ggml_hsa_assign(ggml_hsa_get_rows_f{src1}, src0, dst);
}
