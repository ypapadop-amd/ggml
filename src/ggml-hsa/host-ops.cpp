// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/host-ops.hpp"

#include "ggml-hsa/common.hpp"

#include <cassert>

ggml_status ggml_hsa_compute_dup(ggml_backend_hsa_context & /*ctx*/, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 1) && (t->type == t->src[0]->type) &&
           ggml_are_same_shape(t, t->src[0]) && ggml_are_same_stride(t, t->src[0]));

    auto * src = t->src[0];
    auto * dst = t;

    if (dst->view_src == src) {
        // destination tensor is a view of the source tensor
        return GGML_STATUS_SUCCESS;
    }

    // TODO

    return GGML_STATUS_FAILED;
}

ggml_status ggml_hsa_compute_cpy(ggml_backend_hsa_context & /*ctx*/, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 2) && ggml_are_same_shape(t->src[0], t->src[1]) &&
           ggml_are_same_stride(t->src[0], t->src[1]));

    auto * src = t->src[0];
    auto * dst = t->src[1];

    // TODO

    return GGML_STATUS_FAILED;
}

ggml_status ggml_hsa_compute_cont(ggml_backend_hsa_context & /*ctx*/, ggml_tensor * t) {
    assert((ggml_hsa_nsrcs(t) == 1) && (t->type == t->src[0]->type) &&
           (ggml_nelements(t) == ggml_nelements(t->src[0])));

    auto * src = t->src[0];
    auto * dst = t;

    // TODO

    return GGML_STATUS_FAILED;
}
