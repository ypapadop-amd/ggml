// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#include "ggml-hsa/gemm.hpp"

#include <cstdint>

#include "ggml-hsa/common.hpp"

/**
 * @brief Eligibility for the padded bf16 GEMM path, over a raw tensor.
 *
 * Does not check @c mm.type; @c ggml_hsa_prepare_mul_mat_f32 additionally requires an f32
 * destination.
 *
 * @param[in] mm candidate MUL_MAT node.
 * @return @c true if @p mm is a 2-source MUL_MAT with f32/bf16 operands, trivial layout on both
 *         operands and the destination, and no batch/broadcast.
 */
static bool ggml_hsa_mul_mat_is_padded_gemm(const ggml_tensor & mm) {
    if (mm.op != GGML_OP_MUL_MAT || mm.src[0] == nullptr || mm.src[1] == nullptr ||
        mm.src[2] != nullptr) {
        return false;
    }
    const ggml_tensor & a = *mm.src[0]; // [K, M]
    const ggml_tensor & b = *mm.src[1]; // [K, N]

    // f16 is accepted alongside f32/bf16: ggml_conv_2d's im2col emits f16 whenever the conv kernel
    // itself is not bf16 (see ggml_conv_2d in ggml.c), so the MUL_MAT it feeds would otherwise
    // never qualify for the padded path. ggml_hsa_pad_gemm_operand retypes the operand to bf16
    // unconditionally regardless of its original type. The on-device CONVERT_PAD converts only an
    // f32 source, so its kernel build fails for an f16 operand and that operand takes the host
    // fallback (ggml_hsa_assign) instead, which drains the queue before copying.
    const bool a_ok = a.type == GGML_TYPE_F32 || a.type == GGML_TYPE_BF16 || a.type == GGML_TYPE_F16;
    const bool b_ok = b.type == GGML_TYPE_F32 || b.type == GGML_TYPE_BF16 || b.type == GGML_TYPE_F16;
    if (!a_ok || !b_ok) {
        return false;
    }
    if (!ggml_hsa_has_trivial_layout(a) || !ggml_hsa_has_trivial_layout(b) ||
        !ggml_hsa_has_trivial_layout(mm)) {
        return false;
    }
    // no batching / broadcasting
    if (a.ne[2] != 1 || a.ne[3] != 1 || b.ne[2] != 1 || b.ne[3] != 1) {
        return false;
    }
    return true;
}

/**
 * @brief Points a padded-GEMM operand at an internal buffer of @p type, unless it already matches.
 *
 * An operand that is already @p type at exactly the padded shape needs neither conversion nor
 * padding, so it keeps pointing at the parent buffer: leaving @c buffer_size at 0 skips the
 * internal allocation, the @c CONVERT_PAD dispatch, and (when no operand needs one) the whole
 * source synchronization. Redirecting it anyway costs a full-size device copy into an
 * identically-shaped, identically-typed buffer -- measured at ~350 us per 512x512 operand on
 * aie2p, a third of a 512^3 bf16 GEMM spent copying data to itself.
 *
 * The zero-padding itself is not done here: this only points the internal tensor at a buffer of
 * the padded shape, which the @c CONVERT_PAD dispatch (or the host fallback scatter) fills in full,
 * padding included.
 *
 * @param[in] dev_info device information (supplies the buffer alignment)
 * @param[in,out] source internal source node to retype and resize
 * @param[in] ne0 padded extent of dimension 0 (the K dimension for both operands)
 * @param[in] ne1 padded extent of dimension 1 (Mpad for A, Npad for B)
 * @param[in] type dtype the GEMM kernel consumes this operand in
 */
static void ggml_hsa_pad_gemm_operand(const ggml_hsa_device_info::device_info & dev_info,
                                      ggml_backend_hsa_tensor_extra::source_node_t & source,
                                      std::int64_t ne0,
                                      std::int64_t ne1,
                                      ggml_type type) {
    ggml_tensor & operand = source.tensor;
    if (operand.type == type && operand.ne[0] == ne0 && operand.ne[1] == ne1) {
        return;
    }
    operand.type = type;
    operand.ne[0] = ne0;
    operand.ne[1] = ne1;
    ggml_hsa_set_contiguous_strides(operand);
    operand.data = nullptr;
    source.buffer_size = GGML_PAD(ggml_nbytes(&operand), dev_info.alignment);
}

bool ggml_hsa_prepare_mul_mat_f32(const ggml_hsa_device_info::device_info & dev_info,
                                  ggml_backend_hsa_tensor_extra::node_t & node,
                                  ggml_backend_hsa_tensor_extra::sources_t & sources) {
    ggml_tensor & dst = node.tensor;

    // The GEMM microkernel runs in bf16, so both operands must be f32 (converted to bf16 below) or
    // already bf16 (cast in the graph). Eligibility is checked over the internal source tensors via
    // the raw-tensor predicate; here we additionally require an f32 destination (ggml's native
    // MUL_MAT output).
    if (sources.count != 2) {
        return false;
    }
    ggml_tensor & a = sources[0].tensor; // [K, M]
    ggml_tensor & b = sources[1].tensor; // [K, N]

    // Build a raw view of the node/sources for the shared predicate: node_t.tensor already mirrors
    // the parent's op/type/shape at this point in construction.
    ggml_tensor probe = dst;
    probe.src[0] = &a;
    probe.src[1] = &b;
    probe.src[2] = nullptr;
    if (!ggml_hsa_mul_mat_is_padded_gemm(probe)) {
        return false;
    }
    if (dst.type != GGML_TYPE_F32) {
        return false;
    }

    // Pad to the *microkernel granularity*, which is what gemm.py's select_gemm_tile actually
    // requires -- not to an independent "tile" constant. The granularity is the smallest per-core
    // tile the vectorized wrapper accepts:
    //
    //     (gm, gk, gn) = (row_expand * r, s, col_expand * t)
    //
    // with (r, s, t) the mmul MAC dims (gemm.py microkernel_mac_dim_map) and (row_expand,
    // col_expand) the wrapper's mmul expansion (gemm.py microkernel_expansion_map). Both operands
    // are converted to bf16 below, so only the bf16 row of those tables applies:
    //
    //     aie2  (npu)  bf16: (r,s,t)=(4,8,4), expansion (4,4) -> (gm,gk,gn)=(16,8,16), 4 columns
    //     aie2p (npu2) bf16: (r,s,t)=(4,8,8), expansion (2,2) -> (gm,gk,gn)=( 8,8,16), 8 columns
    //
    // select_gemm_tile then needs M%(gm*n_aie_rows)==0, K%gk==0, N%(gn*n_aie_cols)==0, which is
    // exactly what the padding below guarantees. That alone no longer guarantees a tile:
    // select_gemm_tile also caps the column-group count N/(n*n_aie_cols) at 64 (the shim BD
    // iteration limit). With the padded N = gn*n_aie_cols*q and n = gn*d (d <= max_tile/gn = 16),
    // a tile exists only if q has a divisor d <= 16 with q/d <= 64. Otherwise the kernel build
    // raises and MUL_MAT falls back to the CPU: e.g. N = 8576 = 128*67 on aie2p. Padding N further
    // (to a q that factors) would keep such shapes on the NPU; not done yet.
    //
    // The previous constants (tile=16 on aie2p, 32 on aie2, applied to all three dimensions)
    // over-padded K and M: they were a single number standing in for three different granularities.
    // tests/ggml-hsa/test_gemm_tiling.py pins these values against gemm.py's own tables.
    constexpr std::int64_t n_aie_rows = 4;
    std::int64_t gm = 0;
    std::int64_t gk = 0;
    std::int64_t gn = 0;
    std::int64_t n_aie_cols = 0;
    if (dev_info.name == "aie2p") {
        gm = 8;
        gk = 8;
        gn = 16;
        n_aie_cols = 8;
    } else { // aie2
        gm = 16;
        gk = 8;
        gn = 16;
        n_aie_cols = 4;
    }

    const std::int64_t K = a.ne[0];
    const std::int64_t M = a.ne[1];
    const std::int64_t N = b.ne[1];

    const std::int64_t Kpad = GGML_PAD(K, gk);
    const std::int64_t Mpad = GGML_PAD(M, gm * n_aie_rows);
    const std::int64_t Npad = GGML_PAD(N, gn * n_aie_cols);

    // Rewrite sources to padded bf16; only sources that arrive as f32 need a dtype conversion,
    // bf16 sources are just zero-padded to the tile multiples (handled by the pre-processing
    // kernel below, which is selected from the parent tensor's own dtype). An operand that needs
    // neither is left alone -- see ggml_hsa_pad_gemm_operand.
    //
    // The exception is an f32 B, which the GEMM converts on the core and, once B is at
    // least one column group wide, reads unpadded (see gemm.hpp). Its K-tail read runs up to
    // Kpad - K elements past B's last column, which only the read slack of a buffer the HSA buffer
    // type allocated covers (ggml_hsa_buffer_has_read_slack); a B in an imported buffer is padded
    // instead. A B without a buffer yet (the supports_op probe) is taken to read unpadded, since
    // every buffer ggml's allocators create for it comes from the HSA buffer type. A manual,
    // out-of-order allocation -- the consumer initialized first, B then aliased into an imported
    // buffer -- keeps the unpadded kernel and can read up to 28 bytes past the dma-buf; unguarded.
    const bool b_f32_on_core = b.type == GGML_TYPE_F32;
    const bool b_unpadded =
        b_f32_on_core && N >= gn * n_aie_cols &&
        (K == Kpad || b.buffer == nullptr || ggml_hsa_buffer_has_read_slack(b.buffer));
    ggml_hsa_pad_gemm_operand(dev_info, sources[0], Kpad, Mpad, GGML_TYPE_BF16);
    if (!b_unpadded) {
        ggml_hsa_pad_gemm_operand(dev_info, sources[1], Kpad, Npad,
                                  b_f32_on_core ? GGML_TYPE_F32 : GGML_TYPE_BF16);
    }

    // The output is not rewritten: the GEMM writes the dense parent directly. Over padded operands
    // its mem tiles read back only the valid part of each C block; with an unpadded f32 B
    // it shifts its last column group (and, when M is at least one row block, its last row block)
    // back inside the parent instead (kernels/iron_kernels/gemm_c_plan.py). Either way there is no
    // internal output buffer and nothing to post-process.

    return true;
}
