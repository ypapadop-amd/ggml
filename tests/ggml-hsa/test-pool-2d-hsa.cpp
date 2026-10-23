// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Standalone test for GGML_OP_POOL_2D (MAX) on the HSA backend, over deterministic windows: NaNs
// in every position, signed zeros in either order, and windows of only -inf.
//
// Random inputs hold none of these, and they are exactly where an order-free max disagrees with
// the reference, ggml_compute_forward_pool_2d (ggml-cpu/ops.cpp), which folds a window's taps in
// row-then-column order from -FLT_MAX:
//
//     res = -FLT_MAX;  for (tap x) { res = std::max(x, res); }   // (x < res) ? res : x
//
// So a tap that is not below the running maximum replaces it: on a tie the later tap wins (a zero
// maximum takes the sign of the window's last zero), a NaN tap becomes the maximum, and the next
// tap then replaces that NaN whatever it is (even -inf, with no -FLT_MAX floor). The answer is
// decided by the taps after the window's last NaN, and is that NaN if it is the last tap.
//
// Results are compared bit for bit, except that any NaN matches any NaN.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-hsa.h"
#include "ggml.h"

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

struct pool_shape {
    const char * name;
    int iw, ih, channels, k0, k1, s0, s1, p0, p1;

    int ow() const { return (iw + 2 * p0 - k0) / s0 + 1; }
    int oh() const { return (ih + 2 * p1 - k1) / s1 + 1; }
};

/// CPU reference: the POOL_MAX branch of ggml_compute_forward_pool_2d, transcribed.
std::vector<float> reference_max_pool(const pool_shape & s, const std::vector<float> & in) {
    const int ow = s.ow(), oh = s.oh();
    std::vector<float> out(static_cast<std::size_t>(ow * oh * s.channels));
    for (int c = 0; c < s.channels; ++c) {
        const float * plane = in.data() + static_cast<std::size_t>(c) * s.iw * s.ih;
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float res = -FLT_MAX;
                for (int ky = 0; ky < s.k1; ++ky) {
                    const int y = oy * s.s1 - s.p1 + ky;
                    if (y < 0 || y >= s.ih) {
                        continue;
                    }
                    for (int kx = 0; kx < s.k0; ++kx) {
                        const int x = ox * s.s0 - s.p0 + kx;
                        if (x < 0 || x >= s.iw) {
                            continue;
                        }
                        res = std::max(plane[y * s.iw + x], res);
                    }
                }
                out[static_cast<std::size_t>((c * oh + oy) * ow + ox)] = res;
            }
        }
    }
    return out;
}

/// Tap sequences, in the reference's window order, written over one window each. The rest of the
/// input is negative and nonzero, so each window's answer comes from its own taps.
const std::vector<std::vector<float>> & window_cases() {
    static const std::vector<std::vector<float>> cases = {
        {9.f, kNaN, 1.f, -5.f},       // a NaN discards the earlier max: 1
        {1.f, 2.f, 3.f, kNaN},        // trailing NaN: NaN
        {kNaN, kNaN, kNaN, kNaN},     // all NaN: NaN
        {9.f, -kNaN, -1.f, -2.f},     // a negative NaN discards too: -1
        {kNaN, -kInf, -kInf, -kInf},  // after a NaN, no -FLT_MAX floor: -inf
        {-kInf, -kInf, -kInf, -kInf}, // only -inf: -FLT_MAX
        {-0.f, -0.f, -0.f, -0.f},     // only -0: -0
        {0.f, -0.f, -1.f, -2.f},      // the last zero wins a tie: -0
        {-0.f, 0.f, -1.f, -2.f},      // ...in either order: +0
        {-0.f, -1.f, -2.f, -3.f},     // a lone -0 max: -0
        {0.f, 0.f, 0.f, 0.f},         // only +0: +0
        {-1.f, 4.f, 2.f, 4.f},        // plain: 4
    };
    return cases;
}

bool has_nan(const std::vector<float> & taps) {
    return std::any_of(taps.begin(), taps.end(), [](float v) { return v != v; });
}

bool has_neg_zero(const std::vector<float> & taps) {
    return std::any_of(taps.begin(), taps.end(),
                       [](float v) { return v == 0.f && std::signbit(v); });
}

/// Which special values an input holds. The kernel sends a tile holding a NaN or a -0 to an exact
/// path, so each kind is also tested alone (each must be detected by itself) and as a lone
/// element at the input's very start and end (detected in the scan's scalar head and tail).
enum class input_kind {
    all_cases,
    nan_cases,
    neg_zero_cases,
    first_nan,
    first_neg_zero,
    last_neg_zero,
    clean
};

const char * kind_name(input_kind k) {
    switch (k) {
        case input_kind::all_cases:
            return "NaN and -0 windows";
        case input_kind::nan_cases:
            return "NaN windows only";
        case input_kind::neg_zero_cases:
            return "-0 windows only";
        case input_kind::first_nan:
            return "lone NaN first";
        case input_kind::first_neg_zero:
            return "lone -0 first";
        case input_kind::last_neg_zero:
            return "lone -0 last";
        case input_kind::clean:
            return "clean";
    }
    return "?";
}

/// The input for @p kind. The *_cases kinds write the cases of window_cases holding that kind
/// (one window each, in raster order, wrapping) in each channel. The lone kinds put one value
/// into otherwise negative, nonzero data: a NaN as its window's first tap, which the reference
/// discards and an order-free max would return, or a -0 as its window's only zero. clean holds
/// zeros and -inf but no NaN or -0, so the order-free path is the one that runs.
std::vector<float> make_input(const pool_shape & s, input_kind kind) {
    std::vector<float> in(static_cast<std::size_t>(s.iw * s.ih * s.channels));
    for (std::size_t i = 0; i < in.size(); ++i) {
        in[i] = -1.f - static_cast<float>(i % 7);
    }
    switch (kind) {
        case input_kind::first_nan:
            in.front() = kNaN;
            return in;
        case input_kind::first_neg_zero:
            in.front() = -0.f;
            return in;
        case input_kind::last_neg_zero:
            in.back() = -0.f;
            return in;
        default:
            break;
    }
    const bool special = kind != input_kind::clean;
    std::vector<std::vector<float>> cases;
    for (const auto & taps : window_cases()) {
        if ((kind == input_kind::all_cases) ||
            (kind == input_kind::nan_cases && has_nan(taps) && !has_neg_zero(taps)) ||
            (kind == input_kind::neg_zero_cases && !has_nan(taps))) {
            cases.push_back(taps);
        }
    }
    std::size_t next = 0;
    for (int c = 0; c < s.channels; ++c) {
        float * plane = in.data() + static_cast<std::size_t>(c) * s.iw * s.ih;
        for (int oy = 0; oy < s.oh(); ++oy) {
            for (int ox = 0; ox < s.ow(); ++ox) {
                if (!special) {
                    const float clean[] = {0.f, -kInf, 2.f, 0.f};
                    const int x = ox * s.s0 - s.p0;
                    const int y = oy * s.s1 - s.p1;
                    if (x >= 0 && y >= 0) {
                        plane[y * s.iw + x] = clean[(ox + oy) % 4];
                    }
                    continue;
                }
                const auto & taps = cases[next++ % cases.size()];
                std::size_t t = 0;
                for (int ky = 0; ky < s.k1 && t < taps.size(); ++ky) {
                    const int y = oy * s.s1 - s.p1 + ky;
                    for (int kx = 0; kx < s.k0 && t < taps.size(); ++kx) {
                        const int x = ox * s.s0 - s.p0 + kx;
                        if (y >= 0 && y < s.ih && x >= 0 && x < s.iw) {
                            plane[y * s.iw + x] = taps[t++];
                        }
                    }
                }
            }
        }
    }
    return in;
}

std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

bool same(float a, float b) {
    if (a != a || b != b) {
        return a != a && b != b;
    }
    return bits(a) == bits(b);
}

/// Runs MAX POOL_2D over @p s and checks every output against the reference.
bool run(ggml_backend_t backend, const pool_shape & s, input_kind input) {
    const std::size_t ctx_size = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params{
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ggml_init(params), ggml_free};

    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s.iw, s.ih, s.channels);
    ggml_tensor * dst =
        ggml_pool_2d(ctx.get(), src, GGML_OP_POOL_MAX, s.k0, s.k1, s.s0, s.s1, s.p0, s.p1);

    const char * kind = kind_name(input);
    if (!ggml_backend_supports_op(backend, dst)) {
        printf("  %-28s %-24s: op not supported\n", s.name, kind);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dst);

    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> galloc{
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free};
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        printf("  %-28s %-24s: graph allocation failed\n", s.name, kind);
        return false;
    }

    const std::vector<float> in = make_input(s, input);
    ggml_backend_tensor_set(src, in.data(), 0, ggml_nbytes(src));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("  %-28s %-24s: graph compute failed\n", s.name, kind);
        return false;
    }

    std::vector<float> got(static_cast<std::size_t>(ggml_nelements(dst)));
    ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
    const std::vector<float> want = reference_max_pool(s, in);

    int bad = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        if (!same(got[i], want[i])) {
            if (bad < 8) {
                printf("    out[%zu]: got %g (0x%08x) want %g (0x%08x)\n", i, got[i], bits(got[i]),
                       want[i], bits(want[i]));
            }
            ++bad;
        }
    }
    printf("  %-28s %-24s: %s (%d of %zu wrong)\n", s.name, kind, bad == 0 ? "OK" : "MISMATCH", bad,
           want.size());
    return bad == 0;
}

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_hsa_init(0);
    if (backend == nullptr) {
        printf("HSA backend unavailable; skipping.\n");
        return 0;
    }

    const pool_shape shapes[] = {
        // name                       iw  ih  ch k0 k1 s0 s1 p0 p1
        {"2x2 stride 2 (MNIST)", 8, 8, 3, 2, 2, 2, 2, 0, 0},
        {"2x2 stride 2, 14 wide", 14, 14, 2, 2, 2, 2, 2, 0, 0},
        {"3x3 stride 1 pad 1", 8, 8, 2, 3, 3, 1, 1, 1, 1},
        {"2x2 stride 1", 9, 6, 2, 2, 2, 1, 1, 0, 0},
        {"3x3 stride 3 (scalar)", 9, 9, 2, 3, 3, 3, 3, 0, 0},
    };

    bool all_ok = true;
    printf("POOL_2D MAX:\n");
    for (const auto & s : shapes) {
        for (input_kind k :
             {input_kind::all_cases, input_kind::nan_cases, input_kind::neg_zero_cases,
              input_kind::first_nan, input_kind::first_neg_zero, input_kind::last_neg_zero,
              input_kind::clean}) {
            all_ok = run(backend, s, k) && all_ok;
        }
    }

    ggml_backend_free(backend);
    printf("%s\n", all_ok ? "ALL PASSED" : "FAILURES");
    return all_ok ? 0 : 1;
}
