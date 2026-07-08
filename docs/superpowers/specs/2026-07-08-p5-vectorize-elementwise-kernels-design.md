# P5: Vectorize element-wise C++ kernels + pipelining hints

## Goal

Replace the scalar element-wise loops in `scale.cc`, `clamp.cc`, `unary_ops.cc`,
and `binary_ops.cc` with vectorized `aie::` implementations, and add software
pipelining hints. This is item **P5** from `iron-kernels-review.md`.

Scope decision (confirmed with user): **everything possible**, using
**vector-aware templates**, and the rounding ops are **vectorized now** via
`aie::to_fixed`/`aie::to_float` (accepting the known large-magnitude overflow
that C3 addresses separately).

## Constraints discovered in the AIE API headers

- **No operator overloads** on `aie::vector`. All arithmetic must go through
  free functions: `aie::add/sub/mul/div/max/min/abs/neg/sqrt`.
- Those arithmetic free functions have **both scalar and vector overloads**, so
  one generic lambda covers both the vector body and the scalar tail.
- **`aie::select`/`aie::lt`/`aie::ge` have vector overloads only** (no scalar
  form). Ops needing comparison+select (step, sgn, floor, ceil, round) therefore
  need a *different* expression for the scalar tail than for the vector body.
- **No elementwise `aie::floor`/`aie::ceil`/`aie::round`.** Float→int rounding
  uses `aie::to_fixed<int32>(v)` (truncation toward zero) and `aie::to_float`.
  This is exactly the cast C3 flags as overflow-prone for `|v| > 2^31`.
- **No `vec_log`.** `log` stays scalar (`scalar_log`). `sqrt` has `aie::sqrt`.
- `max_tile_size` guarantees `tile_size` divides `num_elements`, but **not** that
  it is a multiple of the vector lane count. Every kernel needs a **vector main
  loop + scalar tail**.

## Mechanism: vector-aware transform templates

Both `transform_n` (unary) and `transform_binary_n` (binary) take **two
functors**: a vector functor and a scalar functor.

```cpp
// V = lanes per 512-bit register for the output type
template <typename T, typename Size, typename VecOp, typename ScalarOp>
void transform_n(const T * __restrict in, Size count, T * __restrict out,
                 VecOp vop, ScalarOp sop) {
    event0();
    constexpr int V = 512 / (sizeof(T) * 8);
    const Size vend = (count / V) * V;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (Size i = 0; i < vend; i += V) {
        auto v = aie::load_v<V>(in + i);
        aie::store_v(out + i, vop(v));
    }
    for (Size i = vend; i < count; ++i) {
        out[i] = sop(in[i]);
    }
    event1();
}
```

For **arithmetic ops** the two functors are the *same* generic lambda
(`[](auto v){ return aie::add(aie::mul(v, s), b); }`), so there is no
duplication — `auto v` binds to either a vector or a scalar and the right
`aie::` overload is selected.

For **divergent ops** (step, sgn, floor, ceil, round) the vector functor uses
`aie::lt`/`aie::ge`/`aie::select`/`aie::to_fixed`, and the scalar functor is the
*existing* ternary/cast expression unchanged. Keeping the current scalar code in
the tail confines C3's present behavior to the tail path.

`transform_binary_n` follows the same shape with two input pointers. It
vectorizes only when `T0 == T1 == TOut` (checked with `if constexpr`); mixed
dtypes take a scalar-only path (unchanged behavior). The **broadcast** variant
(`transform_binary_broadcast_n`) is left scalar — its per-element 4D index
decomposition and `in1` gather do not vectorize with a contiguous load.

`scale.cc` and `clamp.cc` are standalone (not templated). They get the same
vector-body + scalar-tail structure inline.

## Per-kernel functor mapping

### scale.cc  `out = in*scale + bias`
- vec/scalar (same): `aie::add(aie::mul(v, scale), bias)`

### clamp.cc  `out = min(max(in, lo), hi)`
- vec/scalar (same): `aie::min(aie::max(v, lo), hi)`
- also add the missing `event0/event1` — already done in P1/P2; keep.

### unary_ops.cc
Same-functor (arithmetic) ops:
- sqr: `aie::mul(v, v)`
- neg: `aie::neg(v)`
- abs: `aie::abs(v)`
- relu: `aie::max(v, 0)`
- sqrt: `aie::sqrt(v)`
- hardsigmoid: `aie::min(aie::max(aie::mul(aie::add(v,3), 1/6), 0), 1)`
- hardswish: `aie::mul(v, hardsigmoid(v))`

Divergent (vector functor uses select/to_fixed; scalar functor = existing code):
- step: vec `aie::select(0, 1, aie::gt(v, 0))`; scalar `v > 0`
- sgn: vec compose two selects; scalar existing ternary
- floor/ceil/round/trunc: vec via `aie::to_fixed<int32>` (+ correction for
  floor/ceil/round direction) then `aie::to_float` back to the declared
  `OUTPUT_DTYPE`; scalar functor = existing cast expression.
- log: **stays fully scalar** (no vec_log). Keep as a scalar-only call.

### binary_ops.cc
- add/sub/mul/div: `aie::add/sub/mul/div(a, b)` — same functor, vector path
  gated on `T0==T1==TOut`.
- `*_broadcast`: unchanged (scalar).

## Pipelining hints

Add `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_MIN_ITERATION_COUNT(1)` (from
`aie_kernel_utils.h`) to every vector loop. Include `aie_kernel_utils.h` in the
files that don't already (scale.cc, clamp.cc, binary_ops.cc; unary_ops.cc pulls
it via aie_kernel_math.h).

## Testing / verification

- Build the affected kernels for both `aie2` and `aie2p` (they must compile with
  the `-D*_DTYPE` flags the Python drivers pass).
- Run the existing ggml-hsa kernel tests for scale, clamp, unary, binary ops and
  diff outputs against the CPU reference for representative shapes, including at
  least one shape whose `tile_size` is **not** a multiple of the lane count
  (exercises the scalar tail).
- Confirm mixed-dtype binary ops still produce correct results via the scalar
  fallback.

## Out of scope

- C3 (rounding overflow) — the vector rounding path inherits the same overflow;
  C3 fixes both paths later.
- P6 (vec_exp in softmax), P7 (multi-core) — separate items.
- The broadcast binary variant vectorization.
```