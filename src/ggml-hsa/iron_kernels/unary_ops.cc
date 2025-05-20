#include <aie_api/aie.hpp>

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
// using bf16 = ???;
using f32 = float;

template <typename T, typename Size, typename UnaryOp>
void transform_n(const T * __restrict in, Size count, T * __restrict out, UnaryOp op) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in[i]);
    }
    event1();
}

extern "C" {

#ifdef ABS

void FUNC_NAME(const INPUT_DTYPE * __restrict in, OUTPUT_DTYPE * __restrict out, int32_t N) {
    transform_n(in, N, out, [](auto v) { return abs(v); });
}

#endif // ABS

} // extern "C"
