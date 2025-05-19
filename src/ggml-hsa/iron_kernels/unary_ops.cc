#include <aie_api/aie.hpp>

template <typename T, typename Size, typename UnaryOp>
void transform_n(const T * __restrict in, Size count, T * __restrict out, UnaryOp op) {
    event0();
    for (Size i = 0; i < count; ++i) {
        out[i] = op(in[i]);
    }
    event1();
}

// STRING TO TYPE TRANSLATION

#include <algorithm>
#include <cstdint>

template <std::size_t Size> struct StringLiteral {
    char value[Size] = {};

    constexpr StringLiteral(const char (&s)[Size]) { std::copy_n(s, Size, value); }

    constexpr friend bool operator==(const StringLiteral & sl, const char * s) {
        auto * v = sl.value;
        for (; *v != '\0' || *s != '\0';) {
            if (*v++ != *s++) {
                return false;
            }
        }
        return (*v == '\0' && *s == '\0');
    }
};

template <StringLiteral S> constexpr auto convert_to_cxx_type() {
    if constexpr (S == "i8") {
        return std::int8_t{};
    } else if constexpr (S == "i16") {
        return std::int16_t{};
    } else if constexpr (S == "i32") {
        return std::int32_t{};
    }
#if 0
    else if constexpr (S == "bloat16") {
        return ???;
    }
#endif
    else if constexpr (S == "f32") {
        return float{};
    }
}

template <StringLiteral s>
using convert_to_input_cxx_type_t = const decltype(convert_to_cxx_type<s>());

template <StringLiteral s> using convert_to_output_cxx_type_t = decltype(convert_to_cxx_type<s>());

////

extern "C" {

#ifdef ABS

void abs_f32_f32(convert_to_input_cxx_type_t<INPUT_DTYPE> * __restrict in,
                 convert_to_output_cxx_type_t<OUTPUT_DTYPE> * __restrict out,
                 int32_t N) {
    transform_n(in, N, out, [](auto v) { return abs(v); });
}

#endif // ABS

} // extern "C"
