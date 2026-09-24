// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

// Unit tests for ggml_hsa_getenv_int, the shared parser behind the backend's integer environment
// variables (GGML_HSA_DISPATCH_BATCH_SIZE, GGML_HSA_QUEUE_ERROR_DRAIN_TIMEOUT_MS).
//
// The range check is the point of this test. std::from_chars accepts a leading '-' for a signed T,
// so a negative value parses cleanly; GGML_HSA_QUEUE_ERROR_DRAIN_TIMEOUT_MS=-1 used to be taken as
// a valid timeout and produced a deadline already in the past, which turned the bounded teardown
// drain into an instant timeout that silently leaked the dispatch signal. An out-of-range value
// must be rejected and fall back, not accepted.
//
// Needs no device: the parser is pure, so this runs anywhere the backend builds.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "ggml-hsa/common.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++g_failures;
    }
}

/// @brief Sets @p name to @p value (or unsets it when @p value is null) and parses it.
template <typename T> T parse_with(const char * value, T fallback, T min, T max) {
    constexpr const char * name = "GGML_HSA_TEST_ENV_CONFIG";
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, /*overwrite=*/1);
    }
    const T parsed = ggml_hsa_getenv_int<T>(name, fallback, min, max);
    unsetenv(name);
    return parsed;
}

void test_rejects_out_of_range() {
    using rep = std::chrono::milliseconds::rep;
    constexpr rep fallback = 1000;
    constexpr rep min = 0;
    constexpr rep max = 60 * 60 * 1000;

    // The regression: negative parses cleanly through std::from_chars but must not be accepted.
    check(parse_with<rep>("-1", fallback, min, max) == fallback, "negative value falls back");
    check(parse_with<rep>("-2147483648", fallback, min, max) == fallback,
          "large negative value falls back");

    // Above the bound that keeps steady_clock::now() + timeout from overflowing.
    check(parse_with<rep>("9223372036854775807", fallback, min, max) == fallback,
          "value above max falls back");
    check(parse_with<rep>(std::to_string(max + 1).c_str(), fallback, min, max) == fallback,
          "value one past max falls back");
}

void test_rejects_malformed() {
    constexpr std::size_t fallback = 0;
    constexpr std::size_t min = 1;
    constexpr std::size_t max = 1024;

    check(parse_with<std::size_t>("", fallback, min, max) == fallback, "empty value falls back");
    check(parse_with<std::size_t>("abc", fallback, min, max) == fallback,
          "non-numeric value falls back");
    check(parse_with<std::size_t>("12abc", fallback, min, max) == fallback,
          "trailing garbage falls back");
    check(parse_with<std::size_t>("12 ", fallback, min, max) == fallback,
          "trailing space falls back");

    // GGML_HSA_DISPATCH_BATCH_SIZE uses min == 1, so zero means "unset", not "batch of zero".
    check(parse_with<std::size_t>("0", fallback, min, max) == fallback,
          "value below min falls back");
}

void test_accepts_valid() {
    using rep = std::chrono::milliseconds::rep;

    check(parse_with<rep>("0", rep{1000}, rep{0}, rep{3600000}) == 0, "min boundary accepted");
    check(parse_with<rep>("250", rep{1000}, rep{0}, rep{3600000}) == 250, "in-range accepted");
    check(parse_with<rep>("3600000", rep{1000}, rep{0}, rep{3600000}) == 3600000,
          "max boundary accepted");
    check(parse_with<std::size_t>("64", std::size_t{0}, std::size_t{1},
                                  std::size_t{1024}) == 64,
          "unsigned in-range accepted");
}

void test_unset_uses_fallback() {
    check(parse_with<std::size_t>(nullptr, std::size_t{7}, std::size_t{1},
                                 std::size_t{1024}) == 7,
          "unset variable falls back");
}

} // namespace

int main() {
    test_rejects_out_of_range();
    test_rejects_malformed();
    test_accepts_valid();
    test_unset_uses_fallback();

    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All env config checks passed\n";
    return EXIT_SUCCESS;
}
