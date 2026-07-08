// This file defines tests for various GGML ops and backends.
// For the forward pass it asserts that the results of multiple backends computing the same GGML ops
// are consistent. For the backward pass it asserts that the gradients from backpropagation are
// consistent with the gradients obtained via the method of finite differences ("grad" mode, this is
// optional). It is also possible to check the performance ("perf" mode).
//
// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be
// tested, and section 3 defines which tests to run. Quick start for adding a new GGML op: Go to
// section 2 and create a struct that inherits from test_case, then go to section 3 and add an
// instantiation of your struct.

// ##############################
// ## Section 1: General Setup ##
// ##############################

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <future>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef __EMSCRIPTEN__
#define N_THREADS 1
#else
#define N_THREADS std::thread::hardware_concurrency()
#endif

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = N_THREADS;
        // static RNG initialization (revisit if n_threads stops being constant)
        static std::vector<std::default_random_engine> generators = []() {
            std::random_device rd;
            std::vector<std::default_random_engine> vec;
            vec.reserve(n_threads);
            // for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
            for (size_t i = 0; i < n_threads; i++) {
                vec.emplace_back(rd());
            }
            return vec;
        }();

        auto init_thread = [&](size_t ith, size_t start, size_t end) {
            std::uniform_real_distribution<float> distribution(min, max);
            auto & gen = generators[ith];
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        if (n_threads == 1) {
            init_thread(0, 0, nels);
        } else {
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start = i * nels / n_threads;
                size_t end = (i + 1) * nels / n_threads;
                tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 ||
               tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

        // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without
            // imatrix use one of the random numbers to decide
            if (data[0] > 0.5f * (min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), start * blck_size,
                                    end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_quant_threads =
                std::min<size_t>(std::max<size_t>(N_THREADS / 2, 1),
                                 std::max<size_t>(1, n_blocks / min_blocks_per_thread));

            if (n_quant_threads == 1) {
                // single-threaded quantization: do all blocks in the current thread
                quantize_thread(0, n_blocks);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_quant_threads);
                for (size_t i = 0; i < n_quant_threads; i++) {
                    size_t start = i * n_blocks / n_quant_threads;
                    size_t end = (i + 1) * n_blocks / n_quant_threads;
                    tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 ||
               tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific
        // values are again not really meaningful.
        const size_t nbytes_half = ggml_nbytes(tensor) / 2;
        ggml_backend_tensor_set(tensor, data.data(), 0 * nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1 * nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto * tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 / bs * t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t *)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t *)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *)&buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t *)&buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *)&buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *)&buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *)&buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter
// out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the
//     numerical calculation can fail.
static double mean_abs_asymm(const float * a,
                             const float * b,
                             const size_t n,
                             const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum / nvalid;
}

// utils for printing the variables of the test cases

static std::string var_to_str(const std::string & x) { return x; }

template <typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template <typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template <typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) { return ggml_type_name(type); }

static std::string var_to_str(ggml_prec prec) { return prec == GGML_PREC_F32 ? "f32" : "def"; }

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:
            return "avg";
        case GGML_OP_POOL_MAX:
            return "max";
        default:
            return std::to_string(pool);
    }
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i)                                                    \
    VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j)                                                \
    VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k)                                             \
    VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l)                                          \
    VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)
#define VARS_TO_STR13(a, b, c, d, e, f, g, h, i, j, k, l, m)                                       \
    VAR_TO_STR(a) + "," + VARS_TO_STR12(b, c, d, e, f, g, h, i, j, k, l, m)
#define VARS_TO_STR14(a, b, c, d, e, f, g, h, i, j, k, l, m, n)                                    \
    VAR_TO_STR(a) + "," + VARS_TO_STR13(b, c, d, e, f, g, h, i, j, k, l, m, n)
#define VARS_TO_STR15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)                                 \
    VAR_TO_STR(a) + "," + VARS_TO_STR14(b, c, d, e, f, g, h, i, j, k, l, m, n, o)
#define VARS_TO_STR16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)                              \
    VAR_TO_STR(a) + "," + VARS_TO_STR15(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) { return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000; }
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) { return _isinf(f) || f == FLT_MAX || f == -FLT_MAX; }

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}

static bool backend_has_feature(ggml_backend_t backend, const char * feature_name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    auto get_features = (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_get_features");
    if (!get_features) {
        return false;
    }

    const ggml_backend_feature * features = get_features(reg);
    if (!features) {
        return false;
    }

    for (const ggml_backend_feature * f = features; f->name; ++f) {
        if (strcmp(f->name, feature_name) == 0 && strcmp(f->value, "1") == 0) {
            return true;
        }
    }
    return false;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
    MODE_SUPPORT,
};

// Output format support similar to llama-bench
enum output_formats { CONSOLE, SQL, CSV };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CONSOLE:
            return "console";
        case SQL:
            return "sql";
        case CSV:
            return "csv";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "console") {
        format = CONSOLE;
    } else if (s == "sql") {
        format = SQL;
    } else if (s == "csv") {
        format = CSV;
    } else {
        return false;
    }
    return true;
}

// Test result structure for SQL output
struct test_result {
    std::string test_time;
    std::string build_commit;
    std::string backend_name;
    std::string op_name;
    std::string op_params;
    std::string test_mode;
    bool supported;
    bool passed;
    std::string error_message;
    double time_us;
    double flops;
    double bandwidth_gb_s;
    size_t memory_kb;
    int n_runs;
    std::string device_description;
    std::string backend_reg_name;

    test_result() {
        // Initialize with default values
        time_us = 0.0;
        flops = 0.0;
        bandwidth_gb_s = 0.0;
        memory_kb = 0;
        n_runs = 0;
        supported = false;
        passed = false;

        // Set test time
        time_t t = time(NULL);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }

    test_result(const std::string & backend_name,
                const std::string & op_name,
                const std::string & op_params,
                const std::string & test_mode,
                bool supported,
                bool passed,
                const std::string & error_message = "",
                double time_us = 0.0,
                double flops = 0.0,
                double bandwidth_gb_s = 0.0,
                size_t memory_kb = 0,
                int n_runs = 0,
                const std::string & device_description = "",
                const std::string & backend_reg_name = "") :
        backend_name(backend_name),
        op_name(op_name),
        op_params(op_params),
        test_mode(test_mode),
        supported(supported),
        passed(passed),
        error_message(error_message),
        time_us(time_us),
        flops(flops),
        bandwidth_gb_s(bandwidth_gb_s),
        memory_kb(memory_kb),
        n_runs(n_runs),
        device_description(device_description),
        backend_reg_name(backend_reg_name) {
        // Set test time
        time_t t = time(NULL);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // Set build info
        build_commit = ggml_commit();
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "test_time",       "build_commit", "backend_name",
            "op_name",         "op_params",    "test_mode",
            "supported",       "passed",       "error_message",
            "time_us",         "flops",        "bandwidth_gb_s",
            "memory_kb",       "n_runs",       "device_description",
            "backend_reg_name"};
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "supported" || field == "passed") {
            return BOOL;
        }
        if (field == "memory_kb" || field == "n_runs") {
            return INT;
        }
        if (field == "time_us" || field == "flops" || field == "bandwidth_gb_s") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        return {test_time,
                build_commit,
                backend_name,
                op_name,
                op_params,
                test_mode,
                std::to_string(supported),
                std::to_string(passed),
                error_message,
                std::to_string(time_us),
                std::to_string(flops),
                std::to_string(bandwidth_gb_s),
                std::to_string(memory_kb),
                std::to_string(n_runs),
                device_description,
                backend_reg_name};
    }
};

// Printer classes for different output formats
enum class test_status_t { NOT_SUPPORTED, OK, FAIL, SKIPPED };

struct test_operation_info {
    std::string op_name;
    std::string op_params;
    std::string backend_name;
    test_status_t status = test_status_t::OK;
    std::string failure_reason;

    // Additional information fields that were previously in separate structs
    std::string error_component;
    std::string error_details;

    // Gradient info
    int64_t gradient_index = -1;
    std::string gradient_param_name;
    float gradient_value = 0.0f;

    // MAA error info
    double maa_error = 0.0;
    double maa_threshold = 0.0;

    // Flags for different types of information
    bool has_error = false;
    bool has_gradient_info = false;
    bool has_maa_error = false;
    bool is_compare_failure = false;
    bool is_large_tensor_skip = false;

    test_operation_info() = default;

    test_operation_info(const std::string & op_name,
                        const std::string & op_params,
                        const std::string & backend_name,
                        test_status_t status = test_status_t::OK,
                        const std::string & failure_reason = "") :
        op_name(op_name),
        op_params(op_params),
        backend_name(backend_name),
        status(status),
        failure_reason(failure_reason) {}

    // Set error information
    void set_error(const std::string & component, const std::string & details) {
        has_error = true;
        error_component = component;
        error_details = details;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set gradient information
    void set_gradient_info(int64_t index, const std::string & param_name, float value) {
        has_gradient_info = true;
        gradient_index = index;
        gradient_param_name = param_name;
        gradient_value = value;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set MAA error information
    void set_maa_error(double error, double threshold) {
        has_maa_error = true;
        maa_error = error;
        maa_threshold = threshold;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set compare failure
    void set_compare_failure() {
        is_compare_failure = true;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set large tensor skip
    void set_large_tensor_skip() { is_large_tensor_skip = true; }
};

struct test_summary_info {
    size_t tests_passed;
    size_t tests_total;
    bool is_backend_summary = false; // true for backend summary, false for test summary

    test_summary_info() = default;

    test_summary_info(size_t tests_passed, size_t tests_total, bool is_backend_summary = false) :
        tests_passed(tests_passed),
        tests_total(tests_total),
        is_backend_summary(is_backend_summary) {}
};

struct testing_start_info {
    size_t device_count;

    testing_start_info() = default;

    testing_start_info(size_t device_count) : device_count(device_count) {}
};

struct backend_init_info {
    size_t device_index;
    size_t total_devices;
    std::string device_name;
    bool skipped = false;
    std::string skip_reason;
    std::string description;
    size_t memory_total_mb = 0;
    size_t memory_free_mb = 0;
    bool has_memory_info = false;

    backend_init_info() = default;

    backend_init_info(size_t device_index,
                      size_t total_devices,
                      const std::string & device_name,
                      bool skipped = false,
                      const std::string & skip_reason = "",
                      const std::string & description = "",
                      size_t memory_total_mb = 0,
                      size_t memory_free_mb = 0,
                      bool has_memory_info = false) :
        device_index(device_index),
        total_devices(total_devices),
        device_name(device_name),
        skipped(skipped),
        skip_reason(skip_reason),
        description(description),
        memory_total_mb(memory_total_mb),
        memory_free_mb(memory_free_mb),
        has_memory_info(has_memory_info) {}
};

struct backend_status_info {
    std::string backend_name;
    test_status_t status;

    backend_status_info() = default;

    backend_status_info(const std::string & backend_name, test_status_t status) :
        backend_name(backend_name), status(status) {}
};

struct overall_summary_info {
    size_t backends_passed;
    size_t backends_total;
    bool all_passed;

    overall_summary_info() = default;

    overall_summary_info(size_t backends_passed, size_t backends_total, bool all_passed) :
        backends_passed(backends_passed), backends_total(backends_total), all_passed(all_passed) {}
};

struct printer {
    virtual ~printer() {}

    FILE * fout = stdout;

    virtual void print_header() {}

    virtual void print_test_result(const test_result & result) = 0;

    virtual void print_footer() {}

    virtual void print_operation(const test_operation_info & info) { (void)info; }

    virtual void print_summary(const test_summary_info & info) { (void)info; }

    virtual void print_testing_start(const testing_start_info & info) { (void)info; }

    virtual void print_backend_init(const backend_init_info & info) { (void)info; }

    virtual void print_backend_status(const backend_status_info & info) { (void)info; }

    virtual void print_overall_summary(const overall_summary_info & info) { (void)info; }

    virtual void print_failed_tests(const std::vector<std::string> & failed_tests) {
        (void)failed_tests;
    }
};

struct console_printer : public printer {
    void print_test_result(const test_result & result) override {
        if (result.test_mode == "test") {
            print_test_console(result);
        } else if (result.test_mode == "perf") {
            print_perf_console(result);
        } else if (result.test_mode == "support") {
            print_support_console(result);
        }
    }

    void print_operation(const test_operation_info & info) override {
        printf("  %s(%s): ", info.op_name.c_str(), info.op_params.c_str());
        fflush(stdout);

        // Handle large tensor skip first
        if (info.is_large_tensor_skip) {
            printf("skipping large tensors for speed \n");
            return;
        }

        // Handle not supported status
        if (info.status == test_status_t::NOT_SUPPORTED) {
            if (!info.failure_reason.empty()) {
                printf("not supported [%s]\n", info.failure_reason.c_str());
            } else {
                printf("not supported [%s]\n", info.backend_name.c_str());
            }
            return;
        }

        // Handle errors and additional information
        if (info.has_error) {
            if (info.error_component == "allocation") {
                fprintf(stderr, "failed to allocate tensors [%s] ", info.backend_name.c_str());
            } else if (info.error_component == "backend") {
                fprintf(stderr, "  Failed to initialize %s backend\n", info.backend_name.c_str());
            } else {
                fprintf(stderr, "Error in %s: %s\n", info.error_component.c_str(),
                        info.error_details.c_str());
            }
        }

        // Handle gradient info
        if (info.has_gradient_info) {
            printf("[%s] nonfinite gradient at index %" PRId64 " (%s=%f) ", info.op_name.c_str(),
                   info.gradient_index, info.gradient_param_name.c_str(), info.gradient_value);
        }

        // Handle MAA error
        if (info.has_maa_error) {
            printf("[%s] MAA = %.9f > %.9f ", info.op_name.c_str(), info.maa_error,
                   info.maa_threshold);
        }

        // Handle compare failure
        if (info.is_compare_failure) {
            printf("compare failed ");
        }

        // Print final status
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_summary(const test_summary_info & info) override {
        if (info.is_backend_summary) {
            printf("%zu/%zu backends passed\n", info.tests_passed, info.tests_total);
        } else {
            printf("  %zu/%zu tests passed\n", info.tests_passed, info.tests_total);
        }
    }

    void print_backend_status(const backend_status_info & info) override {
        printf("  Backend %s: ", info.backend_name.c_str());
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_testing_start(const testing_start_info & info) override {
        printf("Testing %zu devices\n\n", info.device_count);
    }

    void print_backend_init(const backend_init_info & info) override {
        printf("Backend %zu/%zu: %s\n", info.device_index + 1, info.total_devices,
               info.device_name.c_str());

        if (info.skipped) {
            printf("  %s\n", info.skip_reason.c_str());
            return;
        }

        if (!info.description.empty()) {
            printf("  Device description: %s\n", info.description.c_str());
        }

        if (info.has_memory_info) {
            printf("  Device memory: %zu MB (%zu MB free)\n", info.memory_total_mb,
                   info.memory_free_mb);
        }

        printf("\n");
    }

    void print_overall_summary(const overall_summary_info & info) override {
        printf("%zu/%zu backends passed\n", info.backends_passed, info.backends_total);
        if (info.all_passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_failed_tests(const std::vector<std::string> & failed_tests) override {
        if (failed_tests.empty()) {
            return;
        }

        printf("\nFailing tests:\n");
        for (const auto & test_name : failed_tests) {
            printf("  %s\n", test_name.c_str());
        }
    }

  private:
    void print_test_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported [%s] ", result.backend_name.c_str());
            printf("\n");
            return;
        }

        if (result.passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_perf_console(const test_result & result) {
        int len = printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported\n");
            return;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        printf("    %8d runs - %8.2f us/run - ", result.n_runs, result.time_us);

        if (result.flops > 0) {
            auto format_flops = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                } else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                } else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%6.2f kFLOP", flops / 1e3);
                }
                return buf;
            };
            uint64_t op_flops_per_run = result.flops * result.time_us / 1e6;
            printf("%s/run - \033[1;34m%sS\033[0m", format_flops(op_flops_per_run).c_str(),
                   format_flops(result.flops).c_str());
        } else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m", result.memory_kb,
                   result.bandwidth_gb_s);
        }
        printf("\n");
    }

    void print_support_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (result.supported) {
            printf("\033[1;32mSUPPORTED\033[0m\n");
        } else {
            printf("\033[1;31mNOT SUPPORTED\033[0m\n");
        }
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test_result::get_field_type(field)) {
            case test_result::STRING:
                return "TEXT";
            case test_result::BOOL:
            case test_result::INT:
                return "INTEGER";
            case test_result::FLOAT:
                return "REAL";
            default:
                GGML_ABORT("invalid field type");
        }
    }

    void print_header() override {
        std::vector<std::string> fields = test_result::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS test_backend_ops (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields[i].c_str(), get_sql_field_type(fields[i]).c_str(),
                    i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n\n");
    }

    void print_test_result(const test_result & result) override {
        fprintf(fout, "INSERT INTO test_backend_ops (");
        std::vector<std::string> fields = test_result::get_fields();
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "%s%s", fields[i].c_str(), i < fields.size() - 1 ? ", " : "");
        }
        fprintf(fout, ") VALUES (");
        std::vector<std::string> values = result.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values[i].c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

struct csv_printer : public printer {
    void print_header() override {

        std::vector<std::string> fields = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();
        for (size_t i = 0; i < fields.size(); i++) {
            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) ==
                std::end(fields_csv)) {
                continue;
            }
            printf("\"%s\"%s", fields[i].c_str(), i < fields.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    void print_test_result(const test_result & result) override {

        std::vector<std::string> values = result.get_values();
        std::vector<std::string> fields = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();

        for (size_t i = 0; i < values.size(); i++) {

            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) ==
                std::end(fields_csv)) {
                continue;
            }

            // Escape quotes and wrap in quotes for CSV
            std::string escaped_value = values[i];
            size_t pos = 0;
            while ((pos = escaped_value.find("\"", pos)) != std::string::npos) {
                escaped_value.replace(pos, 1, "\"\"");
                pos += 2;
            }
            printf("\"%s\"%s", escaped_value.c_str(), i < values.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    static std::vector<std::string> get_fields_csv() {
        return {
            "op_name",   "op_params",        "supported",    "error_message",
            "test_mode", "backend_reg_name", "backend_name",
        };
    }
};

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case CONSOLE:
            return std::make_unique<console_printer>();
        case SQL:
            return std::make_unique<sql_printer>();
        case CSV:
            return std::make_unique<csv_printer>();
    }
    GGML_ABORT("invalid output format");
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) { return ggml_op_desc(t); }

    virtual std::string vars() { return ""; }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() { return 1e-7; }

    virtual double max_nmse_err(ggml_backend_t backend) {
        GGML_UNUSED(backend);
        return max_nmse_err();
    }

    virtual double max_maa_err() { return 1e-4; }

    virtual double max_err() { return max_nmse_err(); }

    virtual double max_err(ggml_backend_t backend) { return max_nmse_err(backend); }

    virtual double err(const float * a, const float * b, size_t n) { return nmse(a, b, n); }

    virtual float grad_eps() { return 1e-1f; }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() { return false; }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed
    // up the tests).
    virtual int64_t grad_nmax() { return 10000; }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the
    // values. Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using
    // finite differences is unreliable.
    virtual std::vector<float> grad_expect() { return {}; }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    virtual bool run_whole_graph() { return false; }
    virtual std::vector<ggml_tensor *> fusion_test_nodes() { return {}; }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    std::string current_op_name;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD || mode == MODE_SUPPORT) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the
    // backend

    ggml_tensor *
    ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor *
    ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(
        ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    // Checks an op against the test filter, which is a comma separated list of OP names or specific
    // variations
    bool matches_filter(ggml_tensor * op, const char * op_names_filter) {
        if (op_names_filter) {
            const auto op_name = op_desc(op);
            const auto op_full_name = op_name + "(" + vars() + ")";
            std::string_view filter(op_names_filter);
            while (!filter.empty()) {
                auto comma_pos = filter.find_first_of(',');
                const auto lparen_pos = filter.find_first_of('(');
                if (lparen_pos < comma_pos) {
                    auto rparen_pos = filter.find_first_of(')');
                    comma_pos = filter.find_first_of(',', rparen_pos);
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_full_name) {
                        return true;
                    }
                } else {
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_name) {
                        return true;
                    }
                }
                filter = comma_pos != std::string_view::npos ? filter.substr(comma_pos + 1) : "";
            }
            return false;
        } else {
            return true;
        }
    }

    test_status_t eval(ggml_backend_t backend1,
                       ggml_backend_t backend2,
                       const char * op_names_filter,
                       printer * output_printer) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);
        current_op_name = op_desc(out);

        if (!matches_filter(out, op_names_filter)) {
            // printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return test_status_t::SKIPPED;
        }

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
                 t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    supported = false;
                    break;
                }
            }
        }

        if (!supported) {
            // Create test result for unsupported operation
            test_result result(ggml_backend_name(backend1), current_op_name, vars(), "test", false,
                               false, "not supported");

            if (output_printer) {
                output_printer->print_test_result(result);
            }

            ggml_free(ctx);
            return test_status_t::NOT_SUPPORTED;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return test_status_t::FAIL;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool ok;
            test_case * tc;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud{
            true,
            this,
            backend1,
            backend2,
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2,
                           void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *)user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i],
                           bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1,
                                   f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2,
                               f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
            if (err > ud->tc->max_err(ud->backend1)) {
                printf("[%s] ERR = %.9f > %.9f ", ggml_op_desc(t1), err,
                       ud->tc->max_err(ud->backend1));
                // for (int i = 0; i < (int) f1.size(); i++) {
                //     printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                // }
                // printf("\n");
                // exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        std::vector<ggml_tensor *> fused_nodes_to_verify = fusion_test_nodes();
        if (fused_nodes_to_verify.size() == 0 && run_whole_graph()) {
            fused_nodes_to_verify.push_back(out);
        }
        const bool cmp_ok = ggml_backend_compare_graph_backend(
            backend1, backend2, gf, callback, &ud,
            run_whole_graph() ? fused_nodes_to_verify.data() : nullptr,
            fused_nodes_to_verify.size());

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        // Create test result
        bool test_passed = ud.ok && cmp_ok;
        std::string error_msg = test_passed ? "" : (!cmp_ok ? "compare failed" : "test failed");
        test_result result(ggml_backend_name(backend1), current_op_name, vars(), "test", supported,
                           test_passed, error_msg);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return test_passed ? test_status_t::OK : test_status_t::FAIL;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 +
                ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        ggml_tensor * out = build_graph(ctx.get());
        current_op_name = op_desc(out);
        if (!matches_filter(out, op_names_filter)) {
            // printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        if (!ggml_backend_supports_op(backend, out)) {
            // Create test result for unsupported performance test
            test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", false,
                               false, "not supported");

            output_printer->print_test_result(result);

            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr

        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx.get());

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }

        // determine number of runs
        int n_runs;
        bool is_cpu =
            ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu = 8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t target_flops = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf),
                                            target_flops / op_flops(out)) +
                     1;
        } else {
            // based on memory size
            const size_t GB = 1ULL << 30;
            const size_t target_size_cpu = 8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t target_size = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf),
                                            target_size / op_size(out)) +
                     1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                        ggml_status_to_string(status));
                return false;
            }
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000 * 1000); // run for at least 1 second

        // Create test result
        double avg_time_us = (double)total_time_us / total_runs;
        double calculated_flops =
            (op_flops(out) > 0) ? (op_flops(out) * total_runs) / (total_time_us / 1e6) : 0.0;
        double calculated_bandwidth =
            (op_flops(out) == 0) ? total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0
                                 : 0.0;
        size_t calculated_memory_kb = op_size(out) / 1024;

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", true, true,
                           "", avg_time_us, calculated_flops, calculated_bandwidth,
                           calculated_memory_kb, total_runs);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return true;
    }

    bool
    eval_support(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_SUPPORT;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 +
                ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);

        ggml_tensor * out = build_graph(ctx.get());
        current_op_name = op_desc(out);

        if (!matches_filter(out, op_names_filter)) {
            return true;
        }

        bool supported = ggml_backend_supports_op(backend, out);

        std::string device_desc = ggml_backend_dev_description(ggml_backend_get_device(backend));
        std::string backend_reg_name =
            ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "support",
                           supported, supported, supported ? "yes" : "no", 0.0, 0.0, 0.0, 0, 0,
                           device_desc, backend_reg_name);

        output_printer->print_test_result(result);

        return true;
    }

    bool eval_grad(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead() * 128 +
                2 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
        gb = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);

        ggml_tensor * out = build_graph(ctx.get());

        if (!matches_filter(out, op_names_filter) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            return true;
        }

        if (out->type != GGML_TYPE_F32) {
            output_printer->print_operation(test_operation_info(
                op_desc(out), vars(), ggml_backend_name(backend), test_status_t::NOT_SUPPORTED,
                out->name + std::string("->type != FP32")));
            return true;
        }

        // Print operation info first
        output_printer->print_operation(
            test_operation_info(op_desc(out), vars(), ggml_backend_name(backend)));

        // check if the backend supports the ops
        bool supported = true;
        bool any_params = false;
        std::string failure_reason;

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL;
             t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                supported = false;
                failure_reason = ggml_backend_name(backend);
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    supported = false;
                    failure_reason = std::string(t->name) + "->type != FP32";
                    break;
                }
            }
        }
        if (!any_params) {
            supported = false;
            failure_reason = op_desc(out);
        }

        if (!supported) {
            output_printer->print_operation(
                test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                    test_status_t::NOT_SUPPORTED, failure_reason));
            return true;
        }

        int64_t ngrads = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL;
             t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += ggml_nelements(t);
            }
        }
        if (ngrads > grad_nmax()) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_large_tensor_skip();
            output_printer->print_operation(info);
            return true;
        }

        if (!ggml_is_scalar(out)) {
            out = ggml_sum(ctx.get(), out);
            ggml_set_name(out, "sum_of_out");
        }
        ggml_set_loss(out);

        ggml_build_forward_expand(gf, out);
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx.get(), gb, nullptr);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL;
                 t = ggml_get_next_tensor(ctx.get(), t)) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) ||
                            ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL;
             t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                output_printer->print_operation(
                    test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                        test_status_t::NOT_SUPPORTED, ggml_backend_name(backend)));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                output_printer->print_operation(test_operation_info(
                    op_desc(out), vars(), ggml_backend_name(backend), test_status_t::NOT_SUPPORTED,
                    std::string(t->name) + "->type != FP32"));
                supported = false;
                break;
            }
        }
        if (!supported) {
            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr
        if (buf == NULL) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_error("allocation", "");
            output_printer->print_operation(info);
            return false;
        }

        initialize_tensors(ctx.get()); // Randomizes all tensors (including gradients).
        ggml_graph_reset(gb);          // Sets gradients to 1 if loss, 0 otherwise.

        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }
        status = ggml_backend_graph_compute(backend, gb);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                    ggml_status_to_string(status));
            return false;
        }

        bool ok = true;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr;
             t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char * bn = ggml_backend_name(backend);
            const int64_t ne = ggml_nelements(t);

            std::vector<float> ga;
            struct ggml_tensor * grad = ggml_graph_get_grad(gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            } else {
                ga.resize(ne); // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) { // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                    info.set_gradient_info(i, bn, ga[i]);
                    output_printer->print_operation(info);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne); // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t); // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu = x0[i] + 1.0f * eps;  // x, index i, up
                const float xiuh = x0[i] + 0.5f * eps; // x, index i, up half
                const float xidh = x0[i] - 0.5f * eps; // x, index i, down half
                const float xid = x0[i] - 1.0f * eps;  // x, index i, down

                float fu, fuh, fdh, fd; // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i * sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                            ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fu, 0, ggml_nbytes(out));

                ggml_backend_tensor_set(t, &xid, i * sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__,
                            ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fd, 0, ggml_nbytes(out));

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i * sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n",
                                __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fuh, 0, ggml_nbytes(out));

                    ggml_backend_tensor_set(t, &xidh, i * sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n",
                                __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fdh, 0, ggml_nbytes(out));

                    gn[i] = (8.0 * (double)fuh + (double)fd - (8.0 * (double)fdh + (double)fu)) /
                            (6.0 * (double)eps);
                } else {
                    gn[i] = (fu - fd) / (2.0f * eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, ggml_nbytes(t));
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                info.set_maa_error(err, max_maa_err());
                output_printer->print_operation(info);
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        // Create final test result
        test_operation_info final_info(op_desc(out), vars(), ggml_backend_name(backend));
        if (!ok) {
            final_info.set_compare_failure();
        }
        final_info.status = ok ? test_status_t::OK : test_status_t::FAIL;
        output_printer->print_operation(final_info);

        if (ok) {
            return true;
        }

        return false;
    }
};

// ###################################
// ## Section 2: GGML Op Defintions ##
// ###################################

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override { return VARS_TO_STR3(type, ne_a, v); }

    test_unary(ggml_unary_op op,
               ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
               int v = 0) :
        op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const bool grad_supported = op == GGML_UNARY_OP_ABS || op == GGML_UNARY_OP_SGN ||
                                    op == GGML_UNARY_OP_NEG || op == GGML_UNARY_OP_STEP ||
                                    op == GGML_UNARY_OP_RELU || op == GGML_UNARY_OP_SILU ||
                                    op == GGML_UNARY_OP_EXPM1 || op == GGML_UNARY_OP_SOFTPLUS;

        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a;
            ne[0] *= 3;
            ne[1] *= 2;
            ne[2] *= 5;
            ne[3] *= 4;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2],
                             a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_unary(ctx, a, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
             t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }

    float grad_eps() override { return 15.0f; }

    std::vector<float> grad_expect() override {
        if (op == GGML_UNARY_OP_ABS) {
            return {-1.0f, 1.0f};
        }
        if (op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_STEP) {
            return {0.0f};
        }
        if (op == GGML_UNARY_OP_RELU) {
            return {0.0f, 1.0f};
        }
        return {};
    }
};

// GGML_OP_ARGMAX
struct test_argmax : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_argmax(ggml_type type = GGML_TYPE_F32, std::array<int64_t, 4> ne = {10, 100, 1, 1}) :
        type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argmax(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
             t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override { return 0.0; }
};

// GGML_OP_ADD
// GGML_OP_SUB
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*)(ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    int nf;     // number of fused ops, nf == 1 -> single op (no fusion)
    bool perm1; // permute src1?

    bool run_whole_graph() override { return nf > 1; }

    std::string vars() override { return VARS_TO_STR5(type, ne, nr, nf, perm1); }

    size_t op_size(ggml_tensor * t) override { return ggml_nbytes(t) * 3; }

    test_bin_bcast(op_t op,
                   ggml_type type = GGML_TYPE_F32,
                   std::array<int64_t, 4> ne = {10, 10, 1, 1},
                   std::array<int, 4> nr = {1, 2, 1, 1},
                   int nf = 1,
                   bool perm1 = false) :
        op(op), type(type), ne(ne), nr(nr), nf(nf), perm1(perm1) {}

    double max_nmse_err(ggml_backend_t backend) override {
        // HSA backend converts F16 to BF16, which has lower precision (7-bit mantissa)
        if ((type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) &&
            backend_has_feature(backend, "SUBSTITUTE_FP16_BF16")) {
            return 1e-4; // BF16 precision limit
        }
        return 1e-7;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(nf <= 16);

        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0] * nr[0], ne[1] * nr[1], ne[2] * nr[2],
                                             ne[3] * nr[3]);
        ggml_set_name(a, "a");

        ggml_tensor * b[16];
        for (int i = 0; i < nf; ++i) {
            if (perm1) {
                const int p[4] = {1, 2, 0, 3}; // hardcoded for now

                b[i] = ggml_new_tensor_4d(ctx, type, ne[p[0]], ne[p[1]], ne[p[2]], ne[p[3]]);
                b[i] = ggml_permute(ctx, b[i], p[0], p[1], p[2], p[3]);
            } else {
                b[i] = ggml_new_tensor(ctx, type, 4, ne.data());
            }
            ggml_set_name(b[i], (std::string("b") + std::to_string(i)).c_str());
        }

        // The backward pass supports broadcasting only for GGML_ADD:
        const bool grad_supported =
            op == ggml_add && ggml_are_same_shape(a, b[0]) && nf == 1 && !perm1;
        if (grad_supported) {
            ggml_set_param(a);
            ggml_set_param(b[0]);
        }

        ggml_tensor * out = a;

        for (int i = 0; i < nf; ++i) {
            out = op(ctx, out, b[i]);
        }

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
             t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_mul || op == ggml_div) {
                // MUL and DIV have numerical issues around zero:
                init_tensor_uniform(t, 0.9f, 1.1f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    float grad_eps() override {
        return 0.1f * (op == ggml_mul ? ne[0] * ne[1] * ne[2] * ne[3] : 1);
    }

    bool grad_precise() override { return op == ggml_div; }

    double max_maa_err() override { return op == ggml_add ? 1e-4 : 1e-3; }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions
    const int64_t k_v; // size of k in memory, resulting in a non-contiguous view for k_v > k, no
                       // view for k_v == 0
    const uint32_t o;  // number of outputs

    std::string vars() override {
        return VARS_TO_STR10(type_a, type_b, m, n, k, bs, nr, per, k_v, o);
    }

    double max_nmse_err() override { return 5e-4; }

    double max_nmse_err(ggml_backend_t backend) override {
        // for blackwell we quantize activations to mxfp4 instead of q8_1 so we add higher tolerance
        if (type_a == GGML_TYPE_MXFP4 && backend_has_feature(backend, "BLACKWELL_NATIVE_FP4")) {
            return 2e-2;
        }
        return max_nmse_err();
    }

    int64_t grad_nmax() override { return 20000; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32,
                 ggml_type type_b = GGML_TYPE_F32,
                 int64_t m = 32,
                 int64_t n = 32,
                 int64_t k = 32,
                 std::array<int64_t, 2> bs = {10, 10},
                 std::array<int64_t, 2> nr = {2, 2},
                 std::array<int64_t, 4> per = {0, 1, 2, 3},
                 int64_t k_v = 0,
                 uint32_t o = 1) :
        type_a(type_a),
        type_b(type_b),
        m(m),
        n(n),
        k(k),
        bs(bs),
        nr(nr),
        per(per),
        k_v(k_v),
        o(o) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(k_v == 0); // not handled
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions
            // given by m,n,k.
            const int64_t ne_a[4] = {k, m, bs[0], bs[1]};
            const int64_t ne_b[4] = {k, n, bs[0] * nr[0], bs[1] * nr[1]};

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]],
                                   ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]],
                                   ne_b[per[3]]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else {
            const int64_t k_physical = k_v == 0 ? k : k_v;
            a = ggml_new_tensor_4d(ctx, type_a, k_physical, m, bs[0], bs[1]);
            b = ggml_new_tensor_4d(ctx, type_b, k_physical, n, bs[0] * nr[0], bs[1] * nr[1]);

            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }

            if (k_v != 0) {
                GGML_ASSERT(k_v > k);
                a = ggml_view_4d(ctx, a, k, m, bs[0], bs[1], a->nb[1], a->nb[2], a->nb[3], 0);
                b = ggml_view_4d(ctx, b, k, n, bs[0] * nr[0], bs[1] * nr[1], b->nb[1], b->nb[2],
                                 b->nb[3], 0);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");
        for (uint32_t i = 1; i < o; ++i) {
            ggml_tensor * out2 = ggml_mul_mat(ctx, a, b);
            ggml_set_name(out2, "out2");
            out = ggml_add(ctx, out, out2);
        }

        return out;
    }

    bool run_whole_graph() override { return o > 1; }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return ggml_op_name(GGML_OP_MUL_MAT);
    }
};

static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->type == GGML_TYPE_I32) {
            if (ggml_is_view_op(t->op)) {
                continue;
            }
            // ids
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                std::vector<int32_t> data(t->ne[0]);
                for (int i = 0; i < t->ne[0]; i++) {
                    data[i] = i % n_mats;
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
            }
        } else {
            init_tensor_uniform(t);
        }
    }
}

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const bool sinks;
    const ggml_type m_prec;
    const std::array<int64_t, 2> nr23; // broadcast only dims 2 and 3
    const float scale;
    const float max_bias;
    const bool inplace;

    std::string vars() override {
        return VARS_TO_STR9(type, ne, mask, sinks, m_prec, nr23, scale, max_bias, inplace);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE
    // = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override { return 1e-6; }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
                  std::array<int64_t, 4> ne = {10, 5, 4, 3},
                  bool mask = false,
                  bool sinks = false,
                  ggml_type m_prec = GGML_TYPE_F32,
                  std::array<int64_t, 2> nr23 = {1, 1},
                  float scale = 1.0f,
                  float max_bias = 0.0f,
                  bool inplace = false) :
        type(type),
        ne(ne),
        mask(mask),
        sinks(sinks),
        m_prec(m_prec),
        nr23(nr23),
        scale(scale),
        max_bias(max_bias),
        inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a =
            ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2] * nr23[0], ne[3] * nr23[1]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_4d(ctx, m_prec, ne[0], ne[1], ne[2], ne[3]);
            ggml_set_name(mask, "mask");
        }

        ggml_tensor * sinks = nullptr;
        if (this->sinks) {
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[2] * nr23[0]);
            ggml_set_name(sinks, "sinks");
        }

        ggml_tensor * out;
        if (inplace) {
            out = ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias);
        } else {
            out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        }
        ggml_soft_max_add_sinks(out, sinks);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override { return true; }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
                ggml_type type_input = GGML_TYPE_F32,
                std::array<int64_t, 4> ne_input =
                    {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
                int k0 = 3,
                int k1 = 3,
                int s0 = 1,
                int s1 = 1,
                int p0 = 1,
                int p1 = 1) :
        pool_type(pool_type),
        type_input(type_input),
        ne_input(ne_input),
        k0(k0),
        k1(k1),
        s0(s0),
        s1(s1),
        p0(p0),
        p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        ggml_set_name(out, "out");

        return out;
    }
};

// CONV_2D
struct test_conv_2d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const ggml_type type_kernel;
    const int stride0;
    const int stride1;
    const int padding0;
    const int padding1;
    const int dilation0;
    const int dilation1;
    // Whether the inputs are contiguous in the channel dim or the width dim
    const bool cwhn;

    // If true, the direct CONV_2D will be used in the graph, otherwise it
    // uses ggml_conv_2d:
    // * if the program is called with -o CONV_2D_DIRECT_IMPL, the
    // CONV_2D graph will be built, while
    // * if the program is called with -o CONV_2D_INDIRECT_IMPL, the
    // IM2COL -> MUL_MM graph will be built.

    std::string vars() override {
        return VARS_TO_STR10(ne_input, ne_kernel, type_kernel, stride0, stride1, padding0, padding1,
                             dilation0, dilation1, cwhn);
    }

    double max_nmse_err() override { return 5e-4; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // KxCRS @ CRSxNPQ = KxNPQ --> KxNPQx(CRS+CRS-1) flops

        // Copied from ggml.c: int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s,
        // int p, int d)
        auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
            return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
        };

        int64_t W = ne_input[0];
        int64_t H = ne_input[1];
        int64_t KW = ne_kernel[0];
        int64_t KH = ne_kernel[1];
        int64_t Cin = ne_kernel[2];
        int64_t Cout = ne_kernel[3];
        int64_t N = ne_input[3];
        int64_t OH = calc_conv_output_size(H, KH, stride0, padding0, dilation0);
        int64_t OW = calc_conv_output_size(W, KW, stride0, padding0, dilation0);

        int64_t K = Cout;
        int64_t CRS = Cin * KH * KW;
        int64_t NPQ = N * OH * OW;

        return K * NPQ * (2 * CRS - 1);
    }

    test_conv_2d(std::array<int64_t, 4> ne_input = {64, 64, 16, 1},
                 std::array<int64_t, 4> ne_kernel = {3, 3, 1, 16},
                 ggml_type type_kernel = GGML_TYPE_F32,
                 int stride0 = 1,
                 int stride1 = 1,
                 int padding0 = 0,
                 int padding1 = 0,
                 int dilation0 = 1,
                 int dilation1 = 1,
                 bool cwhn = false) :
        ne_input(ne_input),
        ne_kernel(ne_kernel),
        type_kernel(type_kernel),
        stride0(stride0),
        stride1(stride1),
        padding0(padding0),
        padding1(padding1),
        dilation0(dilation0),
        dilation1(dilation1),
        cwhn(cwhn) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        if (cwhn) {
            // change memory layout to channel-most-contiguous (CWHN),
            // then permute it back so NE matches the original input
            input = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
            input = ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        ggml_tensor * out = ggml_conv_2d_direct(ctx, kernel, input, stride0, stride1, padding0,
                                                padding1, dilation0, dilation1);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS
struct test_cross_entropy_loss : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_cross_entropy_loss(ggml_type type = GGML_TYPE_F32,
                            std::array<int64_t, 4> ne = {10, 5, 4, 3}) :
        type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(logits);
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        // The labels are assumed to be constant -> no gradients.
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss(ctx, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // For larger abs. diffs between logits softmax is more linear, therefore more precise num.
        // gradients.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
             t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }

    float grad_eps() override { return 1.0f; }

    bool grad_precise() override { return true; }
};

// ## Section 3: GGML Op Test Instantiation ##
// ###########################################

#ifdef _MSC_VER
// Workaround long compile time with msvc
#pragma optimize("", off)
#endif

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep
// the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // MNIST-MLP layer tests (FP32, batch=500)
    // FC1: images [784, 500] x fc1_weight [784, 500] -> [500, 500]
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 500, 500, 784, {1, 1}, {1, 1}));
    // FC1 bias: [500, 500] + fc1_bias [500] -> [500, 500] (broadcast)
    test_cases.emplace_back(
        new test_bin_bcast(ggml_add, GGML_TYPE_F32, {500, 1, 1, 1}, {1, 500, 1, 1}));
    // FC1 ReLU: [500, 500] -> [500, 500]
    test_cases.emplace_back(new test_unary(GGML_UNARY_OP_RELU, GGML_TYPE_F32, {500, 500, 1, 1}));
    // FC2: fc1_out [500, 500] x fc2_weight [500, 10] -> [10, 500]
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 10, 500, 500, {1, 1}, {1, 1}));
    // FC2 bias: [10, 500] + fc2_bias [10] -> [10, 500] (broadcast)
    test_cases.emplace_back(
        new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 1, 1, 1}, {1, 500, 1, 1}));
    // Argmax of logits [10, 500] -> [500]
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {10, 500, 1, 1}));
    // Cross entropy loss on logits [10, 500]
    test_cases.emplace_back(new test_cross_entropy_loss(GGML_TYPE_F32, {10, 500, 1, 1}));
    // Softmax on logits [10, 500]
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {10, 500, 1, 1}));

    // MNIST-CNN layer tests (FP32, batch=500, NCB=8)
    // Conv1: images [28, 28, 1, 500] x conv1_kernel [3, 3, 1, 8], stride=1, pad=1 -> [28, 28, 8,
    // 500]
    test_cases.emplace_back(
        new test_conv_2d({28, 28, 1, 500}, {3, 3, 1, 8}, GGML_TYPE_F32, 1, 1, 1, 1, 1, 1));
    // Conv1 bias + ReLU: [28, 28, 8, 500]
    test_cases.emplace_back(new test_unary(GGML_UNARY_OP_RELU, GGML_TYPE_F32, {28, 28, 8, 500}));
    // MaxPool1: [28, 28, 8, 500] with 2x2 kernel, stride=2, pad=0 -> [14, 14, 8, 500]
    test_cases.emplace_back(
        new test_pool2d(GGML_OP_POOL_MAX, GGML_TYPE_F32, {28, 28, 8, 500}, 2, 2, 2, 2, 0, 0));
    // MaxPool1 with padding: [28, 28, 8, 500] with 3x3 kernel, stride=2, pad=1 -> [14, 14, 8, 500]
    test_cases.emplace_back(
        new test_pool2d(GGML_OP_POOL_MAX, GGML_TYPE_F32, {28, 28, 8, 500}, 3, 3, 2, 2, 1, 1));
    // MaxPool1 F16 input: [28, 28, 8, 500] with 2x2 kernel, stride=2, pad=0 -> [14, 14, 8, 500]
    test_cases.emplace_back(
        new test_pool2d(GGML_OP_POOL_MAX, GGML_TYPE_F16, {28, 28, 8, 500}, 2, 2, 2, 2, 0, 0));
    // Conv2: [14, 14, 8, 500] x conv2_kernel [3, 3, 8, 16], stride=1, pad=1 -> [14, 14, 16, 500]
    test_cases.emplace_back(
        new test_conv_2d({14, 14, 8, 500}, {3, 3, 8, 16}, GGML_TYPE_F32, 1, 1, 1, 1, 1, 1));
    // Conv2 bias + ReLU: [14, 14, 16, 500]
    test_cases.emplace_back(new test_unary(GGML_UNARY_OP_RELU, GGML_TYPE_F32, {14, 14, 16, 500}));
    // MaxPool2: [14, 14, 16, 500] with 2x2 kernel, stride=2, pad=0 -> [7, 7, 16, 500]
    test_cases.emplace_back(
        new test_pool2d(GGML_OP_POOL_MAX, GGML_TYPE_F32, {14, 14, 16, 500}, 2, 2, 2, 2, 0, 0));
    // Dense: flattened [784, 500] x dense_weight [784, 10] -> [10, 500]  (784 = 7*7*16)
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 10, 500, 784, {1, 1}, {1, 1}));
    // Dense bias: [10, 500] + dense_bias [10] -> [10, 500] (broadcast)
    test_cases.emplace_back(
        new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 1, 1, 1}, {1, 500, 1, 1}));
    // Argmax of CNN logits [10, 500] -> [500]
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {10, 500, 1, 1}));

    return test_cases;
}
#ifdef _MSC_VER
#pragma optimize("", on)
#endif

// Test cases for performance evaluation: should be representative of MNIST workloads
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // MNIST-MLP: FC1 [784, 500] x [784, 500]
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 500, 500, 784, {1, 1}, {1, 1}));
    // MNIST-MLP: FC2 [500, 500] x [500, 10]
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 10, 500, 500, {1, 1}, {1, 1}));
    // MNIST-MLP: bias broadcast add [500, 500] + [500]
    test_cases.emplace_back(
        new test_bin_bcast(ggml_add, GGML_TYPE_F32, {500, 1, 1, 1}, {1, 500, 1, 1}));
    // MNIST-MLP: softmax on logits [10, 500]
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {10, 500, 1, 1}, false, false,
                                              GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    // MNIST-MLP: argmax of logits [10, 500]
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {10, 500, 1, 1}));

    // MNIST-CNN: Conv1 [28, 28, 1, 500] x [3, 3, 1, 8], stride=1, pad=1
    test_cases.emplace_back(
        new test_conv_2d({28, 28, 1, 500}, {3, 3, 1, 8}, GGML_TYPE_F32, 1, 1, 1, 1, 1, 1, false));
    // MNIST-CNN: Conv2 [14, 14, 8, 500] x [3, 3, 8, 16], stride=1, pad=1
    test_cases.emplace_back(
        new test_conv_2d({14, 14, 8, 500}, {3, 3, 8, 16}, GGML_TYPE_F32, 1, 1, 1, 1, 1, 1, false));
    // MNIST-CNN: Dense [784, 500] x [784, 10]  (784 = 7*7*16)
    test_cases.emplace_back(
        new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 10, 500, 784, {1, 1}, {1, 1}));

    return test_cases;
}

static bool test_backend(ggml_backend_t backend,
                         test_mode mode,
                         const char * op_names_filter,
                         const char * params_filter,
                         printer * output_printer) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases,
                                const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        if (backend_cpu == NULL) {
            test_operation_info info("", "", "CPU");
            info.set_error("backend", "Failed to initialize CPU backend");
            output_printer->print_operation(info);
            return false;
        }
        // Use reference implementation on the CPU backend for comparison
        using ggml_backend_cpu_set_use_ref_t = void (*)(ggml_backend_t, bool);
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_use_ref = (ggml_backend_cpu_set_use_ref_t)ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_use_ref");
        if (set_use_ref) {
            set_use_ref(backend_cpu, true);
        }

        size_t n_ok = 0;
        size_t tests_run = 0;
        std::vector<std::string> failed_tests;
        for (auto & test : test_cases) {
            test_status_t status =
                test->eval(backend, backend_cpu, op_names_filter, output_printer);
            if (status == test_status_t::SKIPPED || status == test_status_t::NOT_SUPPORTED) {
                continue;
            }
            tests_run++;
            if (status == test_status_t::OK) {
                n_ok++;
            } else if (status == test_status_t::FAIL) {
                failed_tests.push_back(test->current_op_name + "(" + test->vars() + ")");
            }
        }
        output_printer->print_summary(test_summary_info(n_ok, tests_run, false));
        output_printer->print_failed_tests(failed_tests);

        ggml_backend_free(backend_cpu);

        return n_ok == tests_run;
    }

    if (mode == MODE_GRAD) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);
        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval_grad(backend, op_names_filter, output_printer)) {
                n_ok++;
            }
        }
        output_printer->print_summary(test_summary_info(n_ok, test_cases.size(), false));

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        auto test_cases = make_test_cases_perf();
        filter_test_cases(test_cases, params_filter);
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_names_filter, output_printer);
        }
        return true;
    }

    if (mode == MODE_SUPPORT) {
        auto test_cases = make_test_cases_eval();
        filter_test_cases(test_cases, params_filter);

        // Filter out fusion cases
        test_cases.erase(std::remove_if(test_cases.begin(), test_cases.end(),
                                        [](const std::unique_ptr<test_case> & tc) {
                                            return tc->run_whole_graph();
                                        }),
                         test_cases.end());

        for (auto & test : test_cases) {
            test->eval_support(backend, op_names_filter, output_printer);
        }
        return true;
    }

    GGML_ABORT("fatal error");
}

static void list_all_ops() {
    printf("GGML operations:\n");
    std::set<std::string> all_ops;

    for (int i = 1; i < GGML_OP_COUNT; i++) {
        all_ops.insert(ggml_op_name((enum ggml_op)i));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    for (const auto & op : all_ops) {
        printf("  %s\n", op.c_str());
    }
    printf("\nTotal: %zu operations\n", all_ops.size());
}

static void show_test_coverage() {
    std::set<std::string> all_ops;
    for (int i = 1; i < GGML_OP_COUNT; i++) {
        auto op = (enum ggml_op)i;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
            op == GGML_OP_TRANSPOSE || op == GGML_OP_CONT || op == GGML_OP_GLU ||
            op == GGML_OP_UNARY) {
            continue;
        }
        all_ops.insert(ggml_op_name(op));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    auto test_cases = make_test_cases_eval();
    // Filter out fusion cases
    test_cases.erase(
        std::remove_if(test_cases.begin(), test_cases.end(),
                       [](const std::unique_ptr<test_case> & tc) { return tc->run_whole_graph(); }),
        test_cases.end());

    std::set<std::string> tested_ops;

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead() * 128 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };

    for (auto & test_case : test_cases) {
        ggml_context * ctx = ggml_init(params);
        if (ctx) {
            test_case->mode = MODE_TEST;
            ggml_tensor * out = test_case->build_graph(ctx);
            if (out && out->op != GGML_OP_NONE) {
                if (out->op == GGML_OP_UNARY) {
                    tested_ops.insert(ggml_unary_op_name(ggml_get_unary_op(out)));
                } else if (out->op == GGML_OP_GLU) {
                    tested_ops.insert(ggml_glu_op_name(ggml_get_glu_op(out)));
                } else {
                    tested_ops.insert(ggml_op_name(out->op));
                }
            }
            ggml_free(ctx);
        }
    }
    std::set<std::string> covered_ops;
    std::set<std::string> uncovered_ops;
    for (const auto & op : all_ops) {
        if (tested_ops.count(op) > 0) {
            covered_ops.insert(op);
        } else {
            uncovered_ops.insert(op);
        }
    }

    printf("Operations covered by tests (%zu):\n", covered_ops.size());
    for (const auto & op : covered_ops) {
        printf("  ✓ %s\n", op.c_str());
    }
    printf("\nOperations without tests (%zu):\n", uncovered_ops.size());
    for (const auto & op : uncovered_ops) {
        printf("  ✗ %s\n", op.c_str());
    }

    printf("\nCoverage Summary:\n");
    printf("  Total operations: %zu\n", all_ops.size());
    printf("  Tested operations: %zu\n", covered_ops.size());
    printf("  Untested operations: %zu\n", uncovered_ops.size());
    printf("  Coverage: %.1f%%\n", (double)covered_ops.size() / all_ops.size() * 100.0);
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output "
           "<console|sql|csv>] [--list-ops] [--show-coverage]\n",
           argv[0]);
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - grad (compare gradients from backpropagation with method of finite "
           "differences)\n");
    printf("      - perf (performance evaluation)\n");
    printf("      - support (probe backend operation support)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),\n");
    printf("        optionally including the full test case string (e.g. "
           "\"ADD(type=f16,ne=[1,1,8,1],nr=[1,1,1,1],nf=1)\")\n");
    printf("    --output specifies output format (default: console, options: console, sql, csv)\n");
    printf("    --list-ops lists all available GGML operations\n");
    printf("    --show-coverage shows test coverage\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    output_formats output_format = CONSOLE;
    const char * op_names_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "grad") == 0) {
            mode = MODE_GRAD;
        } else if (strcmp(argv[i], "support") == 0) {
            mode = MODE_SUPPORT;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_names_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                if (!output_format_from_str(argv[++i], output_format)) {
                    usage(argv);
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--list-ops") == 0) {
            list_all_ops();
            return 0;
        } else if (strcmp(argv[i], "--show-coverage") == 0) {
            show_test_coverage();
            return 0;
        } else {
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    // Create printer for output format
    std::unique_ptr<printer> output_printer = create_printer(output_format);
    if (output_printer) {
        output_printer->print_header();
    }

    output_printer->print_testing_start(testing_start_info(ggml_backend_dev_count()));

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            output_printer->print_backend_init(backend_init_info(
                i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping"));
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU &&
            mode != MODE_GRAD) {
            output_printer->print_backend_init(backend_init_info(i, ggml_backend_dev_count(),
                                                                 ggml_backend_dev_name(dev), true,
                                                                 "Skipping CPU backend"));
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend, N_THREADS);
        }

        size_t free, total; // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        output_printer->print_backend_init(backend_init_info(
            i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), false, "",
            ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024, true));

        bool ok = test_backend(backend, mode, op_names_filter, params_filter, output_printer.get());

        if (ok) {
            n_ok++;
        }
        output_printer->print_backend_status(backend_status_info(
            ggml_backend_name(backend), ok ? test_status_t::OK : test_status_t::FAIL));

        ggml_backend_free(backend);
    }

    ggml_quantize_free();

    if (output_printer) {
        output_printer->print_footer();
    }

    output_printer->print_overall_summary(
        overall_summary_info(n_ok, ggml_backend_dev_count(), n_ok == ggml_backend_dev_count()));

    if (n_ok != ggml_backend_dev_count()) {
        return 1;
    }

    return 0;
}
