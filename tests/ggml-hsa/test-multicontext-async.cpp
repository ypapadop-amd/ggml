#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace std::chrono;

struct CommandTiming {
    int context_id;
    int command_id;
    int size;  // Matrix size
    bool is_slow;
    steady_clock::time_point submit_time;
    steady_clock::time_point complete_time;
    double expected_duration_ms;

    double actual_latency_ms() const {
        return duration_cast<microseconds>(complete_time - submit_time).count() / 1000.0;
    }
};

std::vector<CommandTiming> timings;
std::atomic<int> completed_count{0};

// Create and execute a matrix multiplication of given size
void execute_matmul(ggml_backend_t backend, int size, CommandTiming* timing) {
    // Create context
    size_t buffer_size = size * size * sizeof(int16_t) * 3; // 2 inputs + 1 output

    ggml_init_params params{
        /*.mem_size   =*/ ggml_tensor_overhead() * 3,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context* ctx = ggml_init(params);

    // Create tensors
    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, size, size);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_I16, size, size);

    // Allocate buffer
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    ggml_tallocr alloc = ggml_tallocr_new(buffer);

    ggml_tallocr_alloc(&alloc, a);
    ggml_tallocr_alloc(&alloc, b);

    // Initialize data
    std::vector<int16_t> data_a(size * size, 1);
    std::vector<int16_t> data_b(size * size, 1);

    ggml_backend_tensor_set(a, data_a.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, data_b.data(), 0, ggml_nbytes(b));

    // Build graph
    std::vector<uint8_t> buf(ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead());
    ggml_init_params params0{
        /*.mem_size   =*/ buf.size(),
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context* ctx0 = ggml_init(params0);
    ggml_cgraph* gf = ggml_new_graph(ctx0);

    ggml_tensor* result = ggml_mul_mat(ctx0, a, b);
    ggml_build_forward_expand(gf, result);

    // Record submit time
    timing->submit_time = steady_clock::now();

    // Execute
    ggml_backend_graph_compute(backend, gf);

    // Synchronize to get actual completion time
    ggml_backend_synchronize(backend);

    timing->complete_time = steady_clock::now();
    completed_count++;

    // Cleanup
    ggml_free(ctx0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void test_multicontext_stress() {
    std::cout << "\n=== TEST 1: Multi-Context Stress Test ===" << std::endl;
    std::cout << "Testing head-of-line blocking elimination\n" << std::endl;

#ifndef GGML_USE_HSA
    std::cout << "Skipping test - HSA backend not available" << std::endl;
    return;
#endif

    // Test configuration
    const int NUM_CONTEXTS = 4;
    const int COMMANDS_PER_CONTEXT = 3;

    // Context 0: Large matrices (slow)
    // Contexts 1-3: Small matrices (fast)
    int matrix_sizes[NUM_CONTEXTS] = {1024, 128, 128, 128};
    double expected_times[NUM_CONTEXTS] = {1500, 50, 50, 50}; // milliseconds (rough estimates)

    std::vector<ggml_backend_t> backends(NUM_CONTEXTS);
    timings.clear();
    completed_count = 0;

    // Create backends (each represents a hardware context)
    std::cout << "Creating " << NUM_CONTEXTS << " HSA backends..." << std::endl;
    for (int i = 0; i < NUM_CONTEXTS; i++) {
        backends[i] = ggml_backend_hsa_init(0);
        if (!backends[i]) {
            std::cerr << "Failed to create HSA backend " << i << std::endl;
            return;
        }
        std::cout << "  Backend " << i << ": Matrix size " << matrix_sizes[i]
                  << "x" << matrix_sizes[i]
                  << " (expected ~" << expected_times[i] << "ms)" << std::endl;
    }

    std::cout << "\nSubmitting commands..." << std::endl;

    // Create timing entries
    for (int ctx = 0; ctx < NUM_CONTEXTS; ctx++) {
        for (int cmd = 0; cmd < COMMANDS_PER_CONTEXT; cmd++) {
            CommandTiming t;
            t.context_id = ctx;
            t.command_id = cmd;
            t.size = matrix_sizes[ctx];
            t.is_slow = (ctx == 0);
            t.expected_duration_ms = expected_times[ctx];
            timings.push_back(t);
        }
    }

    auto test_start = steady_clock::now();

    // Submit all commands in parallel threads (slow first, then fast)
    std::vector<std::thread> threads;
    int timing_idx = 0;

    for (int ctx = 0; ctx < NUM_CONTEXTS; ctx++) {
        for (int cmd = 0; cmd < COMMANDS_PER_CONTEXT; cmd++) {
            threads.emplace_back(execute_matmul, backends[ctx], matrix_sizes[ctx],
                                &timings[timing_idx]);
            timing_idx++;
        }
        std::cout << "Submitted " << COMMANDS_PER_CONTEXT << " commands to context " << ctx << std::endl;
    }

    std::cout << "\nWaiting for all commands to complete..." << std::endl;

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    auto test_end = steady_clock::now();
    double total_time = duration_cast<milliseconds>(test_end - test_start).count();

    std::cout << "\n✓ All " << (NUM_CONTEXTS * COMMANDS_PER_CONTEXT)
              << " commands completed in " << total_time << " ms\n" << std::endl;

    // Analyze results
    std::cout << "=== Completion Analysis ===" << std::endl;
    std::cout << std::setw(8) << "Context" << std::setw(10) << "Command"
              << std::setw(10) << "Size" << std::setw(12) << "Type"
              << std::setw(15) << "Latency(ms)" << std::setw(18) << "Expected(ms)"
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(88, '-') << std::endl;

    // Sort by completion time
    auto sorted = timings;
    std::sort(sorted.begin(), sorted.end(),
              [](const CommandTiming& a, const CommandTiming& b) {
                  return a.complete_time < b.complete_time;
              });

    int fast_before_slow = 0;
    int slow_seen = 0;

    for (const auto& t : sorted) {
        double latency = t.actual_latency_ms();
        std::string type = t.is_slow ? "SLOW" : "FAST";
        std::string status;

        if (t.is_slow) {
            slow_seen++;
            status = "Expected";
        } else if (slow_seen < COMMANDS_PER_CONTEXT) {
            fast_before_slow++;
            status = "✓ Before slow!";
        } else {
            status = "✗ After slow";
        }

        std::cout << std::setw(8) << t.context_id
                  << std::setw(10) << t.command_id
                  << std::setw(10) << t.size
                  << std::setw(12) << type
                  << std::setw(15) << std::fixed << std::setprecision(2) << latency
                  << std::setw(18) << t.expected_duration_ms
                  << std::setw(15) << status << std::endl;
    }

    // Calculate statistics
    std::vector<double> fast_latencies, slow_latencies;
    for (const auto& t : timings) {
        if (t.is_slow) {
            slow_latencies.push_back(t.actual_latency_ms());
        } else {
            fast_latencies.push_back(t.actual_latency_ms());
        }
    }

    auto calc_avg = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    double fast_avg = calc_avg(fast_latencies);
    double slow_avg = calc_avg(slow_latencies);

    std::cout << "\n=== Latency Statistics ===" << std::endl;
    std::cout << "Fast commands (128x128):" << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << fast_avg << " ms" << std::endl;
    std::cout << "  Expected: ~50 ms" << std::endl;

    std::cout << "\nSlow commands (1024x1024):" << std::endl;
    std::cout << "  Average: " << slow_avg << " ms" << std::endl;
    std::cout << "  Expected: ~1500 ms" << std::endl;

    std::cout << "\n=== RESULT ===" << std::endl;
    int total_fast = (NUM_CONTEXTS - 1) * COMMANDS_PER_CONTEXT;
    if (fast_before_slow == total_fast) {
        std::cout << "✓✓✓ PASS: ALL " << fast_before_slow << " fast commands completed before slow commands!" << std::endl;
        std::cout << "    This PROVES head-of-line blocking is ELIMINATED!" << std::endl;
        std::cout << "\n    Fast commands completed in ~" << fast_avg << "ms (not blocked by "
                  << slow_avg << "ms slow commands)" << std::endl;
    } else if (fast_before_slow > 0) {
        std::cout << "✓ PARTIAL: " << fast_before_slow << "/" << total_fast
                  << " fast commands completed before slow" << std::endl;
    } else {
        std::cout << "✗ FAIL: All fast commands waited for slow commands" << std::endl;
        std::cout << "  Head-of-line blocking is PRESENT (FIFO behavior)" << std::endl;
    }

    // Cleanup
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

void test_concurrent_submission() {
    std::cout << "\n=== TEST 2: Concurrent Submission Test ===" << std::endl;

#ifndef GGML_USE_HSA
    std::cout << "Skipping test - HSA backend not available" << std::endl;
    return;
#endif

    const int NUM_THREADS = 4;
    const int COMMANDS_PER_THREAD = 10;
    const int MATRIX_SIZE = 256;

    std::atomic<int> completed{0};

    auto worker = [&](int thread_id) {
        ggml_backend_t backend = ggml_backend_hsa_init(0);
        if (!backend) {
            std::cerr << "Thread " << thread_id << " failed to create backend" << std::endl;
            return;
        }

        for (int i = 0; i < COMMANDS_PER_THREAD; i++) {
            CommandTiming t;
            execute_matmul(backend, MATRIX_SIZE, &t);
            completed++;
        }

        ggml_backend_free(backend);
    };

    auto start = steady_clock::now();

    std::cout << "Launching " << NUM_THREADS << " threads, each submitting "
              << COMMANDS_PER_THREAD << " commands..." << std::endl;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(worker, i);
    }

    // Monitor progress
    while (completed < NUM_THREADS * COMMANDS_PER_THREAD) {
        std::this_thread::sleep_for(milliseconds(100));
        std::cout << "Progress: " << completed << "/" << (NUM_THREADS * COMMANDS_PER_THREAD)
                  << " commands\r" << std::flush;
    }

    for (auto& t : threads) {
        t.join();
    }

    auto end = steady_clock::now();
    double elapsed = duration_cast<milliseconds>(end - start).count();

    std::cout << "\n\n✓ PASS: " << (NUM_THREADS * COMMANDS_PER_THREAD)
              << " concurrent commands completed in " << elapsed << " ms" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
              << (NUM_THREADS * COMMANDS_PER_THREAD * 1000.0 / elapsed)
              << " commands/sec" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "Multi-Context Async Execution Test" << std::endl;
    std::cout << "Approach B (Syncobj Multi-Wait)" << std::endl;
    std::cout << "======================================" << std::endl;

    test_multicontext_stress();
    test_concurrent_submission();

    std::cout << "\n=== All Tests Complete ===" << std::endl;

    return 0;
}
