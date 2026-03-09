/*
    Copyright (C) 2014 - 2022 Xilinx, Inc. All rights reserved.
    Copyright (C) 2022 - 2025 Advanced Micro Devices, Inc. All rights reserved.
    SPDX-License-Identifier: MIT
*/

/**
 * @file aie_kernel_utils.h
 * @brief Compiler-agnostic macros for AIE kernel loop optimization hints.
 *
 * This header provides portable macros that map to compiler-specific pragmas
 * and attributes for loop optimization on AIE cores. The macros support three
 * compilation environments:
 * - Chess compiler (__chess__): Uses Chess-specific attributes
 * - AIECC compiler (__AIECC__): Uses Clang pragmas
 * - Other compilers: Macros expand to empty (no-op)
 *
 * @defgroup loop_macros Loop Optimization Macros
 * @{
 */

#ifndef _AIE_KERNEL_UTILS_
#define _AIE_KERNEL_UTILS_

#if defined(__chess__)
/**
 * @def AIE_LOOP_UNROLL(x)
 * @brief Unrolls a loop by factor x.
 * @param x The unroll factor (number of iterations to unroll).
 */
#define AIE_LOOP_UNROLL(x) [[chess::unroll_loop(x)]]

/**
 * @def AIE_LOOP_UNROLL_FULL
 * @brief Fully unrolls a loop (all iterations).
 */
#define AIE_LOOP_UNROLL_FULL [[chess::unroll_loop()]]

/**
 * @def AIE_LOOP_NO_UNROLL
 * @brief Prevents the compiler from unrolling a loop.
 */
#define AIE_LOOP_NO_UNROLL [[chess::no_unroll]]

/**
 * @def AIE_LOOP_MIN_ITERATION_COUNT(x)
 * @brief Hints the minimum number of loop iterations to the compiler.
 * @param x The minimum iteration count.
 */
#define AIE_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]

/**
 * @def AIE_LOOP_MAX_ITERATION_COUNT(x)
 * @brief Hints the maximum number of loop iterations to the compiler.
 * @param x The maximum iteration count.
 */
#define AIE_LOOP_MAX_ITERATION_COUNT(x) [[chess::max_loop_count(x)]]

/**
 * @def AIE_LOOP_RANGE(a, ...)
 * @brief Hints both minimum and optionally maximum loop iteration counts.
 * @param a The minimum iteration count.
 * @param ... Optional maximum iteration count.
 */
#define AIE_LOOP_RANGE(a, ...)                                                                     \
    [[chess::min_loop_count(a)]] __VA_OPT__([[chess::max_loop_count(__VA_ARGS__)]])

/**
 * @def AIE_PREPARE_FOR_PIPELINING
 * @brief Prepares a loop for software pipelining optimization.
 */
#define AIE_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]

/**
 * @def AIE_NO_PREPARE_FOR_PIPELINING
 * @brief Prevents software pipelining preparation for a loop.
 */
#define AIE_NO_PREPARE_FOR_PIPELINING [[chess::no_prepare_for_pipelining]]

/**
 * @def AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
 * @brief Sets the modulo scheduling budget ratio for pipelining.
 * @param x The budget ratio value.
 */
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x) [[chess::modulo_scheduling_budget_ratio(x)]]

/**
 * @def AIE_KEEP_SW_LOOP
 * @brief Keeps the loop as a software loop (prevents hardware loop conversion).
 */
#define AIE_KEEP_SW_LOOP [[chess::keep_sw_loop]]

/**
 * @def AIE_PEEL_PIPELINED_LOOP(x)
 * @brief Peels iterations from a pipelined loop.
 * @param x The number of iterations to peel.
 */
#define AIE_PEEL_PIPELINED_LOOP(x) [[chess::peel_pipelined_loop(x)]]

/**
 * @def AIE_KEEP_FREE_FOR_PIPELINING(x)
 * @brief Reserves resources for pipelining.
 * @param x The resource specification.
 */
#define AIE_KEEP_FREE_FOR_PIPELINING(x) [[chess::keep_free_for_pipelining(x)]]

/**
 * @def AIE_ALLOCATE(x)
 * @brief Specifies register allocation hints.
 * @param x The allocation specification.
 */
#define AIE_ALLOCATE(x) [[chess::allocate(x)]]

/**
 * @def AIE_NO_HW_LOOP
 * @brief Prevents conversion to a hardware loop.
 */
#define AIE_NO_HW_LOOP [[chess::no_hw_loop]]

/**
 * @def AIE_TRY_INITIATION_INTERVAL(x)
 * @brief Attempts to achieve a specific initiation interval for pipelining.
 * @param x The target initiation interval.
 * @note No-op on Chess compiler; effective on AIECC.
 */
#define AIE_TRY_INITIATION_INTERVAL(x)

/**
 * @def AIE_PREPARE_FOR_POSTPIPELINING
 * @brief Prepares for post-pipelining optimization.
 * @note No-op on Chess compiler; effective on AIECC.
 */
#define AIE_PREPARE_FOR_POSTPIPELINING

/**
 * @def AIE_LOOP_FLATTEN
 * @brief Flattens nested loops for optimization.
 */
#define AIE_LOOP_FLATTEN chess_flatten_loop

/* AIECC compiler (Clang-based) - uses Clang pragmas */
#elif defined(__AIECC__)
#ifndef __STRINGIFY
#define __STRINGIFY(a) #a
#endif
#define AIE_LOOP_UNROLL(x) _Pragma(__STRINGIFY(clang loop unroll_count(x)))
#define AIE_LOOP_UNROLL_FULL _Pragma("clang loop unroll(full)")
#define AIE_LOOP_NO_UNROLL _Pragma("clang loop unroll(disable)")
#define AIE_LOOP_MIN_ITERATION_COUNT(x) _Pragma(__STRINGIFY(clang loop min_iteration_count(x)))
#define AIE_LOOP_MAX_ITERATION_COUNT(x) _Pragma(__STRINGIFY(clang loop max_iteration_count(x)))
#define AIE_LOOP_RANGE(a, ...)                                                                     \
    AIE_LOOP_MIN_ITERATION_COUNT(a)                                                                \
    __VA_OPT__(AIE_LOOP_MAX_ITERATION_COUNT(__VA_ARGS__))
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)                                                             \
    _Pragma(__STRINGIFY(clang loop pipeline_initiation_interval(x)))
#define AIE_PREPARE_FOR_POSTPIPELINING _Pragma("clang loop pipeline(disable)")
#define AIE_LOOP_FLATTEN

/* Fallback for other compilers - all macros expand to no-ops */
#else
#define AIE_LOOP_UNROLL(x)
#define AIE_LOOP_UNROLL_FULL
#define AIE_LOOP_NO_UNROLL
#define AIE_LOOP_MIN_ITERATION_COUNT(x)
#define AIE_LOOP_MAX_ITERATION_COUNT(x)
#define AIE_LOOP_RANGE(a, ...)
#define AIE_PREPARE_FOR_PIPELINING
#define AIE_NO_PREPARE_FOR_PIPELINING
#define AIE_MODULO_SCHEDULING_BUDGET_RATIO(x)
#define AIE_KEEP_SW_LOOP
#define AIE_PEEL_PIPELINED_LOOP(x)
#define AIE_KEEP_FREE_FOR_PIPELINING(x)
#define AIE_ALLOCATE(x)
#define AIE_NO_HW_LOOP
#define AIE_TRY_INITIATION_INTERVAL(x)
#define AIE_PREPARE_FOR_POSTPIPELINING
#define AIE_LOOP_FLATTEN
#endif

/** @} */ /* End of loop_macros group */

#endif /* _AIE_KERNEL_UTILS_ */