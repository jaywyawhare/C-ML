/**
 * @file profiling.h
 * @brief Performance profiling tools
 */

#ifndef CML_CORE_PROFILING_H
#define CML_CORE_PROFILING_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Timer structure
typedef struct Timer {
    struct timespec start_time;
    struct timespec end_time;
    double elapsed_ms;
    bool is_running;
    char* name; // Owned by this struct
} Timer;

// Profiler structure
typedef struct Profiler {
    Timer** timers;
    int num_timers;
    int capacity;
    bool enabled;
} Profiler;

// Timer Functions

/**
 * @brief Create a new timer
 *
 * @param name Timer name
 * @return New timer, or NULL on failure
 */
Timer* profiler_timer_create(const char* name);

/**
 * @brief Free timer
 *
 * @param timer Timer to free
 */
void profiler_timer_free(Timer* timer);

/**
 * @brief Start timer
 *
 * @param timer Timer to start
 * @return 0 on success, negative on failure
 */
int profiler_timer_start(Timer* timer);

/**
 * @brief Stop timer
 *
 * @param timer Timer to stop
 * @return Elapsed time in milliseconds, or negative on failure
 */
double profiler_timer_stop(Timer* timer);

/**
 * @brief Get elapsed time without stopping
 *
 * @param timer Timer
 * @return Elapsed time in milliseconds, or negative on failure
 */
double profiler_timer_elapsed(Timer* timer);

/**
 * @brief Reset timer
 *
 * @param timer Timer to reset
 */
void profiler_timer_reset(Timer* timer);

// Profiler Functions

/**
 * @brief Create profiler
 *
 * @return New profiler, or NULL on failure
 */
Profiler* profiler_create(void);

/**
 * @brief Free profiler
 *
 * @param profiler Profiler to free
 */
void profiler_free(Profiler* profiler);

/**
 * @brief Enable/disable profiler
 *
 * @param profiler Profiler
 * @param enabled Enable flag
 */
void profiler_set_enabled(Profiler* profiler, bool enabled);

/**
 * @brief Start timing an operation
 *
 * @param profiler Profiler
 * @param name Operation name
 * @return Timer ID, or negative on failure
 */
int profiler_start(Profiler* profiler, const char* name);

/**
 * @brief Stop timing an operation
 *
 * @param profiler Profiler
 * @param timer_id Timer ID
 * @return Elapsed time in milliseconds, or negative on failure
 */
double profiler_stop(Profiler* profiler, int timer_id);

/**
 * @brief Print profiling report
 *
 * @param profiler Profiler
 */
void profiler_print_report(Profiler* profiler);

/**
 * @brief Get total time for an operation
 *
 * @param profiler Profiler
 * @param name Operation name
 * @return Total time in milliseconds, or negative on failure
 */
double profiler_get_total_time(Profiler* profiler, const char* name);

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_PROFILING_H
