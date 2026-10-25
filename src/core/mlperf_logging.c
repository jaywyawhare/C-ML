#define _POSIX_C_SOURCE 199309L

#include "core/mlperf_logging.h"
#include "backend/profiling.h"

#include <stdio.h>
#include <time.h>
#include <string.h>

/**
 * Emit a timestamped MLPerf :::MLLOG line with a string value (or null).
 * @param key (const char*)
 * @param value (const char*)
 * @return (void)
 */
void mlperf_log_event(const char* key, const char* value) {
    if (!key)
        return;
    double t = cml_monotonic_ms();
    if (value)
        printf(":::MLLOG {\"key\": \"%s\", \"value\": \"%s\", \"time_ms\": %.3f}\n", key, value, t);
    else
        printf(":::MLLOG {\"key\": \"%s\", \"value\": null, \"time_ms\": %.3f}\n", key, t);
    fflush(stdout);
}

/**
 * Emit a timestamped MLPerf :::MLLOG line with a numeric value.
 * @param key (const char*)
 * @param value (double)
 * @return (void)
 */
void mlperf_log_metric(const char* key, double value) {
    if (!key)
        return;
    double t = cml_monotonic_ms();
    printf(":::MLLOG {\"key\": \"%s\", \"value\": %.6f, \"time_ms\": %.3f}\n", key, value, t);
    fflush(stdout);
}

/**
 * Emit the MLPerf run_start marker for a benchmark.
 * @param benchmark (const char*)
 * @return (void)
 */
void mlperf_log_start(const char* benchmark) {
    if (!benchmark)
        return;
    double t = cml_monotonic_ms();
    printf(":::MLLOG {\"key\": \"run_start\", \"value\": \"%s\", \"time_ms\": %.3f}\n", benchmark,
           t);
    fflush(stdout);
}

/**
 * Emit the MLPerf run_stop marker; status defaults to "success" when NULL.
 * @param benchmark (const char*)
 * @param status (const char*)
 * @return (void)
 */
void mlperf_log_end(const char* benchmark, const char* status) {
    if (!benchmark)
        return;
    double t = cml_monotonic_ms();
    printf(":::MLLOG {\"key\": \"run_stop\", \"value\": \"%s\", \"status\": \"%s\", \"time_ms\": "
           "%.3f}\n",
           benchmark, status ? status : "success", t);
    fflush(stdout);
}
