#define _POSIX_C_SOURCE 199309L

#include "core/mlperf_logging.h"

#include <stdio.h>
#include <time.h>
#include <string.h>

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void mlperf_log_event(const char* key, const char* value) {
    if (!key) return;
    double t = get_time_ms();
    if (value)
        printf(":::MLLOG {\"key\": \"%s\", \"value\": \"%s\", \"time_ms\": %.3f}\n",
               key, value, t);
    else
        printf(":::MLLOG {\"key\": \"%s\", \"value\": null, \"time_ms\": %.3f}\n",
               key, t);
    fflush(stdout);
}

void mlperf_log_metric(const char* key, double value) {
    if (!key) return;
    double t = get_time_ms();
    printf(":::MLLOG {\"key\": \"%s\", \"value\": %.6f, \"time_ms\": %.3f}\n",
           key, value, t);
    fflush(stdout);
}

void mlperf_log_start(const char* benchmark) {
    if (!benchmark) return;
    double t = get_time_ms();
    printf(":::MLLOG {\"key\": \"run_start\", \"value\": \"%s\", \"time_ms\": %.3f}\n",
           benchmark, t);
    fflush(stdout);
}

void mlperf_log_end(const char* benchmark, const char* status) {
    if (!benchmark) return;
    double t = get_time_ms();
    printf(":::MLLOG {\"key\": \"run_stop\", \"value\": \"%s\", \"status\": \"%s\", \"time_ms\": %.3f}\n",
           benchmark, status ? status : "success", t);
    fflush(stdout);
}
