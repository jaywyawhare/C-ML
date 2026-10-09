/**
 * Timing and statistics primitives shared by every benchmark and profiling
 * harness in this directory.
 *
 * Deliberately free of project headers: bench_torch_c.c measures the torch_c
 * API and bench_cross_framework.c measures CML, so anything they share must not
 * drag one's headers into the other's translation unit.
 */
#ifndef CML_BENCH_TIMING_H
#define CML_BENCH_TIMING_H

#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CLOCK_MONOTONIC, not wall time: these harnesses time sub-millisecond regions
 * and must not be perturbed by clock adjustment. */
static inline double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline void fill_random(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX - 0.5f;
}

/* Let boost clocks settle between benchmarks, so an earlier heavy case does not
 * charge its thermal cost to the next one. */
static inline void cooldown_ms(int ms) {
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static inline int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* Sorts `arr` in place. */
static inline double median(double* arr, int n) {
    qsort(arr, (size_t)n, sizeof(double), cmp_double);
    return (n % 2) ? arr[n / 2] : (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
}

/* Distribution of per-trial timings, all in seconds. */
typedef struct {
    double min, median, p95, mean, stddev;
    int n;
} BenchStats;

/* Summarise `n` per-trial samples (seconds); sorts `samples` in place. For
 * micro-optimisation A/B comparisons `min` is the most stable signal (least
 * perturbed by scheduler/thermal noise) and `median` resists outliers, which a
 * plain mean-over-iterations does not -- a single slow trial skews the mean and
 * hides whether a change is real. */
static inline BenchStats bench_stats(double* samples, int n) {
    BenchStats s = {0};
    if (n <= 0)
        return s;
    s.n        = n;
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += samples[i];
    s.mean     = sum / n;
    double var = 0;
    for (int i = 0; i < n; i++) {
        double d = samples[i] - s.mean;
        var += d * d;
    }
    s.stddev = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    s.min    = samples[0];
    s.median = (n % 2) ? samples[n / 2] : (samples[n / 2 - 1] + samples[n / 2]) / 2.0;
    s.p95    = samples[(int)(0.95 * (n - 1) + 0.5)];
    return s;
}

/* Time `body` (a statement) `trials` times, `inner` reps per trial, into the
 * samples array `buf` (must hold >= `trials` doubles), and return the
 * per-rep BenchStats in seconds. A warmup rep runs first and is not recorded. */
#define BENCH_COLLECT(buf, trials, inner, body)                                                    \
    do {                                                                                           \
        {                                                                                          \
            body;                                                                                  \
        } /* warmup */                                                                             \
        for (int _t = 0; _t < (trials); _t++) {                                                    \
            double _t0 = now();                                                                    \
            for (int _r = 0; _r < (inner); _r++) {                                                 \
                body;                                                                              \
            }                                                                                      \
            (buf)[_t] = (now() - _t0) / (inner);                                                   \
        }                                                                                          \
    } while (0)

/* Print a labelled stats line in ms: min / median / p95 / stddev and n. */
static inline void bench_stats_print(const char* label, BenchStats s) {
    printf("  %-24s min=%7.3f  med=%7.3f  p95=%7.3f  sd=%6.3f ms  (n=%d)\n", label, s.min * 1e3,
           s.median * 1e3, s.p95 * 1e3, s.stddev * 1e3, s.n);
}

/* Emit the results object the Python drivers parse.
 *
 * The drivers place these binaries in adjacent columns and read them key by
 * key, so the schema is a contract between them: same names, same order, same
 * precision. Emitting it from one place is what keeps that true. */
static inline void bench_print_json(double gemm_512, double gemm_1024, double gemm_2048,
                                    double fused_512, double fused_1024, double fused_2048,
                                    double mlp_fwd, double mlp_train, double conv2d_fwd) {
    printf("{\n");
    printf("  \"gemm_512\": %.3f,\n", gemm_512);
    printf("  \"gemm_1024\": %.3f,\n", gemm_1024);
    printf("  \"gemm_2048\": %.3f,\n", gemm_2048);
    printf("  \"fused_512\": %.3f,\n", fused_512);
    printf("  \"fused_1024\": %.3f,\n", fused_1024);
    printf("  \"fused_2048\": %.3f,\n", fused_2048);
    printf("  \"mlp_forward\": %.3f,\n", mlp_fwd);
    printf("  \"mlp_train_step\": %.3f,\n", mlp_train);
    printf("  \"conv2d_forward\": %.3f\n", conv2d_fwd);
    printf("}\n");
}

#endif /* CML_BENCH_TIMING_H */
