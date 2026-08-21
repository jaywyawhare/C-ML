/* Stress test for the fork/join thread pool: coverage (every item processed
 * exactly once), correctness of the parallel sum-reduction slot math, and
 * repeated batches (catches deadlock/lost-wakeup regressions). An alarm guards
 * against hangs so a deadlock fails the test instead of blocking CI. */
#define _POSIX_C_SOURCE 200809L
#include "backend/threadpool.h"
#include "ops/simd_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>

static _Atomic int* g_hits;      /* per-element processing counter */

static void mark_task(void* data, size_t start, size_t end) {
    (void)data;
    for (size_t i = start; i < end; i++)
        atomic_fetch_add(&g_hits[i], 1);
}

static void on_alarm(int sig) {
    (void)sig;
    fprintf(stderr, "  FAIL: thread pool hung (deadlock)\n");
    _exit(1);
}

int main(void) {
    signal(SIGALRM, on_alarm);
    alarm(60);  /* whole test must finish well within this */

    ThreadPool* pool = threadpool_get_global();
    size_t nt = threadpool_get_num_threads(pool);
    printf("thread pool: %zu workers\n", nt);

    int ok = 1;

    /* 1. Coverage + no double-processing across a range of sizes, including
     *    n < num_threads, n not divisible by num_threads, and n == 0. */
    size_t sizes[] = {0, 1, 3, 7, 100, 1000, 99991, 1u << 20};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        size_t n = sizes[s];
        g_hits = calloc(n ? n : 1, sizeof(_Atomic int));
        for (int rep = 0; rep < 3; rep++) {
            for (size_t i = 0; i < n; i++) atomic_store(&g_hits[i], 0);
            threadpool_parallel_for(pool, mark_task, NULL, n);
            for (size_t i = 0; i < n; i++) {
                if (atomic_load(&g_hits[i]) != 1) {
                    printf("  FAIL n=%zu i=%zu hits=%d (expected 1)\n",
                           n, i, atomic_load(&g_hits[i]));
                    ok = 0; break;
                }
            }
        }
        free(g_hits);
    }
    printf("  coverage/no-double: %s\n", ok ? "PASS" : "FAIL");

    /* 2. Parallel sum reduction correctness vs a serial reference. */
    {
        size_t n = 500000;
        float* a = malloc(n * sizeof(float));
        double ref = 0.0;
        for (size_t i = 0; i < n; i++) { a[i] = (float)((i % 7) - 3) * 0.5f; ref += a[i]; }
        int sum_ok = 1;
        for (int rep = 0; rep < 50; rep++) {
            float got = simd_sum_f32_parallel(a, n);
            if (fabs((double)got - ref) > 1e-1) {  /* fp accumulation slack */
                printf("  FAIL sum: got=%f ref=%f\n", got, ref);
                sum_ok = 0; break;
            }
        }
        printf("  parallel-sum: %s\n", sum_ok ? "PASS" : "FAIL");
        ok = ok && sum_ok;
        free(a);
    }

    /* 3. Rapid repeated batches — stress wakeup/generation logic for deadlock. */
    {
        size_t n = 4096;
        g_hits = calloc(n, sizeof(_Atomic int));
        for (int rep = 0; rep < 5000; rep++) {
            for (size_t i = 0; i < n; i++) atomic_store(&g_hits[i], 0);
            threadpool_parallel_for(pool, mark_task, NULL, n);
        }
        int hammer_ok = 1;
        for (size_t i = 0; i < n; i++)
            if (atomic_load(&g_hits[i]) != 1) { hammer_ok = 0; break; }
        printf("  5000-batch hammer: %s\n", hammer_ok ? "PASS" : "FAIL");
        ok = ok && hammer_ok;
        free(g_hits);
    }

    alarm(0);
    printf("%s\n", ok ? "ALL THREADPOOL TESTS PASS" : "THREADPOOL TESTS FAILED");
    return ok ? 0 : 1;
}
