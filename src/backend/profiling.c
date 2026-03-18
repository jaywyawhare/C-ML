#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "backend/profiling.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

static double timespec_to_ms(struct timespec* ts) {
    return (double)ts->tv_sec * 1000.0 + (double)ts->tv_nsec / 1000000.0;
}

Timer* profiler_timer_create(const char* name) {
    Timer* timer = malloc(sizeof(Timer));
    if (!timer)
        return NULL;

    timer->start_time.tv_sec  = 0;
    timer->start_time.tv_nsec = 0;
    timer->end_time.tv_sec    = 0;
    timer->end_time.tv_nsec   = 0;
    timer->elapsed_ms         = 0.0;
    timer->is_running         = false;
    if (name) {
        size_t len  = strlen(name) + 1;
        timer->name = malloc(len);
        if (timer->name) {
            memcpy(timer->name, name, len);
        }
    } else {
        timer->name = NULL;
    }

    return timer;
}

void profiler_timer_free(Timer* timer) {
    if (!timer)
        return;

    if (timer->name) {
        free(timer->name);
    }
    free(timer);
}

int profiler_timer_start(Timer* timer) {
    if (!timer)
        return -1;

    if (clock_gettime(CLOCK_MONOTONIC, &timer->start_time) != 0) {
        LOG_ERROR("Failed to get start time");
        return -1;
    }

    timer->is_running = true;
    timer->elapsed_ms = 0.0;
    return 0;
}

double profiler_timer_stop(Timer* timer) {
    if (!timer || !timer->is_running)
        return -1.0;

    if (clock_gettime(CLOCK_MONOTONIC, &timer->end_time) != 0) {
        LOG_ERROR("Failed to get end time");
        return -1.0;
    }

    double start_ms   = timespec_to_ms(&timer->start_time);
    double end_ms     = timespec_to_ms(&timer->end_time);
    timer->elapsed_ms = end_ms - start_ms;
    timer->is_running = false;

    return timer->elapsed_ms;
}

double profiler_timer_elapsed(Timer* timer) {
    if (!timer)
        return -1.0;

    if (!timer->is_running) {
        return timer->elapsed_ms;
    }

    struct timespec current_time;
    if (clock_gettime(CLOCK_MONOTONIC, &current_time) != 0) {
        return -1.0;
    }

    double start_ms   = timespec_to_ms(&timer->start_time);
    double current_ms = timespec_to_ms(&current_time);
    return current_ms - start_ms;
}

void profiler_timer_reset(Timer* timer) {
    if (!timer)
        return;

    timer->start_time.tv_sec  = 0;
    timer->start_time.tv_nsec = 0;
    timer->end_time.tv_sec    = 0;
    timer->end_time.tv_nsec   = 0;
    timer->elapsed_ms         = 0.0;
    timer->is_running         = false;
}

Profiler* profiler_create(void) {
    Profiler* profiler = malloc(sizeof(Profiler));
    if (!profiler)
        return NULL;

    profiler->timers     = NULL;
    profiler->num_timers = 0;
    profiler->capacity   = 0;
    profiler->enabled    = true;

    return profiler;
}

void profiler_free(Profiler* profiler) {
    if (!profiler)
        return;

    if (profiler->timers) {
        for (int i = 0; i < profiler->num_timers; i++) {
            if (profiler->timers[i]) {
                profiler_timer_free(profiler->timers[i]);
            }
        }
        free(profiler->timers);
    }

    free(profiler);
}

void profiler_set_enabled(Profiler* profiler, bool enabled) {
    if (!profiler)
        return;
    profiler->enabled = enabled;
}

int profiler_start(Profiler* profiler, const char* name) {
    if (!profiler || !name || !profiler->enabled)
        return -1;

    // Resize array if needed
    if (profiler->num_timers >= profiler->capacity) {
        int new_capacity   = profiler->capacity == 0 ? 8 : profiler->capacity * 2;
        Timer** new_timers = realloc(profiler->timers, (size_t)new_capacity * sizeof(Timer*));
        if (!new_timers)
            return -1;

        profiler->timers   = new_timers;
        profiler->capacity = new_capacity;
    }

    // Create new timer
    Timer* timer = profiler_timer_create(name);
    if (!timer)
        return -1;

    if (profiler_timer_start(timer) != 0) {
        profiler_timer_free(timer);
        return -1;
    }

    profiler->timers[profiler->num_timers] = timer;
    int timer_id                           = profiler->num_timers;
    profiler->num_timers++;

    return timer_id;
}

double profiler_stop(Profiler* profiler, int timer_id) {
    if (!profiler || timer_id < 0 || timer_id >= profiler->num_timers) {
        return -1.0;
    }

    Timer* timer = profiler->timers[timer_id];
    if (!timer)
        return -1.0;

    return profiler_timer_stop(timer);
}

void profiler_print_report(Profiler* profiler) {
    if (!profiler)
        return;

    printf("\nProfiling Report\n");
    printf("%-30s %15s\n", "Operation", "Time (ms)");

    double total_time = 0.0;
    for (int i = 0; i < profiler->num_timers; i++) {
        Timer* timer = profiler->timers[i];
        if (timer) {
            double elapsed = timer->is_running ? profiler_timer_elapsed(timer) : timer->elapsed_ms;
            printf("%-30s %15.3f\n", timer->name ? timer->name : "Unknown", elapsed);
            total_time += elapsed;
        }
    }

    printf("%-30s %15.3f\n", "Total", total_time);
    printf("\n");
}

double profiler_get_total_time(Profiler* profiler, const char* name) {
    if (!profiler || !name)
        return -1.0;

    double total = 0.0;
    for (int i = 0; i < profiler->num_timers; i++) {
        Timer* timer = profiler->timers[i];
        if (timer && timer->name && strcmp(timer->name, name) == 0) {
            double elapsed = timer->is_running ? profiler_timer_elapsed(timer) : timer->elapsed_ms;
            total += elapsed;
        }
    }

    return total;
}
