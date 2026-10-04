#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "backend/profiling.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "alloc/cml_allocator.h"

/** Convert a timespec to milliseconds. */
static double timespec_to_ms(struct timespec* ts) {
    return (double)ts->tv_sec * 1000.0 + (double)ts->tv_nsec / 1000000.0;
}

/** Allocate a zeroed timer, copying the optional name. */
Timer* profiler_timer_create(const char* name) {
    Timer* timer = cml_malloc(sizeof(Timer));
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
        timer->name = cml_malloc(len);
        if (timer->name) {
            memcpy(timer->name, name, len);
        }
    } else {
        timer->name = NULL;
    }

    return timer;
}

/** Free a timer and its name string. */
void profiler_timer_free(Timer* timer) {
    if (!timer)
        return;

    if (timer->name) {
        cml_free(timer->name);
    }
    cml_free(timer);
}

/** Start (or restart) a timer from the monotonic clock. */
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

/** Stop a running timer and return the elapsed milliseconds. */
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

/** Elapsed milliseconds: live time if still running, else the recorded total. */
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

/** Clear a timer's times and running state. */
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

/** Allocate an empty, enabled profiler with no timers. */
Profiler* profiler_create(void) {
    Profiler* profiler = cml_malloc(sizeof(Profiler));
    if (!profiler)
        return NULL;

    profiler->timers     = NULL;
    profiler->num_timers = 0;
    profiler->capacity   = 0;
    profiler->enabled    = true;

    profiler->stats          = NULL;
    profiler->num_stats      = 0;
    profiler->stats_capacity = 0;
    profiler->stack_depth    = 0;

    return profiler;
}

/** Free a profiler and all timers it owns. */
void profiler_free(Profiler* profiler) {
    if (!profiler)
        return;

    if (profiler->timers) {
        for (int i = 0; i < profiler->num_timers; i++) {
            if (profiler->timers[i]) {
                profiler_timer_free(profiler->timers[i]);
            }
        }
        cml_free(profiler->timers);
    }

    if (profiler->stats) {
        for (int i = 0; i < profiler->num_stats; i++)
            cml_free(profiler->stats[i].name);
        cml_free(profiler->stats);
    }

    cml_free(profiler);
}

/** Enable or disable timing collection on the profiler. */
void profiler_set_enabled(Profiler* profiler, bool enabled) {
    if (!profiler)
        return;
    profiler->enabled = enabled;
}

/** Start a named timer, growing the timer array as needed; returns its id. */
int profiler_start(Profiler* profiler, const char* name) {
    if (!profiler || !name || !profiler->enabled)
        return -1;

    // Resize array if needed
    if (profiler->num_timers >= profiler->capacity) {
        int new_capacity   = profiler->capacity == 0 ? 8 : profiler->capacity * 2;
        Timer** new_timers = cml_realloc(profiler->timers, (size_t)new_capacity * sizeof(Timer*));
        if (!new_timers)
            return -1;

        profiler->timers   = new_timers;
        profiler->capacity = new_capacity;
    }

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

/** Stop the timer with the given id and return its elapsed milliseconds. */
double profiler_stop(Profiler* profiler, int timer_id) {
    if (!profiler || timer_id < 0 || timer_id >= profiler->num_timers) {
        return -1.0;
    }

    Timer* timer = profiler->timers[timer_id];
    if (!timer)
        return -1.0;

    return profiler_timer_stop(timer);
}

/** Print the aggregated scope table (when scopes were used) and/or the flat
 *  per-timer table. The scope table is sorted by self (exclusive) time, since
 *  that is what points at the real hot spot. */
void profiler_print_report(Profiler* profiler) {
    if (!profiler)
        return;

    if (profiler->num_stats > 0) {
        double sum_self = 0.0;
        for (int i = 0; i < profiler->num_stats; i++)
            sum_self += profiler->stats[i].self_ms;

        /* Index array sorted by self_ms, descending (small N: insertion sort). */
        int order[profiler->num_stats];
        for (int i = 0; i < profiler->num_stats; i++)
            order[i] = i;
        for (int i = 1; i < profiler->num_stats; i++) {
            int key = order[i], j = i - 1;
            while (j >= 0 && profiler->stats[order[j]].self_ms < profiler->stats[key].self_ms) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }

        printf("\nProfiling Report (scopes, by self time)\n");
        printf("%-28s %7s %11s %11s %10s %10s %7s %8s %10s\n", "Scope", "calls", "total(ms)",
               "self(ms)", "avg(ms)", "max(ms)", "self%", "allocs", "net(KB)");
        for (int k = 0; k < profiler->num_stats; k++) {
            const ProfileStat* s = &profiler->stats[order[k]];
            if (s->count == 0)
                continue;
            double avg = s->total_ms / (double)s->count;
            double pct = sum_self > 0.0 ? 100.0 * s->self_ms / sum_self : 0.0;
            printf("%-28s %7ld %11.3f %11.3f %10.4f %10.4f %6.1f%% %8ld %10.1f\n",
                   s->name ? s->name : "?", s->count, s->total_ms, s->self_ms, avg, s->max_ms, pct,
                   s->alloc_count, (double)s->net_bytes / 1024.0);
        }
        printf("%-28s %7s %11.3f\n", "Total self", "", sum_self);
        printf("\n");
    }

    if (profiler->num_timers > 0) {
        printf("\nProfiling Report (timers)\n");
        printf("%-30s %15s\n", "Operation", "Time (ms)");
        double total_time = 0.0;
        for (int i = 0; i < profiler->num_timers; i++) {
            Timer* timer = profiler->timers[i];
            if (timer) {
                double elapsed =
                    timer->is_running ? profiler_timer_elapsed(timer) : timer->elapsed_ms;
                printf("%-30s %15.3f\n", timer->name ? timer->name : "Unknown", elapsed);
                total_time += elapsed;
            }
        }
        printf("%-30s %15.3f\n", "Total", total_time);
        printf("\n");
    }
}

/** Sum the elapsed time of every timer matching the given name. */
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

/* ---------------- Nestable aggregated scopes ---------------- */

/** Monotonic clock in milliseconds. */
static double now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return timespec_to_ms(&ts);
}

/** Index of the stat named `name`, creating a zeroed entry if absent; -1 on OOM.
 *  Linear scan: the scope-name set is small and this is off the timed path. */
static int profiler_stat_index(Profiler* profiler, const char* name) {
    for (int i = 0; i < profiler->num_stats; i++)
        if (profiler->stats[i].name && strcmp(profiler->stats[i].name, name) == 0)
            return i;

    if (profiler->num_stats >= profiler->stats_capacity) {
        int new_cap        = profiler->stats_capacity == 0 ? 16 : profiler->stats_capacity * 2;
        ProfileStat* grown = cml_realloc(profiler->stats, (size_t)new_cap * sizeof(ProfileStat));
        if (!grown)
            return -1;
        profiler->stats          = grown;
        profiler->stats_capacity = new_cap;
    }

    ProfileStat* s = &profiler->stats[profiler->num_stats];
    memset(s, 0, sizeof(*s));
    size_t len = strlen(name) + 1;
    s->name    = cml_malloc(len);
    if (!s->name)
        return -1;
    memcpy(s->name, name, len);
    s->min_ms = 1e300; /* sentinel so the first call sets it */
    return profiler->num_stats++;
}

int profiler_scope_begin(Profiler* profiler, const char* name) {
    if (!profiler || !name || !profiler->enabled)
        return -1;
    if (profiler->stack_depth >= CML_PROF_MAX_DEPTH) {
        LOG_WARNING("profiler: scope stack full (>%d), ignoring '%s'", CML_PROF_MAX_DEPTH, name);
        return -1;
    }
    int si = profiler_stat_index(profiler, name);
    if (si < 0)
        return -1;

    size_t bytes = 0, peak = 0, allocs = 0;
    cml_allocator_get_stats(&bytes, &peak, &allocs);

    ProfScope* sc      = &profiler->stack[profiler->stack_depth++];
    sc->stat           = si;
    sc->child_ms       = 0.0;
    sc->alloc_at_start = allocs;
    sc->bytes_at_start = bytes;
    sc->start_ms       = now_ms(); /* read the clock last so setup is not timed */
    return 0;
}

void profiler_scope_end(Profiler* profiler) {
    if (!profiler || !profiler->enabled)
        return;
    double end = now_ms();
    if (profiler->stack_depth <= 0) {
        LOG_WARNING("profiler: scope_end with no open scope");
        return;
    }

    ProfScope* sc    = &profiler->stack[--profiler->stack_depth];
    double inclusive = end - sc->start_ms;
    double self      = inclusive - sc->child_ms;
    if (self < 0.0)
        self = 0.0; /* clock jitter between parent and child reads */

    /* This scope's inclusive time counts as child time for its parent. */
    if (profiler->stack_depth > 0)
        profiler->stack[profiler->stack_depth - 1].child_ms += inclusive;

    size_t bytes = 0, peak = 0, allocs = 0;
    cml_allocator_get_stats(&bytes, &peak, &allocs);

    ProfileStat* s = &profiler->stats[sc->stat];
    s->count++;
    s->total_ms += inclusive;
    s->self_ms += self;
    if (inclusive < s->min_ms)
        s->min_ms = inclusive;
    if (inclusive > s->max_ms)
        s->max_ms = inclusive;
    s->alloc_count += (long)(allocs - sc->alloc_at_start);
    s->net_bytes += (long long)bytes - (long long)sc->bytes_at_start;
}

const ProfileStat* profiler_find_stat(const Profiler* profiler, const char* name) {
    if (!profiler || !name)
        return NULL;
    for (int i = 0; i < profiler->num_stats; i++)
        if (profiler->stats[i].name && strcmp(profiler->stats[i].name, name) == 0)
            return &profiler->stats[i];
    return NULL;
}

int profiler_write_json(Profiler* profiler, const char* path) {
    if (!profiler || !path)
        return -1;
    FILE* f = fopen(path, "w");
    if (!f)
        return -1;

    fprintf(f, "{\n  \"scopes\": [\n");
    for (int i = 0; i < profiler->num_stats; i++) {
        const ProfileStat* s = &profiler->stats[i];
        double min_ms        = s->min_ms >= 1e300 ? 0.0 : s->min_ms;
        fprintf(
            f,
            "    {\"name\": \"%s\", \"count\": %ld, \"total_ms\": %.6f, \"self_ms\": %.6f, "
            "\"min_ms\": %.6f, \"max_ms\": %.6f, \"alloc_count\": %ld, \"net_bytes\": %lld}%s\n",
            s->name ? s->name : "", s->count, s->total_ms, s->self_ms, min_ms, s->max_ms,
            s->alloc_count, s->net_bytes, i + 1 < profiler->num_stats ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    return 0;
}

/* ---------------- Process-global default profiler ---------------- */

static Profiler* g_default_profiler = NULL;

Profiler* cml_profiler_default(void) {
    if (!g_default_profiler) {
        g_default_profiler = profiler_create();
        if (g_default_profiler) {
            /* Off unless CML_PROFILE is set to something other than "0", so the
             * CML_PROFILE_BEGIN/END macros are free in a normal build. */
            const char* env             = getenv("CML_PROFILE");
            g_default_profiler->enabled = env && env[0] && strcmp(env, "0") != 0;
        }
    }
    return g_default_profiler;
}

void cml_profiler_report(void) {
    if (g_default_profiler)
        profiler_print_report(g_default_profiler);
}
