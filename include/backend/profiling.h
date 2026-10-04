#ifndef CML_CORE_PROFILING_H
#define CML_CORE_PROFILING_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Timer {
    struct timespec start_time;
    struct timespec end_time;
    double elapsed_ms;
    bool is_running;
    char* name; // Owned by this struct
} Timer;

#define CML_PROF_MAX_DEPTH 64

/** Aggregated timing and memory for every scope sharing a name. */
typedef struct ProfileStat {
    char* name;
    long count;          /* number of begin/end pairs */
    double total_ms;     /* inclusive wall time (includes nested children) */
    double self_ms;      /* exclusive wall time (children subtracted) */
    double min_ms;       /* shortest single inclusive call */
    double max_ms;       /* longest single inclusive call */
    long alloc_count;    /* cml_* allocations made inside the scope */
    long long net_bytes; /* net change in live allocator bytes inside the scope */
} ProfileStat;

/** One open scope on the profiler's call stack (see profiler_scope_begin). */
typedef struct ProfScope {
    int stat;              /* index into Profiler.stats */
    double start_ms;       /* monotonic start */
    double child_ms;       /* inclusive time of direct children, for self time */
    size_t alloc_at_start; /* allocator alloc_count snapshot at begin */
    size_t bytes_at_start; /* allocator live-bytes snapshot at begin */
} ProfScope;

typedef struct Profiler {
    Timer** timers;
    int num_timers;
    int capacity;
    bool enabled;

    /* Aggregated, nestable scopes (profiler_scope_begin/end). Kept separate
     * from the flat timer list above, which remains for one-shot start/stop
     * timing. A Profiler instance is single-threaded; use one per thread. */
    ProfileStat* stats;
    int num_stats;
    int stats_capacity;
    ProfScope stack[CML_PROF_MAX_DEPTH];
    int stack_depth;
} Profiler;

Timer* profiler_timer_create(const char* name);
void profiler_timer_free(Timer* timer);
int profiler_timer_start(Timer* timer);
double profiler_timer_stop(Timer* timer);
double profiler_timer_elapsed(Timer* timer);
void profiler_timer_reset(Timer* timer);

Profiler* profiler_create(void);
void profiler_free(Profiler* profiler);
void profiler_set_enabled(Profiler* profiler, bool enabled);
int profiler_start(Profiler* profiler, const char* name);
double profiler_stop(Profiler* profiler, int timer_id);
void profiler_print_report(Profiler* profiler);
double profiler_get_total_time(Profiler* profiler, const char* name);

/* --- Nestable aggregated scopes (inclusive vs self time + memory) --------- */

/** Open a scope named `name`. Every begin must be matched by one end; scopes
 *  may nest up to CML_PROF_MAX_DEPTH. Stats accumulate per name. Returns 0, or
 *  -1 if the profiler is disabled, the stack is full, or on allocation failure. */
int profiler_scope_begin(Profiler* profiler, const char* name);

/** Close the innermost open scope, folding its inclusive time into its parent's
 *  child time and updating the named stat (count, inclusive/self time, min/max,
 *  and allocator activity recorded between its begin and end). */
void profiler_scope_end(Profiler* profiler);

/** Aggregated stat for `name`, or NULL if no scope by that name has closed. */
const ProfileStat* profiler_find_stat(const Profiler* profiler, const char* name);

/** Write the aggregated scope stats to `path` as JSON. Returns 0, or -1. */
int profiler_write_json(Profiler* profiler, const char* path);

/** Process-global profiler, created on first use. Enabled iff $CML_PROFILE is
 *  set to a non-zero value (or profiler_set_enabled is called on it). Intended
 *  for the main thread. */
Profiler* cml_profiler_default(void);

/** Print the default profiler's report (no-op if it was never used). */
void cml_profiler_report(void);

/** Scope the default profiler around a block: CML_PROFILE_BEGIN("x"); ...; CML_PROFILE_END(); */
#define CML_PROFILE_BEGIN(name) profiler_scope_begin(cml_profiler_default(), (name))
#define CML_PROFILE_END() profiler_scope_end(cml_profiler_default())

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_PROFILING_H
