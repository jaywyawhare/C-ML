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

typedef struct Profiler {
    Timer** timers;
    int num_timers;
    int capacity;
    bool enabled;
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

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_PROFILING_H
