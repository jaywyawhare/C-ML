#include "core/logging.h"
#include "core/error_stack.h"
#include <stdarg.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

LogLevel g_log_level = LOG_LEVEL_ERROR;

void cml_set_log_level(LogLevel level) { g_log_level = level; }

void cml_log_message(LogLevel level, const char* file, int line, const char* func,
                     const char* format, ...) {
    char msg[512];
    va_list args;
    va_start(args, format);
    vsnprintf(msg, sizeof(msg), format, args);
    va_end(args);

    /* Unified error propagation: every ERROR is recorded in the thread-local
     * error stack so failures are queryable via cml_get_last_error() /
     * cml_get_last_error_code() regardless of a function's NULL/-1 return
     * convention.  Re-entry guarded in case error_stack_push logs. */
    if (level == LOG_LEVEL_ERROR) {
        static __thread bool in_push = false;
        if (!in_push) {
            in_push = true;
            error_stack_push(CM_OPERATION_FAILED, msg, file, line, func);
            in_push = false;
        }
    }

    if (level < g_log_level) {
        return;
    }

    time_t now         = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_str[20];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);

    const char* level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

    fprintf(stderr, "%s [%s] %s:%d %s(): %s\n", time_str, level_str[level], file, line, func, msg);
}
