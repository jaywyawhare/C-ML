#include "core/logging.h"
#include "core/error_stack.h"
#include "core/cml_flags.h"
#include <stdarg.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

    /* Color the level tag on a TTY unless NO_COLOR is set (tinygrad-style). */
    static const char* level_color[] = {"\033[90m", "\033[36m", "\033[33m", "\033[31m"};
    bool use_color = !cml_flag_enabled(CML_FLAG_NO_COLOR) && isatty(fileno(stderr));

    if (use_color) {
        fprintf(stderr, "%s [%s%s\033[0m] %s:%d %s(): %s\n", time_str, level_color[level],
                level_str[level], file, line, func, msg);
    } else {
        fprintf(stderr, "%s [%s] %s:%d %s(): %s\n", time_str, level_str[level], file, line, func,
                msg);
    }
}

void cml_json_write_escaped(FILE* f, const char* s) {
    if (!s) {
        fputs("null", f);
        return;
    }
    fputc('"', f);
    for (const char* p = s; *p; p++) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', f);
            fputc(*p, f);
        } else if ((unsigned char)*p < 0x20) {
            fprintf(f, "\\u%04x", (unsigned char)*p);
        } else {
            fputc(*p, f);
        }
    }
    fputc('"', f);
}
