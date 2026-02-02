#include "core/logging.h"
#include <stdarg.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

LogLevel g_log_level = LOG_LEVEL_ERROR;

/**
 * @brief Set the global log level
 *
 * @param level The log level to set
 */
void cml_set_log_level(LogLevel level) { g_log_level = level; }

/**
 * @brief Core logging function
 *
 * @param level Log level
 * @param file Source file name
 * @param line Line number
 * @param func Function name
 * @param format Format string
 * @param ... Variable arguments
 */
void cml_log_message(LogLevel level, const char* file, int line, const char* func,
                     const char* format, ...) {
    if (level < g_log_level) {
        return;
    }

    time_t now         = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_str[20];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);

    const char* level_str[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

    fprintf(stderr, "%s [%s] %s:%d %s(): ", time_str, level_str[level], file, line, func);

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");
}
