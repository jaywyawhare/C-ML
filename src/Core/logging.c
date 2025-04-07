#include "../../include/Core/logging.h"
#include <stdarg.h>
#include <time.h>

// Default to INFO level
LogLevel g_log_level = LOG_LEVEL_INFO;

/**
 * @brief Set the global log level
 * 
 * @param level The log level to set
 */
void set_log_level(LogLevel level) {
    g_log_level = level;
}

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
void log_message(LogLevel level, const char *file, int line, const char *func, const char *format, ...) {
    // Skip if below current log level
    if (level < g_log_level) {
        return;
    }
    
    // Get current time
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_str[20];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    
    // Level strings
    const char *level_str[] = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR"
    };
    
    // Print log header
    fprintf(stderr, "%s [%s] %s:%d %s(): ", 
            time_str, level_str[level], file, line, func);
    
    // Print the actual message with variable arguments
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    // Add newline
    fprintf(stderr, "\n");
}
