#ifndef C_ML_LOGGING_H
#define C_ML_LOGGING_H

#include <stdio.h>

/**
 * @brief Log levels for the C-ML library
 */
typedef enum {
    LOG_LEVEL_DEBUG,   // Detailed information, typically of interest only when diagnosing problems
    LOG_LEVEL_INFO,    // Confirmation that things are working as expected
    LOG_LEVEL_WARNING, // Indication that something unexpected happened, but the program is still working
    LOG_LEVEL_ERROR    // Due to a more serious problem, the program has not been able to perform a function
} LogLevel;

// Global log level (can be set at runtime)
extern LogLevel g_log_level;

/**
 * @brief Set the global log level
 * 
 * @param level The log level to set
 */
void set_log_level(LogLevel level);

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
void log_message(LogLevel level, const char *file, int line, const char *func, const char *format, ...);

// Logging macros
#define LOG_DEBUG(format, ...) \
    log_message(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

#define LOG_INFO(format, ...) \
    log_message(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

#define LOG_WARNING(format, ...) \
    log_message(LOG_LEVEL_WARNING, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

#define LOG_ERROR(format, ...) \
    log_message(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

#endif // C_ML_LOGGING_H
