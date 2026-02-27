#ifndef CML_LOGGING_H
#define CML_LOGGING_H

#include <stdarg.h>

typedef enum {
    LOG_LEVEL_DEBUG   = 0,
    LOG_LEVEL_INFO    = 1,
    LOG_LEVEL_WARNING = 2,
    LOG_LEVEL_ERROR   = 3
} LogLevel;

#ifdef __cplusplus
extern "C" {
#endif

// Global log level
extern LogLevel g_log_level;

void cml_set_log_level(LogLevel level);
void cml_log_message(LogLevel level, const char* file, int line, const char* func,
                     const char* format, ...);

#define LOG_DEBUG(...) cml_log_message(LOG_LEVEL_DEBUG, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_INFO(...) cml_log_message(LOG_LEVEL_INFO, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_WARNING(...)                                                                           \
    cml_log_message(LOG_LEVEL_WARNING, __FILE__, __LINE__, __func__, __VA_ARGS__)
#define LOG_ERROR(...) cml_log_message(LOG_LEVEL_ERROR, __FILE__, __LINE__, __func__, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // CML_LOGGING_H
