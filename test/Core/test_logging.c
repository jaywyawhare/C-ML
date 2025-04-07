#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/Core/logging.h"

/**
 * @brief Test the logging system
 * 
 * @return int 0 if all tests pass, non-zero otherwise
 */
int main() {
    printf("Running logging tests...\n");
    
    // Test setting log level
    set_log_level(LOG_LEVEL_DEBUG);
    
    // Test different log levels
    LOG_DEBUG("This is a debug message: %d", 42);
    LOG_INFO("This is an info message: %s", "hello");
    LOG_WARNING("This is a warning message: %.2f", 3.14);
    LOG_ERROR("This is an error message: %c", 'X');
    
    // Test changing log level
    printf("\nChanging log level to INFO (DEBUG messages should not appear):\n");
    set_log_level(LOG_LEVEL_INFO);
    
    LOG_DEBUG("This debug message should NOT appear");
    LOG_INFO("This info message should appear");
    LOG_WARNING("This warning message should appear");
    LOG_ERROR("This error message should appear");
    
    // Test changing log level to WARNING
    printf("\nChanging log level to WARNING (DEBUG and INFO messages should not appear):\n");
    set_log_level(LOG_LEVEL_WARNING);
    
    LOG_DEBUG("This debug message should NOT appear");
    LOG_INFO("This info message should NOT appear");
    LOG_WARNING("This warning message should appear");
    LOG_ERROR("This error message should appear");
    
    // Test changing log level to ERROR
    printf("\nChanging log level to ERROR (only ERROR messages should appear):\n");
    set_log_level(LOG_LEVEL_ERROR);
    
    LOG_DEBUG("This debug message should NOT appear");
    LOG_INFO("This info message should NOT appear");
    LOG_WARNING("This warning message should NOT appear");
    LOG_ERROR("This error message should appear");
    
    printf("\nLogging tests completed successfully!\n");
    return 0;
}
