/**
 * @file error_stack.c
 * @brief Global error stack implementation
 *
 * This module provides a global error stack that automatically tracks errors
 * without requiring explicit null checks in user code.
 */

#include "core/error_stack.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ERROR_STACK_SIZE 256
#define MAX_ERROR_MESSAGE_LEN 512

static ErrorEntry* g_error_stack      = NULL;
static size_t g_error_stack_size      = 0;
static size_t g_error_stack_capacity  = 0;
static bool g_error_stack_initialized = false;

static char g_error_message_buffer[MAX_ERROR_STACK_SIZE][MAX_ERROR_MESSAGE_LEN];
static char g_error_file_buffer[MAX_ERROR_STACK_SIZE][256];
static char g_error_function_buffer[MAX_ERROR_STACK_SIZE][128];
static size_t g_message_buffer_index = 0;

void error_stack_init(void) {
    if (g_error_stack_initialized) {
        return;
    }

    g_error_stack_capacity = MAX_ERROR_STACK_SIZE;
    g_error_stack          = (ErrorEntry*)malloc(sizeof(ErrorEntry) * g_error_stack_capacity);
    if (!g_error_stack) {
        // Can't use error stack if allocation fails
        fprintf(stderr, "FATAL: Failed to initialize error stack\n");
        return;
    }

    g_error_stack_size        = 0;
    g_error_stack_initialized = true;
    g_message_buffer_index    = 0;

    LOG_DEBUG("Error stack initialized");
}

void error_stack_cleanup(void) {
    if (!g_error_stack_initialized) {
        return;
    }

    if (g_error_stack) {
        free(g_error_stack);
        g_error_stack = NULL;
    }

    g_error_stack_size        = 0;
    g_error_stack_capacity    = 0;
    g_error_stack_initialized = false;
    g_message_buffer_index    = 0;

    LOG_DEBUG("Error stack cleaned up");
}

void error_stack_push(int code, const char* message, const char* file, int line,
                      const char* function) {
    if (!g_error_stack_initialized) {
        error_stack_init();
    }

    if (!g_error_stack) {
        fprintf(stderr, "Error stack not available: %s (code: %d) at %s:%d in %s\n",
                message ? message : "Unknown error", code, file ? file : "unknown", line,
                function ? function : "unknown");
        return;
    }

    // Check if stack is full
    if (g_error_stack_size >= g_error_stack_capacity) {
        // Remove oldest error (FIFO behavior)
        memmove(&g_error_stack[0], &g_error_stack[1],
                sizeof(ErrorEntry) * (g_error_stack_size - 1));
        g_error_stack_size--;
    }

    // Get buffer for message, file, and function strings
    size_t msg_idx = g_message_buffer_index % MAX_ERROR_STACK_SIZE;

    // Copy message
    if (message) {
        strncpy(g_error_message_buffer[msg_idx], message, MAX_ERROR_MESSAGE_LEN - 1);
        g_error_message_buffer[msg_idx][MAX_ERROR_MESSAGE_LEN - 1] = '\0';
    } else {
        g_error_message_buffer[msg_idx][0] = '\0';
    }

    // Copy file
    if (file) {
        strncpy(g_error_file_buffer[msg_idx], file, 255);
        g_error_file_buffer[msg_idx][255] = '\0';
    } else {
        g_error_file_buffer[msg_idx][0] = '\0';
    }

    // Copy function
    if (function) {
        strncpy(g_error_function_buffer[msg_idx], function, 127);
        g_error_function_buffer[msg_idx][127] = '\0';
    } else {
        g_error_function_buffer[msg_idx][0] = '\0';
    }

    // Add error to stack
    ErrorEntry* entry = &g_error_stack[g_error_stack_size];
    entry->code       = code;
    entry->message    = g_error_message_buffer[msg_idx];
    entry->file       = g_error_file_buffer[msg_idx];
    entry->line       = line;
    entry->function   = g_error_function_buffer[msg_idx];

    g_error_stack_size++;
    g_message_buffer_index++;

    LOG_DEBUG("Error pushed: %s (code: %d) at %s:%d in %s", entry->message, entry->code,
              entry->file, entry->line, entry->function);
}

ErrorEntry* error_stack_peek(void) {
    if (!g_error_stack_initialized || !g_error_stack || g_error_stack_size == 0) {
        return NULL;
    }

    return &g_error_stack[g_error_stack_size - 1];
}

bool error_stack_has_errors(void) {
    if (!g_error_stack_initialized) {
        return false;
    }

    return g_error_stack_size > 0;
}

void error_stack_print_all(void) {
    if (!g_error_stack_initialized || !g_error_stack || g_error_stack_size == 0) {
        return;
    }

    fprintf(stderr, "\n=== Error Stack (%zu error(s)) ===\n", g_error_stack_size);
    for (size_t i = 0; i < g_error_stack_size; i++) {
        ErrorEntry* entry = &g_error_stack[i];
        fprintf(stderr, "[%zu] Error %d: %s\n", i + 1, entry->code, entry->message);
        fprintf(stderr, "    at %s:%d in %s\n", entry->file, entry->line, entry->function);
    }
    fprintf(stderr, "================================\n\n");
}

const char* error_stack_get_last_message(void) {
    ErrorEntry* entry = error_stack_peek();
    return entry ? entry->message : NULL;
}

int error_stack_get_last_code(void) {
    ErrorEntry* entry = error_stack_peek();
    return entry ? entry->code : CM_SUCCESS;
}
