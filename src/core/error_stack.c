#include "core/error_stack.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_ERROR_STACK_SIZE 256
#define MAX_ERROR_MESSAGE_LEN 512

typedef void (*ErrorStackNotifyFn)(int code, const char* message, void* context);

static __thread ErrorEntry* g_error_stack      = NULL;
static __thread size_t g_error_stack_size      = 0;
static __thread size_t g_error_stack_capacity  = 0;
static __thread bool g_error_stack_initialized = false;
static __thread char g_error_message_buffer[MAX_ERROR_STACK_SIZE][MAX_ERROR_MESSAGE_LEN];
static __thread char g_error_file_buffer[MAX_ERROR_STACK_SIZE][256];
static __thread char g_error_function_buffer[MAX_ERROR_STACK_SIZE][128];
static __thread size_t g_message_buffer_index = 0;

static ErrorStackNotifyFn g_error_notify_fn = NULL;
static void* g_error_notify_context         = NULL;
static pthread_mutex_t g_notify_lock        = PTHREAD_MUTEX_INITIALIZER;

/** Lazily allocate the thread-local error stack on first use. */
static void error_stack_ensure_initialized(void) {
    if (g_error_stack_initialized)
        return;

    g_error_stack_capacity = MAX_ERROR_STACK_SIZE;
    g_error_stack          = (ErrorEntry*)cml_malloc(sizeof(ErrorEntry) * g_error_stack_capacity);
    if (!g_error_stack) {
        fprintf(stderr, "FATAL: Failed to initialize error stack\n");
        return;
    }

    g_error_stack_size        = 0;
    g_error_stack_initialized = true;
    g_message_buffer_index    = 0;
}

/** Install a process-wide callback invoked (under lock) on every pushed error. */
void error_stack_set_notify(ErrorStackNotifyFn fn, void* context) {
    pthread_mutex_lock(&g_notify_lock);
    g_error_notify_fn      = fn;
    g_error_notify_context = context;
    pthread_mutex_unlock(&g_notify_lock);
}

/** Map a CM_* error code to a short human-readable description. */
const char* cml_error_string(int code) {
    switch (code) {
    case CM_SUCCESS:
        return "success";
    case CM_MEMORY_ALLOCATION_ERROR:
        return "memory allocation error";
    case CM_INVALID_ARGUMENT:
        return "invalid argument";
    case CM_OPERATION_FAILED:
        return "operation failed";
    case CM_NOT_IMPLEMENTED:
        return "not implemented";
    case CM_INVALID_STATE:
        return "invalid state";
    default:
        return "unknown error";
    }
}

/** Eagerly initialize the thread-local error stack. */
void error_stack_init(void) { error_stack_ensure_initialized(); }

/** Drop all recorded errors without freeing the backing storage. */
void error_stack_clear(void) {
    if (!g_error_stack_initialized || !g_error_stack)
        return;
    g_error_stack_size     = 0;
    g_message_buffer_index = 0;
}

/** Free the thread-local error stack and reset it to the uninitialized state. */
void error_stack_cleanup(void) {
    if (!g_error_stack_initialized)
        return;

    if (g_error_stack) {
        cml_free(g_error_stack);
        g_error_stack = NULL;
    }

    g_error_stack_size        = 0;
    g_error_stack_capacity    = 0;
    g_error_stack_initialized = false;
    g_message_buffer_index    = 0;
}

/**
 * Record an error (code, message, source location) on the thread-local stack,
 * copying strings into ring buffers. Oldest entry is evicted when full, and any
 * registered notify callback fires. Falls back to stderr if allocation failed.
 */
void error_stack_push(int code, const char* message, const char* file, int line,
                      const char* function) {
    error_stack_ensure_initialized();

    if (!g_error_stack) {
        fprintf(stderr, "Error stack not available: %s (code: %d) at %s:%d in %s\n",
                message ? message : "Unknown error", code, file ? file : "unknown", line,
                function ? function : "unknown");
        return;
    }

    if (g_error_stack_size >= g_error_stack_capacity) {
        memmove(&g_error_stack[0], &g_error_stack[1],
                sizeof(ErrorEntry) * (g_error_stack_size - 1));
        g_error_stack_size--;
    }

    size_t msg_idx = g_message_buffer_index % MAX_ERROR_STACK_SIZE;

    if (message) {
        strncpy(g_error_message_buffer[msg_idx], message, MAX_ERROR_MESSAGE_LEN - 1);
        g_error_message_buffer[msg_idx][MAX_ERROR_MESSAGE_LEN - 1] = '\0';
    } else {
        g_error_message_buffer[msg_idx][0] = '\0';
    }

    if (file) {
        strncpy(g_error_file_buffer[msg_idx], file, 255);
        g_error_file_buffer[msg_idx][255] = '\0';
    } else {
        g_error_file_buffer[msg_idx][0] = '\0';
    }

    if (function) {
        strncpy(g_error_function_buffer[msg_idx], function, 127);
        g_error_function_buffer[msg_idx][127] = '\0';
    } else {
        g_error_function_buffer[msg_idx][0] = '\0';
    }

    ErrorEntry* entry = &g_error_stack[g_error_stack_size];
    entry->code       = code;
    entry->message    = g_error_message_buffer[msg_idx];
    entry->file       = g_error_file_buffer[msg_idx];
    entry->line       = line;
    entry->function   = g_error_function_buffer[msg_idx];

    g_error_stack_size++;
    g_message_buffer_index++;

    pthread_mutex_lock(&g_notify_lock);
    ErrorStackNotifyFn notify_fn = g_error_notify_fn;
    void* notify_ctx             = g_error_notify_context;
    pthread_mutex_unlock(&g_notify_lock);
    if (notify_fn && entry->message)
        notify_fn(code, entry->message, notify_ctx);
}

/** Return the most recently pushed error, or NULL if the stack is empty. */
ErrorEntry* error_stack_peek(void) {
    if (!g_error_stack_initialized || !g_error_stack || g_error_stack_size == 0)
        return NULL;

    return &g_error_stack[g_error_stack_size - 1];
}

/** True if the thread's error stack holds at least one entry. */
bool error_stack_has_errors(void) {
    if (!g_error_stack_initialized)
        return false;

    return g_error_stack_size > 0;
}

/** Print the full error stack (oldest first) with source locations to stderr. */
void error_stack_print_all(void) {
    if (!g_error_stack_initialized || !g_error_stack || g_error_stack_size == 0)
        return;

    fprintf(stderr, "\nError Stack (%zu error(s))\n", g_error_stack_size);
    for (size_t i = 0; i < g_error_stack_size; i++) {
        ErrorEntry* entry = &g_error_stack[i];
        fprintf(stderr, "[%zu] Error %d: %s\n", i + 1, entry->code, entry->message);
        fprintf(stderr, "    at %s:%d in %s\n", entry->file, entry->line, entry->function);
    }
    fprintf(stderr, "\n");
}

/** Message of the most recent error, or NULL if none. */
const char* error_stack_get_last_message(void) {
    ErrorEntry* entry = error_stack_peek();
    return entry ? entry->message : NULL;
}

/** Code of the most recent error, or CM_SUCCESS if none. */
int error_stack_get_last_code(void) {
    ErrorEntry* entry = error_stack_peek();
    return entry ? entry->code : CM_SUCCESS;
}
