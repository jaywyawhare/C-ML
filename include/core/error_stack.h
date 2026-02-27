/**
 * @file error_stack.h
 * @brief Global error stack and automatic error checking
 *
 * This header provides a global error stack system and automatic error checking
 * macros. Users don't need to write explicit error checks - everything is automatic!
 */

#ifndef CML_CORE_ERROR_STACK_H
#define CML_CORE_ERROR_STACK_H

#include "core/error_codes.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error entry in the error stack
 */
typedef struct {
    int code;             // Error code
    const char* message;  // Error message
    const char* file;     // Source file
    int line;             // Line number
    const char* function; // Function name
} ErrorEntry;

/**
 * @brief Initialize the error stack system
 */
void error_stack_init(void);

/**
 * @brief Cleanup the error stack system
 */
void error_stack_cleanup(void);

/**
 * @brief Push an error onto the error stack
 *
 * @param code Error code
 * @param message Error message
 * @param file Source file
 * @param line Line number
 * @param function Function name
 */
void error_stack_push(int code, const char* message, const char* file, int line,
                      const char* function);

/**
 * @brief Get the last error without removing it
 *
 * @return Error entry, or NULL if stack is empty
 */
ErrorEntry* error_stack_peek(void);

/**
 * @brief Check if there are any errors
 *
 * @return true if errors exist, false otherwise
 */
bool error_stack_has_errors(void);

/**
 * @brief Print all errors to stderr
 */
void error_stack_print_all(void);

/**
 * @brief Get the last error message
 *
 * @return Error message, or NULL if no errors
 */
const char* error_stack_get_last_message(void);

/**
 * @brief Get the last error code
 *
 * @return Error code, or CM_SUCCESS if no errors
 */
int error_stack_get_last_code(void);

/**
 * @brief Automatically check if a pointer is NULL and push error if so
 *
 * This is used internally by RAII wrappers - users don't need to call this.
 */
#define CML_AUTO_CHECK_PTR(ptr, msg)                                                               \
    do {                                                                                           \
        if ((ptr) == NULL) {                                                                       \
            error_stack_push(CM_OPERATION_FAILED, (msg), __FILE__, __LINE__, __func__);            \
        }                                                                                          \
    } while (0)

/**
 * @brief Automatically check if a return code indicates error
 *
 * This is used internally by RAII wrappers - users don't need to call this.
 */
#define CML_AUTO_CHECK_CODE(code, msg)                                                             \
    do {                                                                                           \
        if ((code) != CM_SUCCESS) {                                                                \
            error_stack_push((code), (msg), __FILE__, __LINE__, __func__);                         \
        }                                                                                          \
    } while (0)

/**
 * @brief Automatically check if a pointer is NULL (returns NULL if error)
 *
 * This is used internally by RAII wrappers - users don't need to call this.
 */
#define CML_AUTO_CHECK_PTR_RET(ptr, msg)                                                           \
    ((ptr) == NULL                                                                                 \
         ? (error_stack_push(CM_OPERATION_FAILED, (msg), __FILE__, __LINE__, __func__), (ptr))     \
         : (ptr))

/**
 * @brief Automatically check a function result and push error if NULL
 *
 * Usage:
 *   Tensor* t = CML_CHECK(tensor_empty(...), "Failed to create tensor");
 *
 * This macro automatically checks if the result is NULL and pushes an error
 * to the error stack if so. The original value is returned.
 */
#define CML_CHECK(expr, msg) CML_AUTO_CHECK_PTR_RET((expr), (msg))

/**
 * @brief Automatically check a function result and push error if NULL (with default message)
 *
 * Usage:
 *   Tensor* t = CML_CHECK_AUTO(tensor_empty(...));
 *
 * This macro automatically checks if the result is NULL and pushes an error
 * with a default message based on the expression.
 */
#define CML_CHECK_AUTO(expr) CML_AUTO_CHECK_PTR_RET((expr), "Operation failed: " #expr)

/**
 * @brief Check if there are errors and handle them
 *
 * Usage:
 *   if (CML_HAS_ERRORS()) {
 *       error_stack_print_all();
 *       return -1;
 *   }
 */
#define CML_HAS_ERRORS() error_stack_has_errors()

/**
 * @brief Get the last error message
 */
#define CML_LAST_ERROR() error_stack_get_last_message()

/**
 * @brief Get the last error code
 */
#define CML_LAST_ERROR_CODE() error_stack_get_last_code()

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_ERROR_STACK_H
