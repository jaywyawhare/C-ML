#ifndef CML_CORE_ERROR_STACK_H
#define CML_CORE_ERROR_STACK_H

#include "core/error_codes.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int code;             // Error code
    const char* message;  // Error message
    const char* file;     // Source file
    int line;             // Line number
    const char* function; // Function name
} ErrorEntry;

void error_stack_init(void);
void error_stack_cleanup(void);
void error_stack_push(int code, const char* message, const char* file, int line,
                      const char* function);
ErrorEntry* error_stack_peek(void);
bool error_stack_has_errors(void);
void error_stack_print_all(void);
const char* error_stack_get_last_message(void);
int error_stack_get_last_code(void);

#define CML_AUTO_CHECK_PTR(ptr, msg)                                                               \
    do {                                                                                           \
        if ((ptr) == NULL) {                                                                       \
            error_stack_push(CM_OPERATION_FAILED, (msg), __FILE__, __LINE__, __func__);            \
        }                                                                                          \
    } while (0)

#define CML_AUTO_CHECK_CODE(code, msg)                                                             \
    do {                                                                                           \
        if ((code) != CM_SUCCESS) {                                                                \
            error_stack_push((code), (msg), __FILE__, __LINE__, __func__);                         \
        }                                                                                          \
    } while (0)

#define CML_AUTO_CHECK_PTR_RET(ptr, msg)                                                           \
    ((ptr) == NULL                                                                                 \
         ? (error_stack_push(CM_OPERATION_FAILED, (msg), __FILE__, __LINE__, __func__), (ptr))     \
         : (ptr))

/* CML_CHECK(tensor_empty(...), "Failed to create tensor") */
#define CML_CHECK(expr, msg) CML_AUTO_CHECK_PTR_RET((expr), (msg))

/* CML_CHECK_AUTO(tensor_empty(...)) */
#define CML_CHECK_AUTO(expr) CML_AUTO_CHECK_PTR_RET((expr), "Operation failed: " #expr)

#define CML_HAS_ERRORS() error_stack_has_errors()
#define CML_LAST_ERROR() error_stack_get_last_message()
#define CML_LAST_ERROR_CODE() error_stack_get_last_code()

#ifdef __cplusplus
}
#endif

#endif // CML_CORE_ERROR_STACK_H
