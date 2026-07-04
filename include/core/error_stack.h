#ifndef CML_CORE_ERROR_STACK_H
#define CML_CORE_ERROR_STACK_H

#include "core/error_codes.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int code;
    const char* message;
    const char* file;
    int line;
    const char* function;
} ErrorEntry;

void error_stack_init(void);
void error_stack_cleanup(void);
void error_stack_clear(void);
void error_stack_push(int code, const char* message, const char* file, int line,
                      const char* function);
void error_stack_set_notify(void (*fn)(int code, const char* message, void* context),
                            void* context);
ErrorEntry* error_stack_peek(void);
bool error_stack_has_errors(void);
void error_stack_print_all(void);
const char* error_stack_get_last_message(void);
int error_stack_get_last_code(void);

#define CML_ERR(code, msg)                                                                         \
    do {                                                                                           \
        error_stack_push((code), (msg), __FILE__, __LINE__, __func__);                             \
    } while (0)

#define CML_ERR_RET(code, msg, ret)                                                                \
    do {                                                                                           \
        error_stack_push((code), (msg), __FILE__, __LINE__, __func__);                             \
        return (ret);                                                                              \
    } while (0)

#define CML_ERR_NULL(msg) CML_ERR_RET(CM_OPERATION_FAILED, (msg), NULL)
#define CML_ERR_INT(msg) CML_ERR_RET(CM_OPERATION_FAILED, (msg), -1)

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

#define CML_CHECK(expr, msg) CML_AUTO_CHECK_PTR_RET((expr), (msg))
#define CML_CHECK_AUTO(expr) CML_AUTO_CHECK_PTR_RET((expr), "Operation failed: " #expr)

#define CML_HAS_ERRORS() error_stack_has_errors()
#define CML_LAST_ERROR() error_stack_get_last_message()
#define CML_LAST_ERROR_CODE() error_stack_get_last_code()

#ifdef __cplusplus
}
#endif

#endif /* CML_CORE_ERROR_STACK_H */
