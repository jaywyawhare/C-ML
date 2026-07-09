#ifndef CML_ERROR_CODES_H
#define CML_ERROR_CODES_H

/*
 * Canonical error convention (AUDIT #12)
 * -------------------------------------
 * - Pointer-returning functions return NULL on failure.
 * - Int-returning functions return a CM_* code (CM_SUCCESS == 0, negatives on
 *   failure).
 * - EVERY failure logged via LOG_ERROR is also recorded in the thread-local
 *   error stack (see core/error_stack.h), so callers have ONE reliable way to
 *   introspect any failure regardless of the return convention:
 *
 *       Tensor* t = some_op(...);
 *       if (!t)
 *           fprintf(stderr, "%s (%s)\n", cml_get_last_error(),
 *                   cml_error_string(cml_get_last_error_code()));
 *
 *   Key on the RETURN VALUE for success/failure; consult the error stack for
 *   the reason. Call cml_clear_last_error() before an op for a clean slate.
 */

#define CM_SUCCESS 0
#define CM_MEMORY_ALLOCATION_ERROR -1
#define CM_INVALID_ARGUMENT -2
#define CM_OPERATION_FAILED -3
#define CM_NOT_IMPLEMENTED -4
#define CM_INVALID_STATE -5

#ifdef __cplusplus
extern "C" {
#endif

/* Human-readable name for a CM_* code (stable, never NULL). */
const char* cml_error_string(int code);

#ifdef __cplusplus
}
#endif

#endif // CML_ERROR_CODES_H
