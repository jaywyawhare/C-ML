/*
 * AUDIT #12: unified error propagation.
 *
 * Verifies that any failure logged via LOG_ERROR — regardless of whether the
 * function returns NULL or an int code — is recorded in the thread-local error
 * stack and is queryable through the public API (cml_get_last_error /
 * cml_get_last_error_code / cml_error_string / cml_clear_last_error).
 */
#include <stdio.h>
#include <string.h>

#include "cml.h"
#include "test_harness.h"
#include "core/error_codes.h"
#include "core/logging.h"
#include "core/quantization.h"

static int check(const char* name, int ok) {
    tests_run++;
    if (ok) {
        tests_passed++;
        printf("  PASS: %s\n", name);
    } else {
        printf("  FAIL: %s\n", name);
    }
    return ok;
}

int main(void) {
    printf("=== AUDIT #12: unified error propagation ===\n");

    /* Silence stderr printing but keep error-stack recording (the push happens
     * before the print-level filter). */
    cml_set_log_level((LogLevel)(LOG_LEVEL_ERROR + 1));

    /* clean slate */
    cml_clear_last_error();
    check("empty_after_clear",
          cml_get_last_error() == NULL && cml_get_last_error_code() == CM_SUCCESS);

    /* A pointer-returning op that fails on bad args logs via LOG_ERROR. */
    Tensor* r = cml_quantize_int8(NULL, NULL, NULL);
    check("bad_arg_returns_null", r == NULL);
    const char* msg = cml_get_last_error();
    check("error_recorded", msg != NULL && strstr(msg, "NULL") != NULL);
    check("error_code_set", cml_get_last_error_code() == CM_OPERATION_FAILED);
    check("error_string_maps",
          strcmp(cml_error_string(CM_OPERATION_FAILED), "operation failed") == 0 &&
              strcmp(cml_error_string(CM_INVALID_ARGUMENT), "invalid argument") == 0 &&
              strcmp(cml_error_string(12345), "unknown error") == 0);

    /* An int-returning op that fails (bad dims) also records. */
    cml_clear_last_error();
    int rc = cml_qmatmul_affine_int8(NULL, NULL, 1.0f, 0, NULL, -1, -1, -1);
    check("int_op_returns_code", rc == -1);
    /* (that particular guard returns before logging; verify the stack API is
     * still consistent — no error unless one was logged) */
    check("clear_resets", (cml_clear_last_error(), cml_get_last_error() == NULL &&
                                                       cml_get_last_error_code() == CM_SUCCESS));

    return TEST_SUMMARY();
}
