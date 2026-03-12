/**
 * @file z3_verify.c
 * @brief Z3 SMT solver for IR validation (stub)
 */

#include "ops/ir/z3_verify.h"
#include <stdlib.h>
#include <string.h>

bool cml_z3_available(void) {
    /* Would check for libz3 */
    return false;
}

CMLZ3Verifier* cml_z3_verifier_create(int timeout_ms) {
    CMLZ3Verifier* v = (CMLZ3Verifier*)calloc(1, sizeof(CMLZ3Verifier));
    if (v) v->timeout_ms = timeout_ms;
    return v;
}

void cml_z3_verifier_free(CMLZ3Verifier* v) {
    free(v);
}

CMLVerifyResult cml_z3_verify_equivalence(CMLZ3Verifier* v,
                                           CMLGraph_t original,
                                           CMLGraph_t optimized) {
    (void)original; (void)optimized;
    if (!v || !cml_z3_available()) return CML_VERIFY_UNSUPPORTED;
    return CML_VERIFY_UNSUPPORTED;
}

CMLVerifyResult cml_z3_verify_bounds(CMLZ3Verifier* v, CMLGraph_t ir) {
    (void)ir;
    if (!v || !cml_z3_available()) return CML_VERIFY_UNSUPPORTED;
    return CML_VERIFY_UNSUPPORTED;
}

CMLVerifyResult cml_z3_verify_schedule(CMLZ3Verifier* v,
                                        CMLGraph_t ir, void* schedule) {
    (void)ir; (void)schedule;
    if (!v || !cml_z3_available()) return CML_VERIFY_UNSUPPORTED;
    return CML_VERIFY_UNSUPPORTED;
}

void cml_z3_verifier_stats(const CMLZ3Verifier* v,
                            int* checks, int* passed, int* failed) {
    if (!v) return;
    if (checks) *checks = v->num_checks;
    if (passed) *passed = v->num_passed;
    if (failed) *failed = v->num_failed;
}
