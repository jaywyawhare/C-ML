/*
 * Z3 SMT solver for IR validation.
 * Uses Z3 theorem prover to verify IR transformations preserve semantics,
 * validate index bounds, and check schedule correctness.
 */

#ifndef CML_Z3_VERIFY_H
#define CML_Z3_VERIFY_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_VERIFY_PASS = 0,
    CML_VERIFY_FAIL,
    CML_VERIFY_TIMEOUT,
    CML_VERIFY_UNSUPPORTED,
} CMLVerifyResult;

typedef struct CMLZ3Verifier {
    bool initialized;
    void* z3_context;        /* Z3_context handle */
    void* z3_solver;         /* Z3_solver handle */
    int timeout_ms;
    int num_checks;
    int num_passed;
    int num_failed;
} CMLZ3Verifier;

bool cml_z3_available(void);
CMLZ3Verifier* cml_z3_verifier_create(int timeout_ms);
void cml_z3_verifier_free(CMLZ3Verifier* v);
CMLVerifyResult cml_z3_verify_equivalence(CMLZ3Verifier* v,
                                           CMLGraph_t original,
                                           CMLGraph_t optimized);
CMLVerifyResult cml_z3_verify_bounds(CMLZ3Verifier* v, CMLGraph_t ir);
CMLVerifyResult cml_z3_verify_schedule(CMLZ3Verifier* v,
                                        CMLGraph_t ir, void* schedule);
void cml_z3_verifier_stats(const CMLZ3Verifier* v,
                            int* checks, int* passed, int* failed);

#ifdef __cplusplus
}
#endif

#endif /* CML_Z3_VERIFY_H */
