/*
 * IR graph validation — mirrors TinyGrad's uop/validate.py.
 *
 * Checks structural and semantic correctness of a CMLGraph before lowering:
 *   - No NULL or dangling inputs
 *   - No cycles (DAG check)
 *   - Correct input count per UOpType
 *   - Shape consistency across edges
 *   - DType legality per operation
 *   - Reduction axes in range
 *   - No dead outputs
 */

#ifndef CML_OPS_IR_VALIDATE_H
#define CML_OPS_IR_VALIDATE_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Error codes ---- */

typedef enum {
    CML_VALID_OK                = 0,
    CML_VALID_NULL_GRAPH        = 1,
    CML_VALID_NULL_NODE         = 2,
    CML_VALID_CYCLE             = 3,  /* graph contains a cycle */
    CML_VALID_MISSING_INPUT     = 4,  /* required input slot is NULL */
    CML_VALID_TOO_MANY_INPUTS   = 5,
    CML_VALID_SHAPE_MISMATCH    = 6,
    CML_VALID_DTYPE_MISMATCH    = 7,
    CML_VALID_INVALID_AXIS      = 8,  /* reduction/slice axis out of range */
    CML_VALID_UNKNOWN_OP        = 9,
    CML_VALID_EMPTY_GRAPH       = 10,
    CML_VALID_DEAD_OUTPUT       = 11, /* output node unreachable from any sink */
    CML_VALID_INTERNAL          = 99,
} CMLValidateCode;

/* ---- Diagnostic record ---- */

#define CML_VALIDATE_MSG_LEN 256

typedef struct CMLValidateDiag {
    CMLValidateCode code;
    char            message[CML_VALIDATE_MSG_LEN];
    int             node_index;   /* which node triggered the error, or -1 */
} CMLValidateDiag;

/* ---- Validation options ---- */

typedef struct CMLValidateOpts {
    bool check_shapes;   /* check shape consistency (default true)  */
    bool check_dtypes;   /* check dtype legality    (default true)  */
    bool check_cycles;   /* DFS cycle detection     (default true)  */
    bool check_dead;     /* warn on dead nodes      (default false) */
    int  max_diags;      /* cap on diag entries     (0 = unlimited) */
} CMLValidateOpts;

CMLValidateOpts cml_validate_default_opts(void);

/* ---- Main entry points ---- */

/*
 * Validate the graph.
 * Returns CML_VALID_OK (0) if clean, or the first error code encountered.
 * If diags != NULL, fills up to max_diag_count entries and writes the
 * actual count into *num_diags_out.
 */
CMLValidateCode cml_validate_graph(CMLGraph_t ir,
                                   const CMLValidateOpts* opts,
                                   CMLValidateDiag* diags,
                                   int max_diag_count,
                                   int* num_diags_out);

/* Convenience: validate and abort with a log message on failure. */
void cml_validate_graph_or_die(CMLGraph_t ir);

/* Print all diagnostics to stderr. */
void cml_validate_print_diags(const CMLValidateDiag* diags, int num_diags);

/* Human-readable description of a validate code. */
const char* cml_validate_code_str(CMLValidateCode code);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_VALIDATE_H */
