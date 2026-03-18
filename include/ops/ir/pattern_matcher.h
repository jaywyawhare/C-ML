/*
 * Declarative pattern-matcher compiler for IR rewrites.
 * (pattern -> replacement) rules that match IRNode subgraphs.
 */

#ifndef CML_OPS_IR_PATTERN_MATCHER_H
#define CML_OPS_IR_PATTERN_MATCHER_H

#include "ops/uops.h"
#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CML_PATTERN_MAX_INPUTS 4
#define CML_PATTERN_MAX_CAPTURES 8
#define CML_REWRITE_MAX_RULES 64
#define CML_REWRITE_DEFAULT_MAX_ITER 16

typedef enum {
    CML_PAT_OP,       /* match a specific UOpType */
    CML_PAT_CAPTURE,  /* capture any single node by name */
    CML_PAT_ANY,      /* match anything (wildcard) */
} CMLPatternKind;

typedef struct CMLPatternNode {
    CMLPatternKind kind;
    UOpType op_type;                            /* for CML_PAT_OP */
    char capture_name[32];                      /* for CML_PAT_CAPTURE */
    struct CMLPatternNode* inputs[CML_PATTERN_MAX_INPUTS];
    int num_inputs;
} CMLPatternNode;

typedef struct {
    char name[32];
    struct IRNode* node;
} CMLCaptureEntry;

typedef struct {
    CMLCaptureEntry captures[CML_PATTERN_MAX_CAPTURES];
    int num_captures;
    struct IRNode* matched_root;
} CMLMatchResult;

typedef struct IRNode* (*CMLEmitFn)(CMLGraph_t ir, const CMLMatchResult* match);

typedef struct {
    CMLPatternNode* pattern;
    CMLEmitFn emit;
    int priority;         /* higher = applied first */
    const char* name;     /* debug name */
} CMLRewriteRule;

typedef struct {
    CMLRewriteRule rules[CML_REWRITE_MAX_RULES];
    int num_rules;
} CMLRewriteRegistry;

CMLPatternNode* cml_pattern_op(UOpType type, CMLPatternNode** inputs, int num_inputs);
CMLPatternNode* cml_pattern_capture(const char* name);
CMLPatternNode* cml_pattern_any(void);
void cml_pattern_free(CMLPatternNode* node);

CMLRewriteRegistry* cml_rewrite_registry_create(void);
void cml_rewrite_registry_free(CMLRewriteRegistry* reg);

int cml_rewrite_register(CMLRewriteRegistry* reg, CMLPatternNode* pattern,
                         CMLEmitFn emit, int priority, const char* name);

/* Apply all rewrite rules until convergence.
   Returns number of rewrites applied, or -1 on error. */
int cml_rewrite_apply(CMLRewriteRegistry* reg, CMLGraph_t ir, int max_iterations);

CMLRewriteRegistry* cml_rewrite_builtin_rules(void);

/* Remove nodes not reachable from outputs.
   Returns number of nodes removed, or -1 on error. */
int cml_rewrite_dce(CMLGraph_t ir);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_PATTERN_MATCHER_H */
