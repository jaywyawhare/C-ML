#ifndef CML_OPS_IR_SPEC_H
#define CML_OPS_IR_SPEC_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_SPEC_TENSOR,
    CML_SPEC_KERNEL,
    CML_SPEC_LINEAR,
    CML_SPEC_PROGRAM,
} CMLSpecLevel;

typedef struct CMLSpecError {
    int node_id;
    const char* message;
    CMLSpecLevel level;
} CMLSpecError;

typedef struct CMLSpecResult {
    bool valid;
    CMLSpecError* errors;
    int num_errors;
    int capacity;
} CMLSpecResult;

CMLSpecResult* cml_spec_validate(CMLGraph_t graph, CMLSpecLevel level);
void cml_spec_result_free(CMLSpecResult* result);
void cml_spec_result_print(const CMLSpecResult* result);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_SPEC_H */
