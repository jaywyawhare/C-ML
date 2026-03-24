

#ifndef CML_OPS_IR_VALIDATE_H
#define CML_OPS_IR_VALIDATE_H

#include "ops/ir/ir.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CML_VALID_OK                = 0,
    CML_VALID_NULL_GRAPH        = 1,
    CML_VALID_NULL_NODE         = 2,
    CML_VALID_CYCLE             = 3,  
    CML_VALID_MISSING_INPUT     = 4,  
    CML_VALID_TOO_MANY_INPUTS   = 5,
    CML_VALID_SHAPE_MISMATCH    = 6,
    CML_VALID_DTYPE_MISMATCH    = 7,
    CML_VALID_INVALID_AXIS      = 8,  
    CML_VALID_UNKNOWN_OP        = 9,
    CML_VALID_EMPTY_GRAPH       = 10,
    CML_VALID_DEAD_OUTPUT       = 11, 
    CML_VALID_INTERNAL          = 99,
} CMLValidateCode;

#define CML_VALIDATE_MSG_LEN 256

typedef struct CMLValidateDiag {
    CMLValidateCode code;
    char            message[CML_VALIDATE_MSG_LEN];
    int             node_index;   
} CMLValidateDiag;

typedef struct CMLValidateOpts {
    bool check_shapes;   
    bool check_dtypes;   
    bool check_cycles;   
    bool check_dead;     
    int  max_diags;      
} CMLValidateOpts;

CMLValidateOpts cml_validate_default_opts(void);

CMLValidateCode cml_validate_graph(CMLGraph_t ir,
                                   const CMLValidateOpts* opts,
                                   CMLValidateDiag* diags,
                                   int max_diag_count,
                                   int* num_diags_out);

void cml_validate_graph_or_die(CMLGraph_t ir);

void cml_validate_print_diags(const CMLValidateDiag* diags, int num_diags);

const char* cml_validate_code_str(CMLValidateCode code);

#ifdef __cplusplus
}
#endif

#endif 
