#ifndef CML_OPS_IR_HEURISTIC_OPT_H
#define CML_OPS_IR_HEURISTIC_OPT_H

#include "ops/ir/opt_transforms.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CML_HEURISTIC_DEFAULT_LOCAL_SIZE  256
#define CML_HEURISTIC_DEFAULT_VEC_WIDTH   4

typedef struct CMLHeuristicConfig {
    int max_local_size;
    int preferred_vec_width;
    bool use_local_memory;
} CMLHeuristicConfig;

struct LinearProgram;

CMLOptList* cml_heuristic_optimize(struct LinearProgram* prog);
void cml_heuristic_set_config(CMLHeuristicConfig* config);
CMLHeuristicConfig cml_heuristic_get_config(void);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_HEURISTIC_OPT_H */
