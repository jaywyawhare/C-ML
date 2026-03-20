#ifndef CML_OPS_IR_LATE_PASSES_H
#define CML_OPS_IR_LATE_PASSES_H

#include "ops/ir/linearize.h"

#ifdef __cplusplus
extern "C" {
#endif

int cml_devectorize(struct LinearProgram* prog);
int cml_expand_groups(struct LinearProgram* prog);
int cml_late_lower(struct LinearProgram* prog);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_LATE_PASSES_H */
