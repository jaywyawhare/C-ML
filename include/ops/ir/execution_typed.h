#ifndef CML_OPS_IR_EXECUTION_TYPED_H
#define CML_OPS_IR_EXECUTION_TYPED_H

#include "tensor/tensor.h"

struct IRNode;

/* Typed (non-float32) execution.
 *
 * The float32 executor reads every buffer as `float*`. This path covers the
 * same ops for every other dtype: kernels are written once against the
 * load/store accessors in tensor/dtype_access.h, so one body serves f64, the
 * half/fp8 formats and all integer widths.
 *
 * Returns 0 if the node was executed, -1 if no typed kernel applies. A -1 must
 * NOT fall through to the float32 path -- reinterpreting a non-f32 buffer as
 * float silently produces garbage, which is what this path exists to prevent. */
int cml_exec_typed(struct IRNode* node, Tensor* out);

/* True when `node` must be routed through cml_exec_typed rather than the f32
 * kernels, i.e. any participating buffer is not float32. Index/predicate ops
 * whose output dtype differs from f32 by design (argmax, comparisons) are
 * judged on their inputs, not their output. */
bool cml_exec_needs_typed(struct IRNode* node, Tensor* out);

#endif /* CML_OPS_IR_EXECUTION_TYPED_H */
