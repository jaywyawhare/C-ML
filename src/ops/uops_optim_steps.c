/*
 * In-graph optimizer steps: build UOP_SGD_STEP / UOP_ADAM_STEP nodes so the
 * parameter update can be scheduled and fused alongside the backward graph
 * (see FUSE_OPTIM). Split out of uops.c; shares the node-finishing helper via
 * ops/uops_internal.h.
 */
#include "ops/uops.h"
#include "ops/uops_internal.h"
#include "ops/ir/ir.h"
#include "ops/ir/internal.h"
#include "tensor/tensor.h"
#include "alloc/cml_allocator.h"

Tensor* uop_sgd_step(Tensor* param, Tensor* grad, Tensor* momentum_buf,
                     SgdStepParams* p) {
    if (!param || !grad || !p) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    SgdStepParams* cp = cml_malloc(sizeof(SgdStepParams));
    if (!cp) return NULL;
    *cp = *p;

    Tensor* inputs[3];
    int n_inputs = 2;
    inputs[0] = param;
    inputs[1] = grad;
    if (momentum_buf) {
        inputs[2] = momentum_buf;
        n_inputs  = 3;
    }

    if (cml_ir_add_uop(ir, UOP_SGD_STEP, inputs, n_inputs, cp) != 0) {
        cml_free(cp);
        return NULL;
    }

    return cml_uop_finish_source_node(ir, param->shape, param->ndim, param->dtype, param->device);
}

Tensor* uop_adam_step(Tensor* param, Tensor* grad, Tensor* exp_avg,
                      Tensor* exp_avg_sq, Tensor* max_exp_avg_sq,
                      AdamStepParams* p) {
    if (!param || !grad || !exp_avg || !exp_avg_sq || !p) return NULL;

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) return NULL;

    AdamStepParams* cp = cml_malloc(sizeof(AdamStepParams));
    if (!cp) return NULL;
    *cp = *p;

    Tensor* inputs[5];
    int n_inputs = 4;
    inputs[0] = param;
    inputs[1] = grad;
    inputs[2] = exp_avg;
    inputs[3] = exp_avg_sq;
    if (max_exp_avg_sq) {
        inputs[4] = max_exp_avg_sq;
        n_inputs  = 5;
    }

    if (cml_ir_add_uop(ir, UOP_ADAM_STEP, inputs, n_inputs, cp) != 0) {
        cml_free(cp);
        return NULL;
    }

    return cml_uop_finish_source_node(ir, param->shape, param->ndim, param->dtype, param->device);
}
